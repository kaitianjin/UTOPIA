"""The two networks the baselines add: regressors that predict their own variance.

:class:`HeteroscedasticRegressor` (``deep_ensemble``) adds a variance head to the pipeline's trunk
and trains under a Gaussian likelihood on the rescaled ``[0, 1]`` target;
:class:`NegativeBinomialRegressor` (``deep_ensemble_nb``) predicts a count mean and a dispersion on
the raw counts.  Both are for genes; every other baseline runs ``models/default.py`` unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.default import (
    AsymmetricMSELoss,
    ShiftedELU,
    _batches,
    _trunk,
    default_device,
    set_seeds,
)

#: Added to every predicted variance, on the rescaled ``[0, 1]`` count scale, so a member
#: that fits a unit exactly cannot send ``log sigma^2`` to minus infinity.
VAR_FLOOR = 1e-6


class HeteroscedasticRegressor(nn.Module):
    """The pipeline's trunk with two heads: a mean and a predictive variance."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        """Build the shared trunk, the mean head and the log-variance head."""
        super().__init__()
        self.trunk = _trunk(input_dim, 128, dropout)
        self.mean = nn.Sequential(nn.Linear(128, output_dim), ShiftedELU())
        self.logvar = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, var)`` for a batch of feature vectors."""
        hidden = self.trunk(x)
        return self.mean(hidden), F.softplus(self.logvar(hidden)) + VAR_FLOOR


def gaussian_nll(
    mu: torch.Tensor, var: torch.Tensor, y: torch.Tensor, beta: float = 0.5
) -> torch.Tensor:
    """Gaussian negative log-likelihood ``0.5 * (log var + (y - mu)^2 / var)``, averaged over units.

    With a nonzero ``beta`` each unit's loss is first scaled by a stop-gradient ``var^beta``
    (beta-NLL, Seitzer et al. 2022); ``beta = 0`` is plain NLL.
    """
    loss = 0.5 * (torch.log(var) + (y - mu) ** 2 / var)
    if beta:
        loss = loss * var.detach() ** beta
    return loss.mean()


def train_heteroscedastic(
    x: Any,
    y: Any,
    rows: np.ndarray,
    epochs: int = 20,
    warmup_epochs: int = 5,
    batch_size: int = 8096,
    learning_rate: float = 1e-3,
    dropout: float = 0.1,
    loss: str = "amse",
    beta: float = 2.0,
    nll_beta: float = 0.5,
    seed: int = 42,
) -> HeteroscedasticRegressor:
    """Train one ensemble member: warm up on the asymmetric loss, then beta-NLL.

    ``x`` and ``y`` are the ``(N, D)`` features and ``(N, G)`` rescaled targets, of which ``rows``
    are trained on; the first ``warmup_epochs`` of the ``epochs`` fit the mean alone.
    """
    device = default_device()
    set_seeds(seed)
    model = HeteroscedasticRegressor(x.shape[1], y.shape[1], dropout=dropout).to(device)
    warmup = AsymmetricMSELoss(beta=beta if loss == "amse" else 1.0)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = np.random.default_rng(seed)

    model.train()
    for epoch in range(epochs):
        warming = epoch < warmup_epochs
        for batch in _batches(len(rows), batch_size, generator):
            selected = rows[batch]
            features = torch.as_tensor(np.asarray(x[selected], dtype=np.float32), device=device)
            targets = torch.as_tensor(np.asarray(y[selected], dtype=np.float32), device=device)
            mu, var = model(features)
            optimiser.zero_grad()
            # The warm-up leaves the variance head out of the graph, so the mean fits
            # before the likelihood trades mean error against predicted spread.
            criterion = warmup(mu, targets) if warming else gaussian_nll(mu, var, targets, nll_beta)
            criterion.backward()
            optimiser.step()
    return model.eval()


def rebuild_member(
    network: type[nn.Module], state_dict: Any, input_dim: int, output_dim: int, dropout: float
) -> nn.Module:
    """Rebuild a stored member of class ``network`` from its weights, on the CPU in eval mode."""
    model = network(input_dim, output_dim, dropout=dropout)
    model.load_state_dict(state_dict)
    return model.eval()


def _kept_columns(
    mean: torch.Tensor, variance: torch.Tensor, columns: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the wanted output columns of a batch's mean and variance, as float32 arrays."""
    return (
        mean.index_select(1, columns).float().cpu().numpy(),
        variance.index_select(1, columns).float().cpu().numpy(),
    )


@torch.inference_mode()
def predict_heteroscedastic(
    model: HeteroscedasticRegressor,
    features: torch.Tensor,
    columns: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one member over a ``(B, D)`` batch and keep only the wanted output columns.

    Returns ``(mean, variance)`` as ``(B, len(columns))`` float32 arrays.
    """
    mu, var = model(features)
    return _kept_columns(mu, var, columns)


#: Smallest predicted count mean: ``log mu`` enters the likelihood, and a zero mean would
#: take it to minus infinity on the empty units that make up most of the training set.
MU_FLOOR = 1e-6
#: Smallest dispersion: as ``r`` falls the negative binomial piles mass on zero, and the
#: floor bounds that on a gene that is empty everywhere it was trained.
R_FLOOR = 1e-3
#: Largest dispersion, past which the fit is Poisson for these counts; in float32
#: ``lgamma(y + r) - lgamma(r)`` loses all precision once ``r`` passes about ``1e5``.
R_CEIL = 1e4


class NegativeBinomialRegressor(nn.Module):
    """The pipeline's trunk with two heads: a count mean and a dispersion.

    It predicts the raw count rather than its ``[0, 1]`` rescaling, so its output is already on the
    scale a target's truth is measured in.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        """Build the shared trunk, the count-mean head and the dispersion head."""
        super().__init__()
        self.trunk = _trunk(input_dim, 128, dropout)
        self.mean = nn.Linear(128, output_dim)
        self.dispersion = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, r)`` for a batch of feature vectors, both strictly positive."""
        hidden = self.trunk(x)
        mu = F.softplus(self.mean(hidden)) + MU_FLOOR
        dispersion = torch.clamp(F.softplus(self.dispersion(hidden)) + R_FLOOR, max=R_CEIL)
        return mu, dispersion


def negative_binomial_variance(mu: torch.Tensor, dispersion: torch.Tensor) -> torch.Tensor:
    """``Var = mu + mu^2 / r``, the variance the pair implies."""
    return mu + mu * mu / dispersion


def negative_binomial_nll(
    mu: torch.Tensor, dispersion: torch.Tensor, y: torch.Tensor, beta: float = 0.5
) -> torch.Tensor:
    """Mean negative-binomial negative log-likelihood (mean ``mu``, variance ``mu + mu^2 / r``).

    With a nonzero ``beta`` each unit's loss is scaled by a stop-gradient ``Var^beta`` divided by
    its batch mean, which reweights the units without changing the step size.
    """
    total = dispersion + mu
    log_total = torch.log(total)
    loss = -(
        torch.lgamma(y + dispersion)
        - torch.lgamma(dispersion)
        - torch.lgamma(y + 1.0)
        + dispersion * (torch.log(dispersion) - log_total)
        + y * (torch.log(mu) - log_total)
    )
    if beta:
        weight = negative_binomial_variance(mu, dispersion).detach() ** beta
        loss = loss * weight / weight.mean().clamp_min(1e-12)
    return loss.mean()


def poisson_nll(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Poisson negative log-likelihood ``mu - y * log(mu)``, averaged, up to a constant in ``mu``.

    It is the warm-up objective of the negative-binomial member, fitting the mean before the
    dispersion is trained.
    """
    return (mu - y * torch.log(mu)).mean()


def train_negative_binomial(
    x: Any,
    y: Any,
    rows: np.ndarray,
    epochs: int = 20,
    warmup_epochs: int = 5,
    batch_size: int = 8096,
    learning_rate: float = 1e-3,
    dropout: float = 0.1,
    loss: str = "amse",
    beta: float = 2.0,
    nll_beta: float = 0.5,
    seed: int = 42,
) -> NegativeBinomialRegressor:
    """Train one ``deep_ensemble_nb`` member: Poisson warm-up, then beta-NLL.

    ``y`` is the ``(N, G)`` raw count array; ``loss`` and ``beta`` are accepted to match
    :func:`train_heteroscedastic` and ignored.
    """
    device = default_device()
    set_seeds(seed)
    model = NegativeBinomialRegressor(x.shape[1], y.shape[1], dropout=dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = np.random.default_rng(seed)

    model.train()
    for epoch in range(epochs):
        warming = epoch < warmup_epochs
        for batch in _batches(len(rows), batch_size, generator):
            selected = rows[batch]
            features = torch.as_tensor(np.asarray(x[selected], dtype=np.float32), device=device)
            targets = torch.as_tensor(np.asarray(y[selected], dtype=np.float32), device=device)
            mu, dispersion = model(features)
            optimiser.zero_grad()
            criterion = (
                poisson_nll(mu, targets)
                if warming
                else negative_binomial_nll(mu, dispersion, targets, nll_beta)
            )
            criterion.backward()
            optimiser.step()
    return model.eval()


@torch.inference_mode()
def predict_negative_binomial(
    model: NegativeBinomialRegressor,
    features: torch.Tensor,
    columns: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one NB member over a batch and return ``(mean, variance)`` in counts.

    Same contract as :func:`predict_heteroscedastic`: ``(mu, mu + mu^2 / r)`` for the kept
    columns, as ``(B, len(columns))`` float32 arrays.
    """
    mu, dispersion = model(features)
    return _kept_columns(mu, negative_binomial_variance(mu, dispersion), columns)
