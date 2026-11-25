"""The built-in feed-forward predictor for genes and cell types.

Both heads share one bottleneck trunk (``D -> 128 -> 64 -> 128 -> out``). The gene head ends in an
ELU trained with an asymmetric squared error that penalises under-prediction more; the cell-type
head ends in a softmax trained with cross-entropy. Training and prediction index ``x`` and ``y``
by row, so they may be in-memory arrays or memory-mapped ``.npy`` files.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def set_seeds(seed: int) -> None:
    """Seed Python, numpy and torch, and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def default_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShiftedELU(nn.Module):
    """ELU offset by a small positive constant, keeping outputs above zero."""

    def __init__(self, alpha: float = 0.01, beta: float = 0.01) -> None:
        """Build an ELU with slope ``alpha`` below zero, shifted up by ``beta``."""
        super().__init__()
        self.activation = nn.ELU(alpha=alpha)
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shifted activation."""
        return self.activation(x) + self.beta


class AsymmetricMSELoss(nn.Module):
    """Root mean squared error weighting under-prediction ``beta`` times more heavily.

    ``beta = 1`` recovers plain RMSE.
    """

    def __init__(self, beta: float = 2.0) -> None:
        """Store the weight ``beta`` given to under-prediction."""
        super().__init__()
        self.beta = beta

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss between a prediction and its target."""
        difference = target - prediction
        weight = torch.where(difference > 0, self.beta, 1.0)
        return torch.sqrt(torch.mean(weight * difference**2))


def _trunk(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    """Build the shared bottleneck trunk plus a linear output layer."""
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(dropout),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(dropout),
        nn.Linear(128, output_dim),
    )


class FeedForwardRegressor(nn.Module):
    """Predicts a non-negative value per gene from a superpixel's features."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        """Build the trunk followed by the shifted ELU."""
        super().__init__()
        self.network = nn.Sequential(_trunk(input_dim, output_dim, dropout), ShiftedELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict scaled expression for a batch of feature vectors."""
        return self.network(x)


class FeedForwardClassifier(nn.Module):
    """Predicts a cell-type probability vector from a superpixel's features."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3) -> None:
        """Build the trunk, whose linear output layer gives the logits."""
        super().__init__()
        self.network = _trunk(input_dim, output_dim, dropout)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return the logits, which the cross-entropy loss is trained on."""
        return self.network(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a cell-type probability vector for a batch of feature vectors."""
        return torch.softmax(self.logits(x), dim=1)


def enable_dropout(model: nn.Module) -> None:
    """Put only the dropout layers of an evaluated model back into training mode.

    Batch normalisation keeps its running statistics, so Monte-Carlo dropout passes differ only by
    the dropout masks and not by which units share a batch.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def rebuild(
    modality: str, state_dict: Any, input_dim: int, output_dim: int, dropout: float
) -> nn.Module:
    """Rebuild a stored draw model from its weights, on the CPU in eval mode.

    ``modality`` is ``"gene"`` for the regressor and ``"cell_type"`` for the classifier.
    """
    builder = FeedForwardRegressor if modality == "gene" else FeedForwardClassifier
    model = builder(input_dim, output_dim, dropout=dropout)
    model.load_state_dict(state_dict)
    return model.eval()


def _batches(n_rows: int, batch_size: int, generator: np.random.Generator) -> Any:
    """Yield sorted row-index batches covering a shuffled epoch.

    Indices are sorted within a batch so that reads from a memory-mapped array stay sequential.
    """
    order = generator.permutation(n_rows)
    for start in range(0, n_rows, batch_size):
        batch = order[start : start + batch_size]
        if len(batch) > 1:
            yield np.sort(batch)


def train_regressor(
    x: Any,
    y: Any,
    rows: np.ndarray,
    epochs: int = 20,
    batch_size: int = 8096,
    learning_rate: float = 1e-3,
    dropout: float = 0.1,
    loss: str = "amse",
    beta: float = 2.0,
    seed: int = 42,
) -> FeedForwardRegressor:
    """Train the gene regressor on the rows ``rows`` of features ``x`` and scaled expression ``y``.

    ``loss`` is ``"amse"`` for the asymmetric loss or ``"rmse"`` for plain RMSE; the model is
    returned on the default device in eval mode.
    """
    device = default_device()
    set_seeds(seed)
    model = FeedForwardRegressor(x.shape[1], y.shape[1], dropout=dropout).to(device)
    criterion = AsymmetricMSELoss(beta=beta) if loss == "amse" else AsymmetricMSELoss(beta=1.0)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = np.random.default_rng(seed)

    model.train()
    for _ in range(epochs):
        for batch in _batches(len(rows), batch_size, generator):
            selected = rows[batch]
            features = torch.as_tensor(np.asarray(x[selected], dtype=np.float32), device=device)
            targets = torch.as_tensor(np.asarray(y[selected], dtype=np.float32), device=device)
            optimiser.zero_grad()
            criterion(model(features), targets).backward()
            optimiser.step()
    return model.eval()


def train_classifier(
    x: Any,
    labels: Any,
    rows: np.ndarray,
    n_classes: int,
    epochs: int = 20,
    batch_size: int = 8096,
    learning_rate: float = 1e-3,
    dropout: float = 0.3,
    seed: int = 42,
) -> FeedForwardClassifier:
    """Train the cell-type classifier on the rows ``rows`` of features ``x`` and integer ``labels``.

    The model is returned on the default device in eval mode.
    """
    device = default_device()
    set_seeds(seed)
    model = FeedForwardClassifier(x.shape[1], n_classes, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = np.random.default_rng(seed)

    model.train()
    for _ in range(epochs):
        for batch in _batches(len(rows), batch_size, generator):
            selected = rows[batch]
            features = torch.as_tensor(np.asarray(x[selected], dtype=np.float32), device=device)
            targets = torch.as_tensor(np.asarray(labels[selected], dtype=np.int64), device=device)
            optimiser.zero_grad()
            loss = criterion(model.logits(features), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
    return model.eval()


@torch.inference_mode()
def predict(model: nn.Module, x: Any, rows: np.ndarray, batch_size: int = 32768) -> np.ndarray:
    """Run a trained model over the rows ``rows`` of ``x``.

    Returns ``(len(rows), out_dim)`` float32 predictions in the order of ``rows``.
    """
    device = default_device()
    model = model.to(device).eval()
    outputs = []
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        features = torch.as_tensor(np.asarray(x[selected], dtype=np.float32), device=device)
        outputs.append(model(features).float().cpu().numpy())
    if not outputs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(outputs)
