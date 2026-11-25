"""Pure statistics behind the UQ baselines.

Mixing members by the law of total variance, coarsening to blocks, Gaussian and negative-binomial
tail p-values and intervals, the Sun & Cai local-fdr step-up, the Poisson-binomial tail, and
interval coverage.  Every function takes and returns numpy arrays -- no config, no workspace, no
file formats -- so each is testable without a pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import NormalDist
from typing import NamedTuple

import numpy as np
from scipy.special import ndtr
from scipy.stats import nbinom, poisson

from ..aggregate import group_sum

#: Smallest predictive variance any p-value is taken against: at zero variance
#: ``(nu - mu) / sd`` is an infinity whose sign alone would decide the call.
VARIANCE_FLOOR = 1e-12


def mixture(
    mu: np.ndarray, var: float | np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Combine ``(M, ...)`` member means into one predictive mean and variance.

    By the law of total variance, ``mu_bar = mean_m(mu_m)`` and
    ``var_bar = mean_m(var_m) + Var_m(mu_m)``, the first (aleatoric) term being zero when ``var``
    is None.
    """
    mu = np.asarray(mu, dtype=np.float64)
    epistemic = mu.var(axis=0)
    if var is None:
        aleatoric = np.zeros_like(epistemic)
    else:
        aleatoric = np.broadcast_to(np.asarray(var, dtype=np.float64), mu.shape).mean(axis=0)
    return mu.mean(axis=0), aleatoric + epistemic


def block_sums(values: np.ndarray, block_ids: np.ndarray, n_blocks: int) -> np.ndarray:
    """Sum each member's ``(M, N)`` per-unit values within every block, giving ``(M, n_blocks)``.

    When the grouping is the identity the values are returned as they are rather than re-derived,
    so treat the result as read-only.
    """
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    block_ids = np.asarray(block_ids, dtype=np.int64)
    n_blocks = int(n_blocks)
    if n_blocks == values.shape[1] and np.array_equal(block_ids, np.arange(n_blocks)):
        return values
    return np.stack([group_sum(member, block_ids, n_blocks) for member in values])


def residual_variance(
    pred: np.ndarray,
    truth: np.ndarray,
    groups: np.ndarray | None = None,
    n_groups: int | None = None,
    min_units: int = 200,
) -> float | np.ndarray:
    """Observation noise as the mean squared held-out residual, ``mean((truth - pred)^2)``.

    Returns one float, or with ``groups`` and ``n_groups`` an ``(n_groups,)`` array per cluster in
    which a cluster of fewer than ``min_units`` units takes the marginal value.
    """
    squared = (np.asarray(truth, dtype=np.float64) - np.asarray(pred, dtype=np.float64)) ** 2
    marginal = float(squared.mean()) if len(squared) else 0.0
    if groups is None:
        return marginal

    groups = np.asarray(groups, dtype=np.int64)
    n_groups = int(n_groups)
    out = np.full(n_groups, marginal, dtype=np.float64)
    counts = np.bincount(groups[groups >= 0], minlength=n_groups)[:n_groups]
    totals = np.bincount(
        groups[groups >= 0], weights=squared[groups >= 0], minlength=n_groups
    )[:n_groups]
    enough = counts >= min_units
    out[enough] = totals[enough] / counts[enough]
    return out


def gaussian_pvals(mu_bar: np.ndarray, sd_bar: np.ndarray, nu: float) -> np.ndarray:
    """Gaussian tail p-value against ``H0: y <= nu``: ``p = Phi((nu - mu_bar) / sd_bar)``.

    Small values are evidence that ``y > nu``.
    """
    sd = np.sqrt(np.maximum(np.asarray(sd_bar, dtype=np.float64) ** 2, VARIANCE_FLOOR))
    return ndtr((float(nu) - np.asarray(mu_bar, dtype=np.float64)) / sd)


def _nb_shape(
    mu: np.ndarray, var: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moment-match a negative binomial to ``(mu, var)`` per unit: ``r = mu^2 / (var - mu)``.

    Returns ``(mu, r, overdispersed)`` with ``mu`` clipped at zero; where ``var`` does not exceed
    ``mu`` the flag is False, ``r`` is infinite and the caller uses a Poisson of the same mean.
    """
    mu = np.maximum(np.asarray(mu, dtype=np.float64), 0.0)
    var = np.asarray(var, dtype=np.float64)
    # A relative margin takes the Poisson limit slightly early: as var approaches mu from
    # above, r runs to infinity and nbinom loses the tail to rounding.
    overdispersed = var > mu * (1.0 + 1e-6) + 1e-12
    spread = np.where(overdispersed, var - mu, 1.0)
    dispersion = np.where(overdispersed, mu * mu / spread, np.inf)
    return mu, dispersion, overdispersed


def nb_pvals(mu_bar: np.ndarray, var_bar: np.ndarray, nu: float) -> np.ndarray:
    """Count-scale tail p-value against ``H0: y <= nu``: ``p = P(Y <= floor(nu))``.

    ``Y`` is the negative binomial moment-matched to ``(mu_bar, var_bar)``, or a Poisson of mean
    ``mu_bar`` where the variance does not exceed the mean.
    """
    mu, dispersion, overdispersed = _nb_shape(mu_bar, var_bar)
    bound = np.floor(float(nu))
    if bound < 0:
        return np.zeros_like(mu)
    out = poisson.cdf(bound, np.maximum(mu, 0.0))
    if np.any(overdispersed):
        rows = np.flatnonzero(overdispersed)
        shape = dispersion[rows]
        out[rows] = nbinom.cdf(bound, shape, shape / (shape + mu[rows]))
    return np.clip(np.nan_to_num(out, nan=1.0), 0.0, 1.0)


def nb_interval(
    mu_bar: np.ndarray, var_bar: np.ndarray, coverage: float
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-tailed central interval of the count predictive :func:`nb_pvals` takes its tail from.

    Returns ``(lower, upper)``, the ``(1 - coverage) / 2`` and ``(1 + coverage) / 2`` quantiles,
    which are non-negative whole counts.
    """
    mu, dispersion, overdispersed = _nb_shape(mu_bar, var_bar)
    tail = 0.5 * (1.0 - float(coverage))
    lower = poisson.ppf(tail, mu)
    upper = poisson.ppf(1.0 - tail, mu)
    if np.any(overdispersed):
        rows = np.flatnonzero(overdispersed)
        shape = dispersion[rows]
        success = shape / (shape + mu[rows])
        lower[rows] = nbinom.ppf(tail, shape, success)
        upper[rows] = nbinom.ppf(1.0 - tail, shape, success)
    lower = np.nan_to_num(lower, nan=0.0, posinf=0.0, neginf=0.0)
    upper = np.nan_to_num(upper, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(lower, 0.0), np.maximum(upper, lower)


def sort_draws(draws: np.ndarray) -> np.ndarray:
    """Lay ``(M, N)`` draws out as ``(N, M)`` sorted along the member axis.

    This is the layout :func:`draw_interval` takes when ``presorted`` is set.
    """
    return np.sort(np.ascontiguousarray(np.asarray(draws, dtype=np.float64).T), axis=-1)


def _order_statistic(ordered: np.ndarray, prob: float) -> np.ndarray:
    """The ``prob`` quantile along the last axis of already-sorted draws, linearly interpolated.

    Same rule as ``numpy.quantile``'s default: position ``prob * (M - 1)``, interpolating between
    the two neighbouring order statistics.
    """
    members = ordered.shape[-1]
    position = float(prob) * (members - 1)
    low = int(np.floor(position))
    high = min(low + 1, members - 1)
    fraction = position - low
    if fraction <= 0.0:
        return np.array(ordered[..., low])
    return ordered[..., low] * (1.0 - fraction) + ordered[..., high] * fraction


def draw_interval(
    draws: np.ndarray, coverage: float, presorted: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-tailed empirical interval of the draws themselves, with no distributional assumption.

    Returns ``(lower, upper)``, each unit's ``(1 - coverage) / 2`` and ``(1 + coverage) / 2``
    quantiles over its M draws; ``draws`` is ``(M, N)``, or :func:`sort_draws` output when
    ``presorted`` is set.
    """
    ordered = draws if presorted else sort_draws(draws)
    tail = 0.5 * (1.0 - float(coverage))
    lower = _order_statistic(ordered, tail)
    upper = _order_statistic(ordered, 1.0 - tail)
    return lower, np.maximum(upper, lower)


def sun_cai_adjust(lfdr: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """The smallest nominal FDR at which the Sun & Cai step-up selects each unit.

    That is the running mean of the local fdr values sorted ascending (within each group), given
    back in the original order -- the local-fdr counterpart of a BH-adjusted p-value.
    """
    lfdr = np.asarray(lfdr, dtype=np.float64)
    out = np.ones(len(lfdr), dtype=np.float64)
    if groups is None:
        row_sets = [np.arange(len(lfdr))]
    else:
        groups = np.asarray(groups)
        row_sets = [np.flatnonzero(groups == value) for value in np.unique(groups)]
    for rows in row_sets:
        values = lfdr[rows]
        order = np.argsort(values, kind="stable")
        running = np.cumsum(values[order]) / np.arange(1, len(values) + 1)
        adjusted = np.empty(len(values), dtype=np.float64)
        adjusted[order] = np.clip(running, 0.0, 1.0)
        out[rows] = adjusted
    return out


def poisson_binomial_tail(p: np.ndarray, nu: float) -> np.ndarray:
    """``P(sum_i Z_i <= nu)`` for independent ``Z_i ~ Bernoulli(p_i)`` along the last axis.

    Exact, by a dynamic program over the counts ``0..floor(nu)``; a block padded with ``p = 0``
    keeps the tail of its own units.
    """
    p = np.asarray(p, dtype=np.float64)
    ceiling = int(np.floor(float(nu)))
    if ceiling < 0:
        return np.zeros(p.shape[:-1], dtype=np.float64)
    counts = np.zeros(p.shape[:-1] + (ceiling + 1,), dtype=np.float64)
    counts[..., 0] = 1.0
    for unit in range(p.shape[-1]):
        hit = p[..., unit : unit + 1]
        shifted = np.zeros_like(counts)
        shifted[..., 1:] = counts[..., :-1] * hit
        counts = counts * (1.0 - hit) + shifted
    return counts.sum(axis=-1)


class BlockLayout(NamedTuple):
    """Where each unit sits inside the padded block matrix :func:`pad_blocks` builds."""

    sorted_ids: np.ndarray  # (N,) block ids in block order
    ranks: np.ndarray  # (N,) position of each unit within its own block, in block order
    order: np.ndarray  # (N,) permutation putting the units in block order
    n_blocks: int
    width: int  # units in the largest block


def block_layout(block_ids: np.ndarray, n_blocks: int, counts: np.ndarray) -> BlockLayout:
    """Work out the padded layout of one grouping, given the units per block in ``counts``."""
    block_ids = np.asarray(block_ids, dtype=np.int64)
    order = np.argsort(block_ids, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    ranks = np.arange(len(block_ids)) - np.repeat(starts, counts)
    return BlockLayout(block_ids[order], ranks, order, int(n_blocks), int(counts.max()))


def pad_blocks(values: np.ndarray, layout: BlockLayout) -> np.ndarray:
    """Lay ragged ``(N,)`` per-unit values out as one ``(n_blocks, widest block)`` array.

    Short blocks are padded with zeros, which :func:`poisson_binomial_tail` reads as a Bernoulli
    that never fires.
    """
    out = np.zeros((layout.n_blocks, layout.width), dtype=np.float64)
    out[layout.sorted_ids, layout.ranks] = np.asarray(values, dtype=np.float64)[layout.order]
    return out


def _coverage_record(
    group: int | None, coverage: float, inside: np.ndarray, width: np.ndarray
) -> dict:
    """One record of realised coverage and mean width over the units it is handed."""
    return {
        "group": group,
        "nominal": float(coverage),
        "coverage": float(inside.mean()) if len(inside) else 0.0,
        "width": float(width.mean()) if len(width) else 0.0,
        "n_units": int(len(inside)),
    }


def coverage_and_width(
    mu_bar: np.ndarray,
    sd_bar: np.ndarray,
    truth: np.ndarray,
    coverage: float,
    groups: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> Sequence[dict]:
    """Realised coverage and mean width of a central predictive interval.

    The interval is ``mu_bar -+ z * sd_bar`` unless ``bounds`` gives ``(lower, upper)``; returns
    one record over all units (``group`` None), then one per group when ``groups`` is given.
    """
    truth = np.asarray(truth, dtype=np.float64)
    if bounds is None:
        z = NormalDist().inv_cdf(0.5 * (1.0 + float(coverage)))
        mu_bar = np.asarray(mu_bar, dtype=np.float64)
        half = z * np.asarray(sd_bar, dtype=np.float64)
        lower, upper = mu_bar - half, mu_bar + half
    else:
        lower = np.asarray(bounds[0], dtype=np.float64)
        upper = np.asarray(bounds[1], dtype=np.float64)
    inside = (truth >= lower) & (truth <= upper)
    width = upper - lower
    records = [_coverage_record(None, coverage, inside, width)]
    if groups is None:
        return records
    groups = np.asarray(groups)
    for value in np.unique(groups):
        rows = groups == value
        records.append(_coverage_record(int(value), coverage, inside[rows], width[rows]))
    return records
