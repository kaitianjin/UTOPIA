"""Split-conformal p-values for one-sided "is the signal above the null" tests.

The null hypothesis at a spatial unit is ``H0: y <= null``, so a small p-value is evidence that the
value exceeds the null. Calibration units that satisfy the null score ``-yhat`` and the others
``+inf``; a test unit's p-value is the calibration rank of its own ``-yhat``, by default within its
histology cluster and after subtracting an estimated calibration-to-test prediction shift.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

#: How the per-quantile shifts are reduced to one tolerance: ``max`` (the default, conservative)
#: absorbs the largest shift seen in the null part of the distribution, ``mean`` the average one.
_SHIFT_AGGREGATES = {"max": np.max, "mean": np.mean}

#: Quantile grids used to estimate a prediction shift between calibration and test.
_SHIFT_QUANTILES: dict[str, np.ndarray] = {
    "weak": np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]),
    "medium": np.array([0.001, 0.005, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.995, 0.999]),
    "strong": np.array(
        [0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.995, 0.999]
    ),
}


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values of one family, in the original order."""
    count = len(pvals)
    if count == 0:
        return np.zeros(0, dtype=float)
    order = np.argsort(pvals)
    scaled = count * np.asarray(pvals, dtype=float)[order] / np.arange(1, count + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = np.clip(monotone, 0.0, 1.0)
    return adjusted


def bh_adjust_grouped(pvals: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """Benjamini-Hochberg adjustment taken separately within each group.

    ``groups=None`` adjusts all the p-values as one family.
    """
    if groups is None:
        return bh_adjust(pvals)
    adjusted = np.ones(len(pvals), dtype=float)
    groups = np.asarray(groups)
    for group in np.unique(groups):
        rows = groups == group
        adjusted[rows] = bh_adjust(np.asarray(pvals, dtype=float)[rows])
    return adjusted


def empirical_pvals(
    test_scores: np.ndarray, calib_scores: np.ndarray, tolerance: float = 0.0
) -> np.ndarray:
    """Rank test scores against calibration scores: ``p = (#{calib < test} + 1) / (N + 1)``.

    Only calibration scores strictly below a test score are counted, after ``tolerance`` has been
    added to them.
    """
    ordered = np.sort(np.asarray(calib_scores, dtype=float) + tolerance)
    below = np.searchsorted(ordered, np.asarray(test_scores, dtype=float), side="left")
    return (below + 1) / (len(ordered) + 1)


def clip_scores(y: np.ndarray, yhat: np.ndarray, null: float) -> np.ndarray:
    """Nonconformity score for the one-sided test ``H0: y <= null``.

    Units satisfying the null score ``-yhat``; units violating it score ``+inf``, so that they
    cannot make a test unit look extreme by comparison.
    """
    y = np.asarray(y, dtype=float)
    return np.where(y <= null, -np.asarray(yhat, dtype=float), np.inf)


def conformal_pvals(
    calib_y: np.ndarray,
    calib_yhat: np.ndarray,
    test_yhat: np.ndarray,
    null: float,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Marginal split-conformal p-values against ``H0: y <= null``, one per test unit.

    Small values are evidence that ``y > null``; ``tolerance`` is the batch-effect shift added to
    the calibration scores.
    """
    calib_scores = clip_scores(calib_y, calib_yhat, null)
    # A test unit is scored as if it satisfied the null, so its score is ``-yhat``.
    test_scores = -np.asarray(test_yhat, dtype=float)
    return empirical_pvals(test_scores, calib_scores, tolerance=tolerance)


def conformal_marginal_levels(
    calib_y: np.ndarray,
    calib_yhat: np.ndarray,
    test_yhat: np.ndarray,
    null: float,
    bh_groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interval-inversion confidence levels from two-sided, unclustered split conformal.

    The level is the empirical CDF of the calibration ``|y - yhat|`` at ``yhat - null``, and 0 when
    the prediction does not exceed the null; returns ``(level, 1 - level, BH-adjusted 1 - level)``.
    """
    scores = np.sort(np.abs(np.asarray(calib_y, dtype=float) - np.asarray(calib_yhat, dtype=float)))
    gap = np.asarray(test_yhat, dtype=float) - null
    level = np.searchsorted(scores, gap, side="right") / len(scores)
    level[gap <= 0.0] = 0.0
    # A large level is strong evidence, the opposite of a p-value, so BH is applied to 1 - level.
    pvals = 1.0 - level
    return level, pvals, bh_adjust_grouped(pvals, bh_groups)


def batch_effect_tolerance(
    calib_yhat: np.ndarray,
    calib_y: np.ndarray,
    test_yhat: np.ndarray,
    null: float,
    magnitude: str | float = "medium",
    aggregate: str = "max",
    min_null_fraction: float = 0.1,
) -> float:
    """Batch-effect tolerance: minus the shift of test predictions above calibration predictions.

    The shift is measured over the lower part of both distributions, where the null is expected to
    hold, and is 0 unless more than ``min_null_fraction`` of the calibration units satisfy the null.
    """
    # A numeric magnitude is used as the tolerance itself.
    if not isinstance(magnitude, str):
        return float(magnitude)
    if magnitude not in _SHIFT_QUANTILES:
        raise ValueError(f"magnitude must be a number or one of {sorted(_SHIFT_QUANTILES)}")
    if aggregate not in _SHIFT_AGGREGATES:
        raise ValueError(f"aggregate must be one of {sorted(_SHIFT_AGGREGATES)}")

    null_fraction = float(np.mean(np.asarray(calib_y, dtype=float) <= null))
    if null_fraction <= min_null_fraction:
        return 0.0

    calib_sorted = np.sort(np.asarray(calib_yhat, dtype=float))
    test_sorted = np.sort(np.asarray(test_yhat, dtype=float))
    calib_lower = calib_sorted[calib_sorted <= np.quantile(calib_sorted, null_fraction)]
    test_lower = test_sorted[test_sorted <= np.quantile(test_sorted, null_fraction)]
    if len(calib_lower) == 0 or len(test_lower) == 0:
        return 0.0

    grid = _SHIFT_QUANTILES[magnitude]
    shifts = np.quantile(test_lower, grid) - np.quantile(calib_lower, grid)
    return -float(_SHIFT_AGGREGATES[aggregate](np.clip(shifts, 0.0, None)))


def conformal_cluster_pvals(
    calib_y: np.ndarray,
    calib_yhat: np.ndarray,
    calib_cluster: np.ndarray,
    test_yhat: np.ndarray,
    test_cluster: np.ndarray,
    null: float,
    clusters: Iterable[int] | None = None,
    cluster_sizes: Sequence[int] | None = None,
    min_calib_units: int | None = None,
    correct_batch_effect: bool = True,
    magnitude: str | float = "medium",
    aggregate: str = "max",
    min_null_fraction: float = 0.1,
    bh_groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster-conditional (Mondrian) split-conformal p-values against ``H0: y <= null``.

    Returns the raw p-values and their Benjamini-Hochberg adjustment within each cluster (or each
    cluster x ``bh_groups`` pair); a cluster with too few calibration units is left at ``p = 1``.
    """
    calib_cluster = np.asarray(calib_cluster)
    test_cluster = np.asarray(test_cluster)
    pvals = np.ones(len(test_yhat), dtype=float)
    pvals_adj = np.ones(len(test_yhat), dtype=float)
    labels = np.unique(calib_cluster) if clusters is None else np.asarray(list(clusters))

    for cluster in labels:
        calib_rows = calib_cluster == cluster
        test_rows = test_cluster == cluster
        n_calib, n_test = int(calib_rows.sum()), int(test_rows.sum())
        if n_test == 0:
            continue

        # A cluster is calibrated only when enough of it was sampled: by default at least 200
        # units and 5% of the cluster. With no calibration unit at all it always stays at p = 1.
        size = int(cluster_sizes[cluster]) if cluster_sizes is not None else n_calib + n_test
        required = (
            min_calib_units if min_calib_units is not None else max(200, int(0.05 * size))
        )
        if n_calib == 0 or n_calib < required:
            continue

        cluster_calib_y = np.asarray(calib_y)[calib_rows]
        cluster_calib_yhat = np.asarray(calib_yhat)[calib_rows]
        cluster_test_yhat = np.asarray(test_yhat)[test_rows]

        tolerance = 0.0
        if correct_batch_effect:
            tolerance = batch_effect_tolerance(
                cluster_calib_yhat,
                cluster_calib_y,
                cluster_test_yhat,
                null,
                magnitude,
                aggregate=aggregate,
                min_null_fraction=min_null_fraction,
            )

        raw = conformal_pvals(
            cluster_calib_y, cluster_calib_yhat, cluster_test_yhat, null, tolerance=tolerance
        )
        pvals[test_rows] = raw
        groups = None if bh_groups is None else np.asarray(bh_groups)[test_rows]
        pvals_adj[test_rows] = bh_adjust_grouped(raw, groups)

    return pvals, pvals_adj


def _log10(value: float | np.ndarray) -> float | np.ndarray:
    """Base-10 logarithm as ``log(value) / log(10)``; ``np.log10`` can differ in the last bit."""
    return np.log(value) / np.log(10.0)


def pvals_to_confidence(pvals: np.ndarray, p_min: float = 0.01, p_max: float = 0.5) -> np.ndarray:
    """Map p-values onto a 0-to-1 display scale, logarithmic between the two anchors.

    ``p_min`` and anything smaller maps to 1; ``p_max`` and anything larger maps to 0.
    """
    # The sampling methods report a count out of K draws, so an exact zero is attainable;
    # flooring it keeps the logarithm finite and maps it to 1.
    floored = np.maximum(np.asarray(pvals, dtype=float), np.finfo(float).tiny)
    scores = (-_log10(floored) + _log10(p_max)) / (-_log10(p_min) + _log10(p_max))
    return np.clip(scores, 0.0, 1.0)


def fdp_curve(
    pvals: np.ndarray,
    truth: np.ndarray,
    null: float,
    fdr_grid: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Realised false-discovery proportion as a function of the nominal FDR.

    A unit is a discovery when its adjusted p-value is at most the nominal level, and a true one
    when ``truth > null``; returns ``(fdr, fdp, n_discoveries)``, one entry per level.
    """
    grid = np.arange(0.01, 1.0, 0.01) if fdr_grid is None else np.asarray(fdr_grid, dtype=float)
    pvals = np.asarray(pvals, dtype=float)
    positives = np.asarray(truth, dtype=float) > null

    # Sorting once turns the whole curve into two cumulative counts, which avoids building a
    # (len(grid), n_units) boolean array -- gigabytes on a whole slide.
    order = np.argsort(pvals, kind="stable")
    sorted_pvals = pvals[order]
    cumulative_true = np.concatenate(([0], np.cumsum(positives[order])))
    n_discoveries = np.searchsorted(sorted_pvals, grid, side="right")
    n_true = cumulative_true[n_discoveries]
    fdp = np.where(n_discoveries == 0, 0.0, 1.0 - n_true / np.maximum(n_discoveries, 1))
    return grid, fdp, n_discoveries
