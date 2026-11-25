"""Uncertainty quantification, as pure array functions.

Every function takes numpy arrays and returns numpy arrays -- no configuration, workspaces or file
formats -- so the machinery can be reused directly from an analysis script:

    from utopia.uncertainty import DrawAccumulator, conformal_cluster_pvals

    pvals, pvals_adj = conformal_cluster_pvals(calib_y, calib_yhat, calib_cluster,
                                               test_yhat, test_cluster, null=0.05)
    accumulator = DrawAccumulator(null=0.05, n_units=draws.shape[1])
    for draw in draws:                              # (K, N) stack of draws
        accumulator.update(draw)
    result = accumulator.result()

``mc_dropout``, ``ensemble`` and ``resampling`` use :mod:`~utopia.uncertainty.sampling`;
``conformal_marginal`` is :func:`conformal_marginal_levels` and ``UTOPIA`` is
:func:`conformal_cluster_pvals`.
"""

from .base import UncertaintyResult
from .conformal import (
    batch_effect_tolerance,
    bh_adjust,
    bh_adjust_grouped,
    clip_scores,
    conformal_cluster_pvals,
    conformal_marginal_levels,
    conformal_pvals,
    empirical_pvals,
    fdp_curve,
    pvals_to_confidence,
)
from .sampling import DrawAccumulator

__all__ = [
    "DrawAccumulator",
    "UncertaintyResult",
    "batch_effect_tolerance",
    "bh_adjust",
    "bh_adjust_grouped",
    "clip_scores",
    "conformal_cluster_pvals",
    "conformal_marginal_levels",
    "conformal_pvals",
    "empirical_pvals",
    "fdp_curve",
    "pvals_to_confidence",
]
