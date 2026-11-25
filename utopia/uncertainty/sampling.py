"""Uncertainty from a sample of predictive draws.

``mc_dropout``, ``ensemble`` and ``resampling`` differ only in where their K draws come from. The
p-value against ``H0: y <= null`` is the fraction of draws that fail to exceed the null,
``p = 1 - #{draw > null} / K``, so it measures the spread of the model's own predictions, not
their agreement with held-out data.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np

from .base import UncertaintyResult
from .conformal import bh_adjust_grouped, pvals_to_confidence


@dataclass
class DrawAccumulator:
    """Sufficient statistics of a sample of predictive draws, accumulated one draw at a time."""

    null: float  # null value the draws are counted against, as each draw arrives
    n_units: int  # number of spatial units a draw covers
    n_draws: int = field(default=0, init=False)  # draws seen so far
    total: np.ndarray = field(init=False)  # (n_units,) running sum of the draws
    total_sq: np.ndarray = field(init=False)  # (n_units,) running sum of the squared draws
    n_above: np.ndarray = field(init=False)  # (n_units,) number of draws that exceeded the null

    def __post_init__(self) -> None:
        """Start every running statistic at zero."""
        self.total = np.zeros(self.n_units, dtype=np.float64)
        self.total_sq = np.zeros(self.n_units, dtype=np.float64)
        self.n_above = np.zeros(self.n_units, dtype=np.int32)

    def update(self, values: np.ndarray) -> None:
        """Fold one draw's per-unit values into the statistics."""
        values = np.asarray(values, dtype=np.float64)
        self.total += values
        self.total_sq += values * values
        self.n_above += values > self.null
        self.n_draws += 1

    @property
    def mean(self) -> np.ndarray:
        """``(n_units,)`` mean over draws; the point prediction of the method."""
        return self.total / self.n_draws

    @property
    def std(self) -> np.ndarray:
        """``(n_units,)`` standard deviation over draws, with the ``1/K`` divisor."""
        variance = self.total_sq / self.n_draws - self.mean**2
        return np.sqrt(np.maximum(variance, 0.0))

    @property
    def pvals(self) -> np.ndarray:
        """``(n_units,)`` p-values ``1 - #{draw > null} / K`` against ``H0: y <= null``."""
        return 1.0 - self.n_above / self.n_draws

    def interval(self, coverage: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian predictive interval ``(lower, upper)`` at the given central coverage."""
        z = statistics.NormalDist().inv_cdf(0.5 * (1.0 + coverage))
        mean, spread = self.mean, z * self.std
        return mean - spread, mean + spread

    def result(
        self,
        coverage: float = 0.95,
        p_min: float = 0.01,
        p_max: float = 0.5,
        bh_groups: np.ndarray | None = None,
    ) -> UncertaintyResult:
        """Assemble the full per-unit result: mean, std, interval, p-values and display confidence.

        The p-values are Benjamini-Hochberg adjusted within ``bh_groups``, or as one family if None.
        """
        pvals = self.pvals
        adjusted = bh_adjust_grouped(pvals, bh_groups)
        lower, upper = self.interval(coverage)
        return UncertaintyResult(
            prediction=self.mean,
            pvals=pvals,
            pvals_adj=adjusted,
            confidence=pvals_to_confidence(adjusted, p_min=p_min, p_max=p_max),
            std=self.std,
            lower=lower,
            upper=upper,
        )
