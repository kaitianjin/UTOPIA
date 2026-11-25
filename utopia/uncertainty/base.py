"""The common result type every uncertainty method returns.

A method fills the fields it can produce and leaves the rest ``None``, so the code that writes
``result.npz``, colours a map or draws an FDP curve is written once and works for all six methods.
Point predictions are the only field every method has; ``none`` fills nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class UncertaintyResult:
    """Per-unit output of an uncertainty method: ``(N,)`` arrays, one entry per test unit.

    Fields a method cannot produce stay None.
    """

    prediction: np.ndarray  # point prediction; the mean over draws for the sampling methods
    pvals: np.ndarray | None = None  # raw p-values against H0: y <= null; small = above null
    pvals_adj: np.ndarray | None = None  # Benjamini-Hochberg adjusted p-values
    confidence: np.ndarray | None = None  # adjusted p-values on a 0-1 display scale
    std: np.ndarray | None = None  # predictive standard deviation (sampling methods)
    level: np.ndarray | None = None  # interval-inversion confidence level (marginal conformal)
    lower: np.ndarray | None = None  # lower end of the predictive interval (sampling methods)
    upper: np.ndarray | None = None  # upper end of the predictive interval (sampling methods)
    extra: dict[str, np.ndarray] = field(default_factory=dict)  # method-specific diagnostics

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the populated fields as a name-to-array mapping."""
        named = {
            "prediction": self.prediction,
            "pvals": self.pvals,
            "pvals_adj": self.pvals_adj,
            "confidence": self.confidence,
            "std": self.std,
            "level": self.level,
            "lower": self.lower,
            "upper": self.upper,
        }
        result = {key: value for key, value in named.items() if value is not None}
        result.update(self.extra)
        return result
