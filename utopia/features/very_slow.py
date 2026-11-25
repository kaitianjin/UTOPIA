"""Very slow extractor: one UNI forward pass per superpixel of the whole lattice.

It also encodes the superpixels HistoSweep filtered out as background, so a coarser unit's feature
can be averaged over all of its superpixels rather than its tissue ones alone. The PCA is still
fitted on tissue superpixels only.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import Config
from .encoder import Projection
from .slow import extract_positions, fit_rows

__all__ = ["fit_rows", "extract_sample"]


def extract_sample(
    model: torch.nn.Module,
    image: np.ndarray,
    tissue_mask: np.ndarray,
    device: torch.device,
    config: Config,
    projections: tuple[Projection, Projection] | None,
) -> tuple[np.ndarray, dict[str, object], None]:
    """Extract the features of every superpixel of one sample, in row-major lattice order."""
    positions = np.argwhere(np.ones(tissue_mask.shape, dtype=bool))
    return extract_positions(model, image, positions, device, config, projections)
