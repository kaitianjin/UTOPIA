"""Assembly of the final tissue mask."""

from __future__ import annotations

import numpy as np

from ..config import TissueMaskConfig
from .metrics import compute_colour_metrics, low_density_mask, low_ratio_mask
from .texture import refine_by_texture


def remove_small_islands(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove the 8-connected tissue islands that hold fewer than ``min_size`` superpixels.

    Written with scipy because scikit-image changed what ``remove_small_objects`` removes in 0.26.
    """
    from scipy import ndimage

    eight_connected = np.ones((3, 3), dtype=bool)
    island_of, _ = ndimage.label(mask, structure=eight_connected)
    island_sizes = np.bincount(island_of.ravel())
    too_small = island_sizes < min_size
    too_small[0] = False  # label 0 is the background, not an island
    return mask & ~too_small[island_of]


def tissue_mask(image: np.ndarray, settings: TissueMaskConfig, patch_size: int = 16) -> np.ndarray:
    """Compute the ``(H // patch_size, W // patch_size)`` boolean tissue mask of an H&E image.

    A superpixel is background if the colour-density filter flags it (after texture refinement) or
    its std/mean ratio is below the Otsu threshold; small tissue islands are then removed.
    """
    image = np.asarray(image, dtype=np.float32)
    metrics = compute_colour_metrics(image, patch_size=patch_size)
    candidates = low_density_mask(metrics, density_thresh=settings.density_thresh)
    candidates = refine_by_texture(
        image, candidates, patch_size=patch_size, levels=settings.glcm_levels
    )
    ratio_background = low_ratio_mask(metrics.ratio, candidates).reshape(metrics.grid)

    mask = ~(candidates | ratio_background)
    if settings.clean_background:
        mask = remove_small_islands(mask, settings.min_size)
    return mask[: image.shape[0] // patch_size, : image.shape[1] // patch_size]
