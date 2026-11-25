"""Per-superpixel colour statistics and the density filter built on them.

Two behaviours that look like mistakes are kept to match the reference HistoSweep implementation:
each channel's variance is reduced on its own, and the density filter's bin lookup drops the
slide's minimum mean and minimum std as out of range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


def patchify(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Split an image into a flat stack of non-overlapping square patches, edge-padding it to fit.

    Returns the ``(n_patches, patch_size, patch_size, channels)`` patches in row-major grid order
    and the ``(rows, columns)`` shape of that grid.
    """
    height, width = image.shape[:2]
    grid = ((height + patch_size - 1) // patch_size, (width + patch_size - 1) // patch_size)
    padded = np.pad(
        image,
        ((0, grid[0] * patch_size - height), (0, grid[1] * patch_size - width), (0, 0)),
        mode="edge",
    )
    # A strided view gives the patches in row-major grid order; the reshape below copies them.
    strides = (
        padded.strides[0] * patch_size,
        padded.strides[1] * patch_size,
        padded.strides[0],
        padded.strides[1],
        padded.strides[2],
    )
    patches = np.lib.stride_tricks.as_strided(
        padded, shape=(grid[0], grid[1], patch_size, patch_size, image.shape[2]), strides=strides
    )
    return patches.reshape(-1, patch_size, patch_size, image.shape[2]), grid


@dataclass
class ColourMetrics:
    """Per-superpixel colour statistics over the superpixel grid."""

    std: np.ndarray  # (R, C) standard deviation of the pixel values within each superpixel
    weighted_mean: np.ndarray  # (R, C) variance-weighted mean of the per-channel means
    ratio: np.ndarray  # (R * C,) min-max normalised std / weighted_mean ratio
    grid: tuple[int, int]  # (R, C) shape of the superpixel grid


def compute_colour_metrics(image: np.ndarray, patch_size: int = 16) -> ColourMetrics:
    """Compute the colour statistics the density and ratio filters consume."""
    patches, grid = patchify(image, patch_size)
    std = np.std(patches, axis=(1, 2, 3)).reshape(grid)

    # One variance per channel, kept to match the reference HistoSweep implementation: reducing
    # the (N, 3) array along axis 0 rounds differently in float32 and moves superpixels across bins.
    channel_means = np.mean(patches, axis=(1, 2))
    variances = np.stack([np.var(channel_means[:, channel]) for channel in range(3)])
    weighted_mean = (
        channel_means[:, 0] * variances[0]
        + channel_means[:, 1] * variances[1]
        + channel_means[:, 2] * variances[2]
    ) / (variances[0] + variances[1] + variances[2])
    weighted_mean = weighted_mean.reshape(grid)

    def unit_shift(values: np.ndarray) -> np.ndarray:
        """Flatten ``values`` and rescale them to the range 1 to 2."""
        flat = values.ravel()
        return (flat - flat.min()) / (flat.max() - flat.min()) + 1

    ratio = unit_shift(std) / unit_shift(weighted_mean)
    ratio = (ratio - ratio.min()) / (ratio.max() - ratio.min())
    return ColourMetrics(std=std, weighted_mean=weighted_mean, ratio=ratio, grid=grid)


def low_density_mask(metrics: ColourMetrics, density_thresh: float = 100.0) -> np.ndarray:
    """Flag superpixels that are rare in the (mean, std) colour distribution.

    Returns an ``(R, C)`` boolean array, True where the smoothed joint-histogram count is below
    ``density_thresh``: those superpixels are background candidates, since tissue forms dense modes.
    """
    n_bins = max(1, int(metrics.ratio.size * 0.00005))
    sigma = max(1, int(n_bins * 0.02))

    mean, std = metrics.weighted_mean.ravel(), metrics.std.ravel()
    histogram, mean_edges, std_edges = np.histogram2d(mean, std, bins=[n_bins, n_bins])
    smoothed = gaussian_filter(histogram, sigma=sigma)

    # Uncorrected bin lookup, kept to match the reference HistoSweep implementation: the slide's
    # minimum mean and minimum std get index -1 and are always flagged as background candidates.
    mean_bin = np.searchsorted(mean_edges, mean) - 1
    std_bin = np.searchsorted(std_edges, std) - 1
    inside = (
        (mean_bin >= 0)
        & (mean_bin < smoothed.shape[0])
        & (std_bin >= 0)
        & (std_bin < smoothed.shape[1])
    )
    density = np.zeros_like(std)
    density[inside] = smoothed[mean_bin[inside], std_bin[inside]]
    return (density < density_thresh).reshape(metrics.grid)


def low_ratio_mask(ratio: np.ndarray, background_candidates: np.ndarray) -> np.ndarray:
    """Flag superpixels whose std/mean ratio is below an Otsu threshold.

    The threshold is fitted on the superpixels outside ``background_candidates``; the result is an
    ``(R * C,)`` boolean array.
    """
    from skimage.filters import threshold_otsu

    candidates = ratio[~background_candidates.ravel()]
    trimmed = candidates[candidates < np.percentile(candidates, 90)]
    return ratio < threshold_otsu(trimmed)
