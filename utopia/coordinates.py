"""The two coordinate systems an input may be written in, and the conversion between them.

UTOPIA works on the scale-1 superpixel lattice, where ``(row, col)`` counts superpixels down and
right from the image's top-left corner; ``coordinate_type: superpixel`` inputs are already written
that way. With ``coordinate_type: raw_pixel`` the inputs name raw H&E pixels as ``(y, x)``, and
pixel ``y`` lies in superpixel row ``floor(y * pixel_size_raw / superpixel_size)``. Everything
downstream of the functions here sees superpixels only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import Config

#: Added before rounding down, so a pixel lying exactly on a superpixel boundary is not pushed
#: into the previous superpixel by floating-point error. Far smaller than one raw pixel.
BOUNDARY_TOLERANCE = 1e-6


def raw_pixels_per_superpixel(pixel_size_raw: float, superpixel_size: float) -> float:
    """How many raw H&E pixels one superpixel side spans (not necessarily a whole number)."""
    return float(superpixel_size) / float(pixel_size_raw)


def raw_pixel_to_superpixel(
    pixels: np.ndarray, pixel_size_raw: float, superpixel_size: float
) -> np.ndarray:
    """Convert an integer array of raw H&E pixel coordinates to int32 superpixels, shape kept.

    Any pixel inside a superpixel maps to it, so its top-left corner and its centre both work.
    """
    per_superpixel = raw_pixels_per_superpixel(pixel_size_raw, superpixel_size)
    in_superpixels = np.asarray(pixels, dtype=np.float64) / per_superpixel
    return np.floor(in_superpixels + BOUNDARY_TOLERANCE).astype(np.int32)


def raw_pixel_roi_to_superpixel(
    top_left: tuple[int, int],
    height: int,
    width: int,
    pixel_size_raw: float,
    superpixel_size: float,
) -> tuple[tuple[int, int], int, int]:
    """Convert an ROI written in raw H&E pixels to ``(top_left, height, width)`` in superpixels.

    Each edge of the ROI moves to the nearest superpixel boundary, so two ROIs that touch in raw
    pixels still touch, without overlapping, in superpixels.
    """
    per_superpixel = raw_pixels_per_superpixel(pixel_size_raw, superpixel_size)

    def nearest_boundary(pixel: int) -> int:
        """Index of the superpixel boundary nearest to a raw pixel coordinate."""
        return int(np.floor(pixel / per_superpixel + 0.5))

    row_start = nearest_boundary(top_left[0])
    column_start = nearest_boundary(top_left[1])
    row_stop = nearest_boundary(top_left[0] + height)
    column_stop = nearest_boundary(top_left[1] + width)
    return (row_start, column_start), row_stop - row_start, column_stop - column_start


def positions_to_superpixel(positions: np.ndarray, config: Config, sample: str) -> np.ndarray:
    """Return ``(N, 2)`` positions as int32 superpixel ``(row, col)``, whatever they were given in.

    Raw-pixel positions are converted with the named sample's ``pixel_size_raw``; raises unless
    the array is ``(N, 2)`` integers, since a coordinate is never a fraction of a pixel.
    """
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must have shape (N, 2), got {positions.shape}")
    if not np.issubdtype(positions.dtype, np.integer):
        raise ValueError(
            f"positions must be integers (int32), got dtype {positions.dtype}; round "
            f"{config.coordinate_type} coordinates before saving them"
        )
    if config.coordinate_type == "raw_pixel":
        pixel_size_raw = config.sample_pixel_size_raw(sample)
        return raw_pixel_to_superpixel(positions, pixel_size_raw, config.superpixel_size)
    return positions.astype(np.int32)


def load_positions(directory: Path, config: Config, sample: str) -> np.ndarray:
    """Read ``<directory>/positions.npy`` of the named sample as int32 superpixel ``(row, col)``.

    Every reader of an input ``positions.npy`` goes through here, so the file's coordinate type
    is interpreted in one place.
    """
    path = Path(directory) / "positions.npy"
    try:
        return positions_to_superpixel(np.load(path), config, sample)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def count_shared_superpixels(positions: np.ndarray) -> int:
    """Number of superpixel positions that repeat an earlier row of ``positions``.

    More than zero means two measurements claim the same superpixel, which happens when
    raw-pixel coordinates are finer than the superpixel lattice.
    """
    if len(positions) == 0:
        return 0
    rows = positions[:, 0].astype(np.int64)
    columns = positions[:, 1].astype(np.int64)
    # One number per position that is distinct for every distinct (row, col) pair.
    n_columns = int(columns.max() - columns.min()) + 1
    keys = (rows - rows.min()) * n_columns + (columns - columns.min())
    return int(len(keys) - len(np.unique(keys)))
