"""Grouped reductions over flat per-unit arrays.

Every change of spatial resolution (coarsening features, summing gene counts into 2x2 or 4x4
super-units) is "group rows by an integer key, then sum or average them". These pure helpers do
that with a sort plus :func:`numpy.add.reduceat`, which is far faster than ``np.add.at``.
"""

from __future__ import annotations

import numpy as np


def group_index(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factorise integer ``keys`` into ``(sorted unique keys, group id per row, group sizes)``."""
    unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return unique_keys, inverse.astype(np.int64, copy=False), counts


def group_sum(values: np.ndarray, inverse: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum the rows of ``values`` (1-D or 2-D) within each group id of ``inverse``."""
    if values.ndim == 1:
        return np.bincount(
            inverse, weights=np.asarray(values, dtype=np.float64), minlength=n_groups
        )
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    is_group_start = np.concatenate(([True], sorted_inverse[1:] != sorted_inverse[:-1]))
    starts = np.flatnonzero(is_group_start)
    sorted_values = np.asarray(values, dtype=np.float32)[order]
    sums = np.add.reduceat(sorted_values, starts, axis=0)
    out = np.zeros((n_groups, values.shape[1]), dtype=np.float32)
    out[sorted_inverse[starts]] = sums
    return out


def group_mean(
    values: np.ndarray, inverse: np.ndarray, n_groups: int, counts: np.ndarray
) -> np.ndarray:
    """Average the rows of ``values`` within each group; ``counts`` are the group sizes."""
    sums = group_sum(values, inverse, n_groups)
    denominator = counts if sums.ndim == 1 else counts[:, None]
    return sums / denominator


def lattice_keys(sample_index: np.ndarray, positions: np.ndarray, scale: int,
                 span: int) -> np.ndarray:
    """Build one integer key per unit identifying its sample and parent cell at ``scale``.

    ``span`` is any bound on both lattice dimensions (e.g. the largest side over all samples).
    """
    parents = np.asarray(positions, dtype=np.int64) // scale
    samples = np.asarray(sample_index, dtype=np.int64)
    stride = int(span) // scale + 1
    parent_rows = parents[:, 0]
    parent_columns = parents[:, 1]
    return (samples * stride + parent_rows) * stride + parent_columns


def decode_lattice_keys(keys: np.ndarray, scale: int, span: int) -> tuple[np.ndarray, np.ndarray]:
    """Invert :func:`lattice_keys` into ``(sample_index, coarse-lattice (row, col) positions)``."""
    stride = int(span) // scale + 1
    parent_column = keys % stride
    rest = keys // stride
    parent_row = rest % stride
    sample_index = rest // stride
    parent_positions = np.stack([parent_row, parent_column], axis=1)
    return sample_index.astype(np.int32), parent_positions.astype(np.int32)


def block_mean_features(
    features: np.ndarray, lattice_shape: tuple[int, int], scale: int, parents: np.ndarray
) -> np.ndarray:
    """Average the features of every superpixel in each ``scale x scale`` block of ``parents``.

    ``features`` has one row per lattice superpixel in row-major order; a block cut short by the
    lattice edge averages the superpixels it has.
    """
    n_rows, n_columns = lattice_shape
    parent_rows = np.asarray(parents[:, 0], dtype=np.int64)
    parent_columns = np.asarray(parents[:, 1], dtype=np.int64)
    sums = np.zeros((len(parents), features.shape[1]), dtype=np.float32)
    counts = np.zeros(len(parents), dtype=np.float32)

    for row_offset in range(scale):
        for column_offset in range(scale):
            rows = parent_rows * scale + row_offset
            columns = parent_columns * scale + column_offset
            inside = (rows < n_rows) & (columns < n_columns)
            feature_rows = rows[inside] * n_columns + columns[inside]
            sums[inside] += np.asarray(features[feature_rows], dtype=np.float32)
            counts[inside] += 1

    return sums / counts[:, None]


def coarse_shape(shape: tuple[int, int], scale: int) -> tuple[int, int]:
    """Shape of a lattice after coarsening by ``scale``, rounding up to keep partial edge cells."""
    return ((shape[0] + scale - 1) // scale, (shape[1] + scale - 1) // scale)


def coarsen_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    """Downsample a boolean lattice mask; a parent cell is ``True`` if any child is."""
    if scale == 1:
        return np.asarray(mask, dtype=bool)
    rows, columns = coarse_shape(mask.shape, scale)
    padded = np.zeros((rows * scale, columns * scale), dtype=bool)
    padded[: mask.shape[0], : mask.shape[1]] = mask
    return padded.reshape(rows, scale, columns, scale).any(axis=(1, 3))
