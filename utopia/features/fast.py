"""Fast extractor: CATCH-lite window fusion (the default method).

The slide is tiled once with 224x224 windows at stride 112 and only the windows that overlap the
tissue mask are encoded. Each window's class token ("global") and 14x14 patch tokens ("local") are
PCA-projected and scattered onto the superpixel lattice with a Gaussian centre weight, so a
superpixel's feature is a weighted average over the windows that saw it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import Config
from .encoder import (
    GRID,
    PATCH,
    RAW_DIM,
    STRIDE,
    TOKEN,
    PcaFit,
    Projection,
    encode_on_device,
    make_loader,
    normalise,
)

#: Default ``features.fit_patch_stride`` for this extractor: every selected window
#: is encoded for the PCA fit, as in CATCH's standard mode.
FIT_STRIDE = 1


def gaussian_center_weights(size: int = GRID, sigma: float = 0.5) -> np.ndarray:
    """Bell-shaped ``(size, size)`` token-grid weights, peaking at 1 in the window centre."""
    axis = np.linspace(-1.0, 1.0, size)
    columns, rows = np.meshgrid(axis, axis)
    weights = np.exp(-(columns**2 + rows**2) / (2.0 * sigma**2))
    return (weights / weights.max()).astype(np.float32)


def tissue_windows(tissue_mask: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Boolean ``grid``-shaped array: which stride-112 windows overlap a tissue superpixel."""
    # Window (r, c) covers superpixels [7r, 7r + 14) x [7c, 7c + 14): the 2x2 block of 7x7 cells
    # starting at cell (r, c).
    n_rows, n_columns = grid
    half = GRID // 2
    height, width = (n_rows + 1) * half, (n_columns + 1) * half
    covered = np.zeros((height, width), dtype=bool)
    rows, columns = min(height, tissue_mask.shape[0]), min(width, tissue_mask.shape[1])
    covered[:rows, :columns] = tissue_mask[:rows, :columns]
    cells = covered.reshape(n_rows + 1, half, n_columns + 1, half).any(axis=(1, 3))
    return cells[:-1, :-1] | cells[1:, :-1] | cells[:-1, 1:] | cells[1:, 1:]


def select_windows(tissue_mask: np.ndarray, grid: tuple[int, int], config: Config) -> np.ndarray:
    """Boolean ``grid``-shaped array of the windows to encode: those over tissue, or all of them."""
    if config.features.tissue_windows:
        return tissue_windows(tissue_mask, grid)
    return np.ones(grid, dtype=bool)


class _WindowDataset(Dataset):
    """Yields normalised 224x224 windows together with their grid position."""

    def __init__(self, image: np.ndarray) -> None:
        """Lay the stride-112 window grid over ``image``."""
        self.image = image
        self.n_rows = image.shape[0] // STRIDE - 1
        self.n_columns = image.shape[1] // STRIDE - 1

    @property
    def grid(self) -> tuple[int, int]:
        """``(n_rows, n_columns)`` of the window grid."""
        return self.n_rows, self.n_columns

    def __len__(self) -> int:
        """Number of windows in the grid."""
        return self.n_rows * self.n_columns

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        """The normalised window at row-major ``index``, with its grid row and column."""
        row, column = divmod(index, self.n_columns)
        tile = self.image[
            row * STRIDE : row * STRIDE + PATCH, column * STRIDE : column * STRIDE + PATCH
        ]
        return normalise(tile), row, column


def fit_rows(
    model: torch.nn.Module,
    image: np.ndarray,
    tissue_mask: np.ndarray,
    device: torch.device,
    config: Config,
    fit: PcaFit,
) -> None:
    """Encode every ``fit_patch_stride``-th selected window (default: all) into ``fit``."""
    dataset = _WindowDataset(image)
    selected = np.flatnonzero(select_windows(tissue_mask, dataset.grid, config).ravel())
    indices = selected[:: config.features.fit_patch_stride or FIT_STRIDE]
    fit.add(make_loader(dataset, config, indices), model)


def _scatter_row(
    feature_map: np.ndarray,
    count_map: np.ndarray,
    row: int,
    global_row: np.ndarray,
    local_row: np.ndarray | None,
    weights: np.ndarray,
    present: np.ndarray | None = None,
) -> None:
    """Accumulate one complete window row into the lattice feature map and its weight counts.

    ``local_row=None`` fuses the global stream alone; ``present`` marks the encoded windows of the
    row (None means all of them).
    """
    n_columns, global_dim = global_row.shape
    local_dim = 0 if local_row is None else local_row.shape[-1]
    token_row = row * (GRID // 2)

    # Windows overlap their right neighbour by half a window, so the windows of one column parity
    # have disjoint token blocks and are written with one block assignment, not a scatter-add.
    for parity in (0, 1):
        columns = np.arange(parity, n_columns, 2)
        column_weights = weights[:, None, :]
        if present is not None:
            # Trim to the span of encoded windows; unencoded ones inside it get zero weight.
            encoded = np.flatnonzero(present[columns])
            if len(encoded) == 0:
                continue
            columns = columns[encoded[0] : encoded[-1] + 1]
            column_weights = column_weights * present[columns].astype(np.float32)[None, :, None]
        if len(columns) == 0:
            continue
        start = columns[0] * (GRID // 2)
        width = len(columns) * GRID

        block = np.empty((GRID, width, global_dim + local_dim), dtype=np.float32)
        block[:, :, :global_dim] = (
            column_weights[..., None] * global_row[columns][None, :, None, :]
        ).reshape(GRID, width, global_dim)
        if local_row is not None:
            block[:, :, global_dim:] = (
                column_weights[..., None] * local_row[columns].transpose(1, 0, 2, 3)
            ).reshape(GRID, width, local_dim)

        feature_map[token_row : token_row + GRID, start : start + width] += block
        count_map[token_row : token_row + GRID, start : start + width] += np.broadcast_to(
            column_weights, (GRID, len(columns), GRID)
        ).reshape(GRID, width)


def fuse_globals(
    window_globals: np.ndarray, lattice: tuple[int, int], present: np.ndarray | None = None
) -> np.ndarray:
    """Fuse ``(n_rows, n_columns, D)`` window-level global embeddings alone onto the lattice.

    The ``(H // 16, W // 16, D)`` result equals the global half of :func:`fuse_windows`.
    """
    weights = gaussian_center_weights()
    feature_map = np.zeros((*lattice, window_globals.shape[-1]), dtype=np.float32)
    count_map = np.zeros(lattice, dtype=np.float32)
    for row in range(window_globals.shape[0]):
        if present is None:
            _scatter_row(feature_map, count_map, row, window_globals[row], None, weights)
        elif present[row].any():
            _scatter_row(
                feature_map, count_map, row, window_globals[row], None, weights, present[row]
            )
    return feature_map / np.clip(count_map, 1e-6, None)[..., None]


def fuse_windows(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    config: Config,
    projections: tuple[Projection, Projection] | None,
    present: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode the ``present`` windows of an image and fuse their tokens onto the lattice.

    Returns the ``(H // 16, W // 16, D)`` lattice map and the ``(n_rows, n_columns, global_dim)``
    per-window global embeddings Harmony works on, zero at windows that were not encoded.
    """
    from tqdm import tqdm

    dataset = _WindowDataset(image)
    loader = make_loader(dataset, config, np.flatnonzero(present.ravel()))
    expected = present.sum(axis=1)
    weights = gaussian_center_weights()

    global_dim = projections[0].dim if projections else RAW_DIM
    local_dim = projections[1].dim if projections else RAW_DIM
    lattice = (image.shape[0] // TOKEN, image.shape[1] // TOKEN)
    feature_map = np.zeros((*lattice, global_dim + local_dim), dtype=np.float32)
    count_map = np.zeros(lattice, dtype=np.float32)
    window_globals = np.zeros((dataset.n_rows, dataset.n_columns, global_dim), dtype=np.float32)

    pending: dict[int, np.ndarray] = {}
    filled: dict[int, int] = {}
    for batch, rows, columns in tqdm(loader, desc="[features] encoding", unit="batch"):
        global_tokens, local_tokens = encode_on_device(model, batch, device)
        if projections is not None:
            global_tokens = projections[0](global_tokens)
            local_tokens = projections[1](local_tokens)
        global_batch = global_tokens.float().cpu().numpy()
        local_batch = local_tokens.float().reshape(-1, GRID, GRID, local_dim).cpu().numpy()

        for row, column, global_window, local_window in zip(
            rows.numpy(), columns.numpy(), global_batch, local_batch
        ):
            row = int(row)
            window_globals[row, column] = global_window
            if row not in pending:
                pending[row] = np.zeros(
                    (dataset.n_columns, GRID, GRID, local_dim), dtype=np.float32
                )
                filled[row] = 0
            pending[row][column] = local_window
            filled[row] += 1
            if filled[row] == expected[row]:
                _scatter_row(
                    feature_map, count_map, row, window_globals[row], pending.pop(row), weights,
                    present[row],
                )
                del filled[row]

    if pending:
        raise RuntimeError(f"window rows {sorted(pending)} were never completed")
    feature_map /= np.clip(count_map, 1e-6, None)[..., None]
    return feature_map, window_globals


def extract_sample(
    model: torch.nn.Module,
    image: np.ndarray,
    tissue_mask: np.ndarray,
    device: torch.device,
    config: Config,
    projections: tuple[Projection, Projection] | None,
) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    """Extract fused float16 features for one sample's tissue superpixels, in tissue-mask order.

    The third return value holds the per-window global embeddings, which Harmony corrects.
    """
    present = select_windows(tissue_mask, _WindowDataset(image).grid, config)
    feature_map, window_globals = fuse_windows(model, image, device, config, projections, present)
    provenance = {
        "patch": PATCH,
        "stride": STRIDE,
        "token": TOKEN,
        "windows": "tissue" if config.features.tissue_windows else "all",
        "n_windows": int(present.sum()),
        "n_windows_grid": int(present.size),
    }
    return feature_map[tissue_mask].astype(np.float16), provenance, window_globals
