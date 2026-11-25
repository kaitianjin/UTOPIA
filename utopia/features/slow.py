"""Slow extractor: one UNI forward pass per tissue superpixel.

Each superpixel gets its own 224x224 window centred on it and keeps the class token (global) plus
the centre patch token (local). Only the superpixels HistoSweep kept as tissue are encoded;
``very_slow`` encodes every superpixel of the lattice with the same code.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import Config
from .encoder import (
    CENTRE_TOKEN,
    PATCH,
    RAW_DIM,
    TOKEN,
    CropDataset,
    PcaFit,
    Projection,
    encode_on_device,
    make_loader,
)

#: Default ``features.fit_patch_stride``: the PCA is fitted on every 16th tissue
#: superpixel, because a full fit pass would cost as much as the extraction.
FIT_STRIDE = 16


def _window(image: np.ndarray, row: int, column: int) -> np.ndarray:
    """Cut the 224x224 window centred on one superpixel, zero-padded at the image border."""
    centre_row = row * TOKEN + TOKEN // 2
    centre_column = column * TOKEN + TOKEN // 2
    half = PATCH // 2
    tile = image[
        max(0, centre_row - half) : min(image.shape[0], centre_row + half),
        max(0, centre_column - half) : min(image.shape[1], centre_column + half),
    ]
    if tile.shape[:2] == (PATCH, PATCH):
        return tile

    padded = np.zeros((PATCH, PATCH, 3), dtype=image.dtype)
    top = (PATCH - tile.shape[0]) // 2
    left = (PATCH - tile.shape[1]) // 2
    padded[top : top + tile.shape[0], left : left + tile.shape[1]] = tile
    return padded


def fit_rows(
    model: torch.nn.Module,
    image: np.ndarray,
    tissue_mask: np.ndarray,
    device: torch.device,
    config: Config,
    fit: PcaFit,
) -> None:
    """Add every ``fit_patch_stride``-th tissue superpixel of one sample to the PCA fit."""
    dataset = CropDataset(image, np.argwhere(tissue_mask), _window)
    stride = config.features.fit_patch_stride or FIT_STRIDE
    indices = np.arange(0, len(dataset), stride)
    fit.add(make_loader(dataset, config, indices), model)


def encode_superpixels(
    model: torch.nn.Module,
    image: np.ndarray,
    positions: np.ndarray,
    device: torch.device,
    config: Config,
    projections: tuple[Projection, Projection] | None,
) -> np.ndarray:
    """Encode the superpixels at ``positions`` into ``(len(positions), D)`` float16 features.

    Each row is the class token followed by the centre token, both PCA-projected when
    ``projections`` is given.
    """
    from tqdm import tqdm

    dataset = CropDataset(image, positions, _window)
    global_dim = projections[0].dim if projections else RAW_DIM
    local_dim = projections[1].dim if projections else RAW_DIM
    features = np.zeros((len(dataset), global_dim + local_dim), dtype=np.float16)

    loader = make_loader(dataset, config)
    for batch, indices in tqdm(loader, desc="[features] encoding", unit="batch"):
        global_tokens, local_tokens = encode_on_device(model, batch, device)
        local_tokens = local_tokens[:, CENTRE_TOKEN]
        if projections is not None:
            global_tokens = projections[0](global_tokens)
            local_tokens = projections[1](local_tokens)
        fused = torch.cat([global_tokens, local_tokens], dim=1).float().cpu().numpy()
        features[indices.numpy()] = fused.astype(np.float16)

    return features


def extract_positions(
    model: torch.nn.Module,
    image: np.ndarray,
    positions: np.ndarray,
    device: torch.device,
    config: Config,
    projections: tuple[Projection, Projection] | None,
) -> tuple[np.ndarray, dict[str, object], None]:
    """Extract the features of the superpixels at ``positions``, with their provenance record.

    The trailing None stands where the fast extractor returns window embeddings for Harmony.
    """
    features = encode_superpixels(model, image, positions, device, config, projections)
    provenance = {"patch": PATCH, "stride": TOKEN, "token": TOKEN, "n_windows": len(positions)}
    return features, provenance, None


def extract_sample(
    model: torch.nn.Module,
    image: np.ndarray,
    tissue_mask: np.ndarray,
    device: torch.device,
    config: Config,
    projections: tuple[Projection, Projection] | None,
) -> tuple[np.ndarray, dict[str, object], None]:
    """Extract the features of one sample's tissue superpixels, in tissue-mask order."""
    positions = np.argwhere(tissue_mask)
    return extract_positions(model, image, positions, device, config, projections)
