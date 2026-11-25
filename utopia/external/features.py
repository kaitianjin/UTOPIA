"""Stage ``features`` of an external run: one UNI feature vector per spot.

Each spot is cropped from the raw H&E as the square that encloses its footprint (the diameter of
a circular spot, or the width of a square one). The config states the footprint in microns, and
it is converted to pixels with the sample's own ``pixel_size_raw``, so the crop covers the same
tissue area on every H&E. The crop is resized to 224 x 224 pixels, and UNI encodes it. The class
token, 1024 numbers, is the spot's feature. The features are only used to cluster the spots, so
this stage does nothing when the user supplies the cluster labels.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import torch

from ..config import EXTERNAL_EXTRACTOR, Config
from ..features.encoder import (
    PATCH,
    RAW_DIM,
    CropDataset,
    encode_on_device,
    load_encoder,
    make_loader,
)
from ..io import load_image, write_json
from ..workspace import Workspace
from .inputs import provided_cluster_labels, sample_files

#: Value painted where a crop reaches past the edge of the image: white, as an empty slide.
OUTSIDE_IMAGE = 255


def crop_spot(image: np.ndarray, row: int, column: int, width: float) -> np.ndarray:
    """Cut the square of side ``width`` centred on a spot and resize it to ``(224, 224, 3)`` uint8.

    ``row`` and ``column`` are the spot centre in pixels of ``image``; the part of the square
    that lies outside the image is white.
    """
    from PIL import Image

    side = max(1, int(round(width)))
    top = int(round(row - side / 2.0))
    left = int(round(column - side / 2.0))

    # The part of the square that lies inside the image.
    row_start = max(0, top)
    row_stop = min(image.shape[0], top + side)
    column_start = max(0, left)
    column_stop = min(image.shape[1], left + side)

    tile = np.full((side, side, 3), OUTSIDE_IMAGE, dtype=np.uint8)
    if row_stop > row_start and column_stop > column_start:
        tile[
            row_start - top : row_stop - top,
            column_start - left : column_stop - left,
        ] = image[row_start:row_stop, column_start:column_stop, :3]

    if side == PATCH:
        return tile
    resized = Image.fromarray(tile).resize((PATCH, PATCH), resample=Image.BILINEAR)
    return np.asarray(resized)


def encode_spots(
    model: torch.nn.Module,
    image: np.ndarray,
    positions: np.ndarray,
    width: float,
    device: torch.device,
    config: Config,
) -> np.ndarray:
    """Encode the spots centred at ``positions`` into ``(len(positions), 1024)`` float16 features.

    ``positions`` are ``(row, col)`` pixels of ``image`` and ``width`` is the side of the square
    cropped around each of them, in the same pixels.
    """
    from tqdm import tqdm

    dataset = CropDataset(image, positions, partial(crop_spot, width=width))
    loader = make_loader(dataset, config)
    features = np.zeros((len(dataset), RAW_DIM), dtype=np.float16)
    for batch, indices in tqdm(loader, desc="[features] encoding", unit="batch"):
        class_tokens, _ = encode_on_device(model, batch, device)
        features[indices.numpy()] = class_tokens.float().cpu().numpy().astype(np.float16)
    return features


def run(config: Config) -> None:
    """Run the feature stage of an external run for every sample in ``config``."""
    labels = provided_cluster_labels(config)
    if labels:
        print(f"[features] cluster labels {labels} are supplied by the user; UNI is not run")
        return

    workspace = Workspace(config)
    settings = config.features
    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    model = load_encoder(settings.model_dir, device)
    print(f"[features] one UNI class token per spot -> '{EXTERNAL_EXTRACTOR}' on {device}")

    for sample in config.samples:
        files = sample_files(config, sample)
        spot = config.sample_spot(sample.name)
        # The footprint is in microns; this sample's pixel size turns it into pixels of its H&E.
        pixel_size_raw = config.sample_pixel_size_raw(sample.name)
        crop_width_pixels = spot.crop_width_pixels(pixel_size_raw)
        positions = np.load(files.positions).astype(np.int64)
        image = load_image(files.he)
        print(
            f"[features] {sample.name}: {len(positions)} spots, each a {spot.crop_width_um:g} um "
            f"= {crop_width_pixels:.1f}-pixel square of the {image.shape[:2]} H&E resized to "
            f"{PATCH} pixels"
        )
        features = encode_spots(model, image, positions, crop_width_pixels, device, config)
        del image

        path = workspace.features(sample.name, EXTERNAL_EXTRACTOR)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, features)
        write_json(
            {
                "extractor": EXTERNAL_EXTRACTOR,
                "encoder": "UNI vit_large_patch16_224",
                "token": "class token",
                "dim": int(features.shape[1]),
                "n_units": int(features.shape[0]),
                "order": "row order of the sample's genes/positions.npy",
                "spot_shape": spot.shape,
                "crop_width_um": float(spot.crop_width_um),
                "crop_width_raw_pixels": float(crop_width_pixels),
                "pixel_size_raw": float(pixel_size_raw),
                "patch": PATCH,
                "he": str(files.he),
            },
            workspace.features_meta(sample.name, EXTERNAL_EXTRACTOR),
        )
        print(f"[features]   wrote {features.shape} float16")
