"""Tissue mask of an external sample, on the grid its maps are drawn on.

An external model's spots do not cover the whole tissue, so HistoSweep masks the tissue of the H&E
and the mask is painted in grey behind the spots. HistoSweep always runs at the resolution it was
tuned for, 0.5 microns per pixel with 16-pixel (8 micron) superpixels, and the mask is then
resampled onto the drawing grid. A later ``import`` of the same H&E with the same settings reuses
the saved mask instead of reading the image again.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Config, SampleSpec
from ..histosweep import tissue_mask as compute_tissue_mask
from ..io import load_image, load_mask, read_json, save_mask, write_json
from ..workspace import Workspace

#: Pixel size HistoSweep is run at, in microns.
MASK_PIXEL_SIZE = 0.5
#: Side of one HistoSweep superpixel, in pixels of the image at :data:`MASK_PIXEL_SIZE`.
MASK_SUPERPIXEL = 16
#: Physical side of one HistoSweep superpixel, in microns.
MASK_SUPERPIXEL_SIZE = MASK_PIXEL_SIZE * MASK_SUPERPIXEL
#: Output rows resampled per step; a whole slide is too large to resample in one call.
STRIP_ROWS = 1024


def grid_shape(
    image_shape: tuple[int, int], pixel_size_raw: float, superpixel_size: float
) -> tuple[int, int]:
    """Rows and columns of the drawing grid that covers a raw H&E of ``image_shape`` pixels.

    One grid cell is ``superpixel_size`` microns wide; the last row and column are kept even when
    the image covers them only partly.
    """
    rows = int(np.ceil(image_shape[0] * pixel_size_raw / superpixel_size))
    columns = int(np.ceil(image_shape[1] * pixel_size_raw / superpixel_size))
    return rows, columns


def resample_image(image: np.ndarray, factor: float) -> np.ndarray:
    """Resize an ``(H, W, 3)`` uint8 image by ``factor`` with area averaging, strip by strip.

    Each strip of output rows is resized from the source rows it covers, so the memory needed
    beyond the input and the output is one strip.
    """
    from PIL import Image

    height, width = image.shape[:2]
    out_height = max(1, int(round(height * factor)))
    out_width = max(1, int(round(width * factor)))
    resized = np.empty((out_height, out_width, 3), dtype=np.uint8)

    for first_row in range(0, out_height, STRIP_ROWS):
        last_row = min(out_height, first_row + STRIP_ROWS)
        source_first = int(np.floor(first_row / factor))
        source_last = min(height, int(np.ceil(last_row / factor)))
        strip = Image.fromarray(np.ascontiguousarray(image[source_first:source_last, :, :3]))
        strip = strip.resize((out_width, last_row - first_row), resample=Image.BOX)
        resized[first_row:last_row] = np.asarray(strip)
    return resized


def mask_to_grid(
    mask: np.ndarray, mask_cell_size: float, shape: tuple[int, int], cell_size: float
) -> np.ndarray:
    """Resample a boolean mask whose cells are ``mask_cell_size`` microns onto a grid of ``shape``.

    Each grid cell of ``cell_size`` microns takes the value of the mask cell under its centre, or
    of the nearest mask cell when it lies past the edge of the mask.
    """
    row_centres = (np.arange(shape[0]) + 0.5) * cell_size
    column_centres = (np.arange(shape[1]) + 0.5) * cell_size
    mask_rows = np.floor(row_centres / mask_cell_size).astype(np.int64)
    mask_columns = np.floor(column_centres / mask_cell_size).astype(np.int64)
    mask_rows = np.clip(mask_rows, 0, mask.shape[0] - 1)
    mask_columns = np.clip(mask_columns, 0, mask.shape[1] - 1)
    return mask[mask_rows][:, mask_columns]


def tissue_on_grid(image: np.ndarray, pixel_size_raw: float, config: Config) -> np.ndarray:
    """Compute the boolean tissue mask of a raw H&E on its ``superpixel_size`` drawing grid."""
    factor = pixel_size_raw / MASK_PIXEL_SIZE
    if abs(factor - 1.0) > 1e-6:
        resampled = resample_image(image, factor)
    else:
        resampled = image[:, :, :3]

    mask = compute_tissue_mask(resampled, config.tissue_mask, patch_size=MASK_SUPERPIXEL)
    shape = grid_shape(image.shape[:2], pixel_size_raw, config.superpixel_size)
    return mask_to_grid(mask, MASK_SUPERPIXEL_SIZE, shape, config.superpixel_size)


def _provenance(config: Config, sample: SampleSpec, he_path: Path) -> dict[str, object]:
    """Everything a sample's tissue mask depends on, apart from the pixels of the H&E."""
    status = he_path.stat()
    settings = config.tissue_mask
    return {
        "he": str(he_path),
        "he_bytes": int(status.st_size),
        "he_modified": float(status.st_mtime),
        "pixel_size_raw": float(config.sample_pixel_size_raw(sample.name)),
        "superpixel_size": float(config.superpixel_size),
        "density_thresh": float(settings.density_thresh),
        "min_size": int(settings.min_size),
        "clean_background": bool(settings.clean_background),
        "glcm_levels": int(settings.glcm_levels),
    }


def prepare(config: Config, workspace: Workspace, sample: SampleSpec) -> tuple[int, int] | None:
    """Make (or reuse) one sample's tissue mask and return ``(rows, columns)`` of its raw H&E.

    Returns None for a sample without an H&E, after removing any mask an earlier run left for it.
    """
    mask_path = workspace.tissue_mask(sample.name)
    # Records which H&E and settings the mask was made from.
    record_path = workspace.sample_dir(sample.name) / "tissue_mask.json"
    if sample.he is None:
        for stale in (mask_path, record_path):
            if stale.exists():
                stale.unlink()
        print(f"[import]   {sample.name}: no H&E, so its maps are drawn without a tissue mask")
        return None

    he_path = config.sample_dir(sample.name) / sample.he
    wanted = _provenance(config, sample, he_path)
    if mask_path.exists() and record_path.exists():
        record = read_json(record_path)
        if record.get("made_from") == wanted:
            mask = load_mask(mask_path)
            print(
                f"[import]   {sample.name}: reusing the tissue mask {mask.shape} "
                f"({int(mask.sum())} tissue cells)"
            )
            return int(record["image_shape"][0]), int(record["image_shape"][1])

    image = load_image(he_path)
    image_shape = (int(image.shape[0]), int(image.shape[1]))
    pixel_size_raw = config.sample_pixel_size_raw(sample.name)
    mask = tissue_on_grid(image, pixel_size_raw, config)
    del image

    save_mask(mask, mask_path)
    write_json(
        {
            "made_from": wanted,
            "image_shape": list(image_shape),
            "grid_shape": [int(mask.shape[0]), int(mask.shape[1])],
            "tissue_fraction": float(mask.mean()),
        },
        record_path,
    )
    print(
        f"[import]   {sample.name}: H&E {image_shape} at {pixel_size_raw:g} um/pixel, tissue "
        f"mask {mask.shape} with {int(mask.sum())} tissue cells"
    )
    return image_shape
