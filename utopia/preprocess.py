"""Stage 1: rescale the H&E, mask the tissue, and record each sample's geometry.

Each H&E is rescaled with its own ``pixel_size_raw`` to ``superpixel_size / 16`` microns per pixel,
so one 16 x 16 pixel superpixel covers the physical size the user asked for, and padded to
multiples of 224 pixels so the feature extractor's patch grid tiles it exactly. Positions and ROIs
given in raw H&E pixels (``coordinate_type: raw_pixel``) are converted to superpixels here;
everything written to the workspace (image, tissue mask, ``geometry.npz``) is in superpixels.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import SUPERPIXEL, Config, RoiSpec
from .coordinates import (
    count_shared_superpixels,
    load_positions,
    raw_pixel_roi_to_superpixel,
    raw_pixels_per_superpixel,
)
from .geometry import Geometry, Roi, limited_to_positions, positions_to_mask
from .histosweep import tissue_mask as compute_tissue_mask
from .io import load_image, read_json, save_image, save_mask, write_json
from .workspace import Workspace

if TYPE_CHECKING:
    from .inference import Target

#: The feature extractor's patch side; images are padded to a multiple of it.
PATCH = 224


def rescale_and_pad(image: np.ndarray, scale: float, pad_to: int = PATCH) -> np.ndarray:
    """Resample an image by ``scale`` and white-pad both sides up to a multiple of ``pad_to``."""
    from skimage.transform import rescale as skimage_rescale

    if abs(scale - 1.0) > 1e-6:
        image = skimage_rescale(
            image.astype(np.float32),
            (scale, scale, 1),
            preserve_range=True,
            anti_aliasing=scale < 1,
        )
    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape[:2]
    pad = ((0, -height % pad_to), (0, -width % pad_to), (0, 0))
    return np.pad(image, pad, mode="constant", constant_values=255)


def is_canonical(path: Path, scale: float, pad_to: int = PATCH) -> bool:
    """Whether an input image needs no resampling or padding, so the file can simply be copied.

    Only the image header is read.
    """
    from PIL import Image

    if abs(scale - 1.0) > 1e-6 or path.suffix.lower() != ".png":
        return False
    with Image.open(path) as image:
        width, height = image.size
        return image.mode in ("RGB", "RGBA") and width % pad_to == 0 and height % pad_to == 0


def _outside_lattice(positions: np.ndarray, lattice_shape: tuple[int, int]) -> np.ndarray:
    """Boolean array marking the ``(row, col)`` positions that fall outside the lattice."""
    return (
        (positions[:, 0] < 0)
        | (positions[:, 0] >= lattice_shape[0])
        | (positions[:, 1] < 0)
        | (positions[:, 1] >= lattice_shape[1])
    )


def _measured_mask(
    config: Config, directory: Path, modality: str, lattice_shape: tuple[int, int], sample: str
) -> np.ndarray:
    """Build the mask of the superpixels one modality's positions file lists.

    Raises if a position is off the lattice, or if two positions are the same superpixel.
    """
    positions = load_positions(directory, config, sample)
    out_of_bounds = _outside_lattice(positions, lattice_shape)
    if out_of_bounds.any():
        raise ValueError(
            f"sample '{sample}', modality '{modality}': {int(out_of_bounds.sum())} of "
            f"{len(positions)} positions fall outside the {lattice_shape} superpixel lattice; "
            f"{_expected_positions(config, sample)}"
        )
    n_shared = count_shared_superpixels(positions)
    if n_shared:
        raise ValueError(
            f"sample '{sample}', modality '{modality}': "
            f"{_shared_superpixel_message(config, n_shared, sample)}"
        )
    return positions_to_mask(positions, lattice_shape)


def _expected_positions(config: Config, sample: str) -> str:
    """One sentence saying what a sample's ``positions.npy`` must hold under the coordinate type."""
    if config.coordinate_type == "raw_pixel":
        pixel_size_raw = config.sample_pixel_size_raw(sample)
        return (
            "with coordinate_type 'raw_pixel', positions must be (row, col) = (y, x) pixels of "
            f"the raw H&E, whose pixel size is pixel_size_raw={pixel_size_raw:g} um"
        )
    return (
        "with coordinate_type 'superpixel', positions must be scale-1 superpixel indices of the "
        f"processed image ({config.superpixel_size:g} um per superpixel)"
    )


def _shared_superpixel_message(config: Config, n_shared: int, sample: str) -> str:
    """Explain that ``n_shared`` positions repeat a superpixel another position already names."""
    message = (
        f"{n_shared} position(s) name a superpixel that another position already names; each "
        "superpixel can carry one measurement"
    )
    if config.coordinate_type == "raw_pixel":
        pixel_size_raw = config.sample_pixel_size_raw(sample)
        per_superpixel = raw_pixels_per_superpixel(pixel_size_raw, config.superpixel_size)
        message += (
            f". One superpixel spans {per_superpixel:g} raw pixels "
            f"(superpixel_size={config.superpixel_size:g} um / pixel_size_raw="
            f"{pixel_size_raw:g} um), so the measurements must be binned to "
            f"{config.superpixel_size:g} um before they are given to UTOPIA"
        )
    return message


def roi_in_superpixels(config: Config, roi: RoiSpec, sample: str) -> Roi:
    """Turn one configured ROI into a lattice :class:`Roi`, converting from raw pixels if needed."""
    top_left, height, width = roi.top_left, roi.height, roi.width
    if config.coordinate_type == "raw_pixel":
        pixel_size_raw = config.sample_pixel_size_raw(sample)
        top_left, height, width = raw_pixel_roi_to_superpixel(
            roi.top_left, roi.height, roi.width, pixel_size_raw, config.superpixel_size
        )
    if height < 1 or width < 1:
        raise ValueError(
            f"sample '{sample}': the ROI at top_left={list(roi.top_left)} with "
            f"height={roi.height}, width={roi.width} ({config.coordinate_type} coordinates) "
            "covers no whole superpixel"
        )
    return Roi(top_left=top_left, height=height, width=width, roles=roi.roles)


def run(config: Config) -> None:
    """Run the preprocess stage for every sample in ``config``."""
    workspace = Workspace(config)
    workspace.prepare()
    print(
        f"[preprocess] superpixel {config.superpixel_size:g} um = {SUPERPIXEL} pixels at "
        f"{config.pixel_size:g} um/pixel; coordinate_type is '{config.coordinate_type}'"
    )

    for sample in config.samples:
        print(f"[preprocess] {sample.name}")
        pixel_size_raw = config.sample_pixel_size_raw(sample.name)
        scale = config.rescale_factor(sample.name)
        print(f"[preprocess]   raw H&E is {pixel_size_raw:g} um/pixel, rescaled by {scale:.4g}x")
        source = config.sample_dir(sample.name) / sample.he
        if is_canonical(source, scale):
            workspace.he(sample.name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, workspace.he(sample.name))
            image = load_image(workspace.he(sample.name))
            print(f"[preprocess]   copied {source.name}; no rescaling or padding needed")
        else:
            image = rescale_and_pad(load_image(source), scale)
            save_image(image, workspace.he(sample.name))

        lattice_shape = (image.shape[0] // SUPERPIXEL, image.shape[1] // SUPERPIXEL)
        print(f"[preprocess]   image {image.shape[:2]}, lattice {lattice_shape}")

        mask = compute_tissue_mask(image, config.tissue_mask, patch_size=SUPERPIXEL)
        save_mask(mask, workspace.tissue_mask(sample.name))
        print(f"[preprocess]   tissue superpixels: {int(mask.sum())} / {mask.size}")

        measured: dict[str, np.ndarray] = {}
        for modality in config.predict:
            directory = config.sample_dir(sample.name) / _modality_dir(modality)
            measured[modality] = _measured_mask(
                config, directory, modality, lattice_shape, sample.name
            )
            if not limited_to_positions(modality):
                # Genes: an unlisted tissue superpixel is not left out, it holds zero counts.
                n_listed = int((measured[modality] & mask).sum())
                print(
                    f"[preprocess]   {modality}: positions.npy lists {n_listed} of "
                    f"{int(mask.sum())} tissue superpixels; the others count as zero counts"
                )

        rois = [roi_in_superpixels(config, roi, sample.name) for roi in sample.rois]
        if config.coordinate_type == "raw_pixel":
            for roi in rois:
                print(
                    f"[preprocess]   ROI in superpixels: top_left={list(roi.top_left)}, "
                    f"height={roi.height}, width={roi.width}"
                )
        geometry = Geometry(
            name=sample.name,
            lattice_shape=lattice_shape,
            tissue_mask=mask,
            rois=rois,
            measured=measured,
        )
        _check_roi_bounds(geometry)
        geometry.save(workspace.geometry(sample.name))

        for role in ("train", "calib", "test"):
            counts = {}
            for modality in config.predict:
                counts[modality] = int(geometry.unit_mask(role, modality).sum())
            print(f"[preprocess]   usable {role} units: {counts}")

    pixel_size_raw_per_sample = {
        sample.name: config.sample_pixel_size_raw(sample.name) for sample in config.samples
    }
    write_json(
        {
            "pixel_size_raw": pixel_size_raw_per_sample,
            "pixel_size": config.pixel_size,
            "superpixel": SUPERPIXEL,
            "superpixel_size": config.superpixel_size,
            "coordinate_type": config.coordinate_type,
            "samples": config.sample_names,
        },
        workspace.root / "preprocess.json",
    )


def _modality_dir(modality: str) -> str:
    """Input sub-directory holding a modality's positions and values."""
    return "genes" if modality == "gene" else "cell_types"


def _check_roi_bounds(geometry: Geometry) -> None:
    """Raise if any ROI reaches outside the sample's lattice."""
    rows, columns = geometry.lattice_shape
    for index, roi in enumerate(geometry.rois):
        row0, col0, row1, col1 = roi.box
        if row0 < 0 or col0 < 0 or row1 > rows or col1 > columns:
            raise ValueError(
                f"sample '{geometry.name}': ROI {index} spans rows {row0}:{row1}, columns "
                f"{col0}:{col1}, outside the {geometry.lattice_shape} superpixel lattice"
            )


def _requested_genes(config: Config) -> tuple[list[str] | None, str | None]:
    """Read ``predict_genes`` once: the requested genes (None for all), or a problem message."""
    try:
        return config.predicted_genes(), None
    except (OSError, ValueError) as error:
        return None, f"'predict_genes': {error}"


def _gene_selection_problems(
    requested: Sequence[str] | None, sample: str, names: Sequence[str]
) -> list[str]:
    """Problems with the ``requested`` genes against one sample's gene names; empty when usable."""
    if requested is None:
        return []
    if not requested:
        return ["'predict_genes' is empty; remove it to predict every gene instead"]
    missing = sorted(set(requested) - set(names))
    if missing:
        return [
            f"'predict_genes' names {len(missing)} gene(s) sample '{sample}' does not declare, "
            f"e.g. {missing[:5]}"
        ]
    return []


def gene_target_problems(
    targets: Sequence[Target], available: Sequence[str], where: str, advice: str = ""
) -> list[str]:
    """One message per gene target that needs a gene missing from ``available``.

    ``where`` names the list the genes are missing from and ``advice`` is appended to each message.
    """
    known = set(available)
    problems: list[str] = []
    for target in targets:
        if target.modality != "gene":
            continue
        absent = [name for name in target.channels if name not in known]
        if absent:
            problems.append(
                f"target '{target.name}' needs gene(s) {absent}, which are not in {where}{advice}"
            )
    return problems


def _target_problems(
    config: Config, requested: Sequence[str] | None, names: Sequence[str]
) -> list[str]:
    """One message per gene target that needs a gene the run will not predict."""
    from .inference import load_targets

    if not config.targets_path.exists():
        return []
    targets = load_targets(config.targets_path)
    advice = "; a target can only be tested against a gene the run predicts"
    if requested is None:
        return gene_target_problems(targets, names, "the samples' gene_names.json", advice)
    return gene_target_problems(targets, requested, "'predict_genes'", advice)


def check(config: Config) -> None:
    """Validate the input tree without running anything expensive, raising on any problem.

    Covers file presence, array shapes, gene names and targets, and position bounds once a
    geometry exists.
    """
    workspace = Workspace(config)
    problems = []

    if not config.targets_path.exists():
        problems.append(f"missing target list: {config.targets_path}")

    # An unreadable ``predict_genes`` file is one problem; the checks that need it are skipped.
    requested_genes: list[str] | None = None
    unreadable_genes: str | None = None
    if "gene" in config.predict:
        requested_genes, unreadable_genes = _requested_genes(config)
        if unreadable_genes is not None:
            problems.append(unreadable_genes)

    for sample in config.samples:
        directory = config.sample_dir(sample.name)
        if not (directory / sample.he).exists():
            problems.append(f"sample '{sample.name}': missing image {directory / sample.he}")

        for modality in config.predict:
            sub = directory / _modality_dir(modality)
            positions_path, values_path, names_path = _modality_files(sub, modality)
            for path in (positions_path, values_path, names_path):
                if not path.exists():
                    problems.append(f"sample '{sample.name}': missing {path}")
            if any(not path.exists() for path in (positions_path, values_path, names_path)):
                continue

            values = np.load(values_path, mmap_mode="r")
            names = read_json(names_path)
            try:
                positions = load_positions(sub, config, sample.name)
            except ValueError as error:
                problems.append(f"sample '{sample.name}': {error}")
                continue
            if len(positions) and int(positions.min()) < 0:
                problems.append(
                    f"sample '{sample.name}': {positions_path} holds negative coordinates"
                )
            n_shared = count_shared_superpixels(positions)
            if n_shared:
                problems.append(
                    f"sample '{sample.name}': {positions_path}: "
                    f"{_shared_superpixel_message(config, n_shared, sample.name)}"
                )
            if len(values) != len(positions):
                problems.append(
                    f"sample '{sample.name}': {values_path} has {len(values)} rows but "
                    f"{positions_path} has {len(positions)}"
                )
            if modality == "gene" and (values.ndim != 2 or values.shape[1] != len(names)):
                problems.append(
                    f"sample '{sample.name}': expression has {values.shape[-1]} columns but "
                    f"gene_names.json lists {len(names)} genes"
                )
            if modality == "gene" and unreadable_genes is None:
                # A bad ``predict_genes`` name would otherwise surface only at ``calibrate``, and a
                # target that is not predicted only at ``infer``, after the expensive stages.
                problems.extend(_gene_selection_problems(requested_genes, sample.name, names))
                problems.extend(_target_problems(config, requested_genes, names))
            if modality == "cell_type" and values.ndim != 1:
                problems.append(
                    f"sample '{sample.name}': {values_path} must be 1-D cell-type labels"
                )

            geometry_path = workspace.geometry(sample.name)
            if geometry_path.exists():
                shape = Geometry.load(geometry_path).lattice_shape
                outside = _outside_lattice(positions, shape)
                if outside.any():
                    problems.append(
                        f"sample '{sample.name}': {int(outside.sum())} {modality} positions "
                        f"fall outside the {shape} lattice; "
                        f"{_expected_positions(config, sample.name)}"
                    )

    if problems:
        raise ValueError("input validation failed:\n  " + "\n  ".join(problems))
    print(f"[check] {len(config.samples)} sample(s) validated for modalities {config.predict}")
    print(
        f"[check] superpixel {config.superpixel_size:g} um: H&E rescaled to "
        f"{config.pixel_size:g} um/pixel; positions and ROIs are read as "
        f"{config.coordinate_type} coordinates"
    )
    for sample in config.samples:
        pixel_size_raw = config.sample_pixel_size_raw(sample.name)
        if sample.pixel_size_raw is not None:
            origin = "its own entry"
        else:
            origin = "the global value"
        print(
            f"[check]   {sample.name}: raw H&E at {pixel_size_raw:g} um/pixel ({origin}), "
            f"rescaled by {config.rescale_factor(sample.name):.4g}x"
        )
    print(
        f"[check] {config.regime.replace('_', '-')} run: features '{config.extractor}' "
        f"({config.features.method} extractor, harmony {'on' if config.harmony_enabled else 'off'})"
    )


def _modality_files(directory: Path, modality: str) -> tuple[Path, Path, Path]:
    """Return the ``(positions, values, names)`` paths of one modality."""
    positions = directory / "positions.npy"
    if modality == "gene":
        return positions, directory / "expression.npy", directory / "gene_names.json"
    return positions, directory / "labels.npy", directory / "ct_names.json"
