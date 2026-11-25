"""Stage ``import``: turn the sample folders of an external run into UTOPIA's two stores.

Each sample folder holds what another model predicted for one slide::

    samples/<sample>/
    ├── he.jpg                    # the raw H&E; any name, stated as ``he`` in config.yaml
    └── genes/
        ├── positions.npy         # (N, 2) (row, col) centre of each spot, raw H&E pixels
        ├── prediction.npy        # (N, G) the external model's prediction
        ├── expression.npy        # (N, G) measured values; calibration samples need it
        ├── gene_names.json       # list of G gene names, in column order
        └── clusters_<label>.npy  # (N,) optional histology cluster of each spot

The samples whose ``role`` is ``calibration`` are concatenated into the calibration store and
those whose ``role`` is ``test`` into the prediction store, in configuration order. Positions are
converted from raw H&E pixels to the drawing grid, whose cells are ``superpixel_size`` microns
wide, and the spot footprint, which the config states in microns, becomes the drawing radius on
that grid.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import EXTERNAL_ROLES, Config, SampleSpec
from ..coordinates import raw_pixel_to_superpixel
from ..io import (
    TargetStore,
    read_json,
    read_store_meta,
    store_clusterings,
    write_json,
    write_store_units,
)
from ..preprocess import gene_target_problems
from ..workspace import Workspace
from . import tissue

#: Sub-directory of a sample folder that holds the gene predictions.
GENES_DIR = "genes"
#: Prefix and suffix of a user-supplied cluster label file.
CLUSTER_PREFIX = "clusters_"
CLUSTER_SUFFIX = ".npy"
#: Modality of the stores this stage writes; external sample folders carry genes.
MODALITY = "gene"


@dataclass
class SampleFiles:
    """Paths of one external sample's input files; only some of them need to exist."""

    directory: Path  # samples/<sample>/genes
    positions: Path  # (N, 2) spot centres in raw H&E pixels
    prediction: Path  # (N, G) predicted values
    expression: Path  # (N, G) measured values
    gene_names: Path  # list of G gene names
    he: Path | None  # the raw H&E, or None when the sample states none


def sample_files(config: Config, sample: SampleSpec) -> SampleFiles:
    """Resolve the input paths of one external sample."""
    directory = config.sample_dir(sample.name) / GENES_DIR
    he = None
    if sample.he is not None:
        he = config.sample_dir(sample.name) / sample.he
    return SampleFiles(
        directory=directory,
        positions=directory / "positions.npy",
        prediction=directory / "prediction.npy",
        expression=directory / "expression.npy",
        gene_names=directory / "gene_names.json",
        he=he,
    )


def provided_cluster_labels(config: Config) -> list[str]:
    """Labels of the cluster files the user supplies, or an empty list when there are none.

    A label is only usable when every sample carries it, so this raises if the samples disagree.
    """
    per_sample: dict[str, list[str]] = {}
    for sample in config.samples:
        directory = sample_files(config, sample).directory
        per_sample[sample.name] = sorted(store_clusterings(directory))

    first = per_sample[config.samples[0].name]
    for name, labels in per_sample.items():
        if labels != first:
            raise ValueError(
                f"the samples supply different cluster files: '{config.samples[0].name}' has "
                f"{first} but '{name}' has {labels}. Give every sample the same "
                f"{CLUSTER_PREFIX}<label>{CLUSTER_SUFFIX} files, or none, in which case UTOPIA "
                "clusters the spots itself"
            )
    return first


def _sample_problems(
    config: Config, sample: SampleSpec, reference_genes: Sequence[str] | None
) -> tuple[list[str], list[str] | None]:
    """List the problems in one sample folder, and return its gene names when they are readable."""
    files = sample_files(config, sample)
    problems: list[str] = []
    label = f"sample '{sample.name}'"

    if files.he is not None and not files.he.exists():
        problems.append(f"{label}: missing image {files.he}")

    required = [files.positions, files.prediction, files.gene_names]
    if sample.role == "calibration":
        required.append(files.expression)
    missing = [path for path in required if not path.exists()]
    for path in missing:
        if path == files.expression:
            problems.append(f"{label}: missing {path}; a calibration sample needs measured values")
        else:
            problems.append(f"{label}: missing {path}")
    if missing:
        return problems, None

    positions = np.load(files.positions, mmap_mode="r")
    genes = [str(name) for name in read_json(files.gene_names)]

    if positions.ndim != 2 or positions.shape[1] != 2:
        problems.append(f"{label}: {files.positions} must be (N, 2), got {positions.shape}")
        return problems, genes
    if not np.issubdtype(positions.dtype, np.integer):
        problems.append(
            f"{label}: {files.positions} must hold integers, the (row, col) pixel of each spot "
            f"centre on the raw H&E; got {positions.dtype}"
        )
    elif len(positions) and int(positions.min()) < 0:
        problems.append(f"{label}: {files.positions} holds negative coordinates")

    n_spots = len(positions)
    expected_shape = (n_spots, len(genes))
    prediction = np.load(files.prediction, mmap_mode="r")
    if prediction.shape != expected_shape:
        problems.append(
            f"{label}: {files.prediction} must be (spots, genes) = {expected_shape}, "
            f"got {prediction.shape}"
        )
    else:
        # Checking about 10,000 evenly spaced spots is enough to catch a broken file.
        step = max(1, n_spots // 10000)
        sampled = np.asarray(prediction[::step], dtype=np.float32)
        if not np.isfinite(sampled).all():
            problems.append(f"{label}: {files.prediction} holds NaN or inf, which cannot be ranked")

    if files.expression.exists():
        expression = np.load(files.expression, mmap_mode="r")
        if expression.shape != expected_shape:
            problems.append(
                f"{label}: {files.expression} must be (spots, genes) = {expected_shape}, "
                f"got {expression.shape}"
            )

    if reference_genes is not None and genes != list(reference_genes):
        problems.append(
            f"{label}: gene_names.json differs from that of '{config.samples[0].name}'; every "
            "sample must list the same genes in the same order"
        )

    for path in store_clusterings(files.directory).values():
        labels = np.load(path, mmap_mode="r")
        if labels.shape != (n_spots,):
            problems.append(
                f"{label}: {path} must hold one cluster label per spot, ({n_spots},), "
                f"got {labels.shape}"
            )
        elif not np.issubdtype(labels.dtype, np.integer):
            problems.append(f"{label}: {path} must hold integer cluster labels, got {labels.dtype}")
    return problems, genes


def _target_problems(config: Config, genes: Sequence[str]) -> list[str]:
    """One message per target the samples cannot serve: a missing gene, or a non-gene target."""
    from ..inference import load_targets

    targets = load_targets(config.targets_path)
    problems = gene_target_problems(targets, genes, "the samples' gene_names.json")
    for target in targets:
        if target.modality != MODALITY:
            problems.append(
                f"target '{target.name}' is a {target.modality} target; external sample folders "
                "carry gene predictions only"
            )
    return problems


def test_samples_have_truth(config: Config) -> bool:
    """Whether the test samples carry measured values, which turns on the benchmark diagnostics.

    Raises when only some of them do, because the prediction store holds one truth array.
    """
    with_truth = []
    without_truth = []
    for sample in config.samples_with_role("test"):
        if sample_files(config, sample).expression.exists():
            with_truth.append(sample.name)
        else:
            without_truth.append(sample.name)
    if with_truth and without_truth:
        raise ValueError(
            f"test sample(s) {with_truth} have an expression.npy but {without_truth} do not; "
            "give it to every test sample to benchmark the run, or to none"
        )
    return bool(with_truth)


def check(config: Config) -> None:
    """Validate the sample folders of an external run, raising on any problem.

    Only array headers and a sample of the predictions are read, so this is cheap.
    """
    problems: list[str] = []
    if not config.targets_path.exists():
        problems.append(f"missing target list: {config.targets_path}")

    reference_genes: list[str] | None = None
    for sample in config.samples:
        found, genes = _sample_problems(config, sample, reference_genes)
        problems.extend(found)
        if reference_genes is None:
            reference_genes = genes
    if reference_genes is not None and config.targets_path.exists():
        problems.extend(_target_problems(config, reference_genes))

    labels: list[str] = []
    try:
        labels = provided_cluster_labels(config)
    except ValueError as error:
        problems.append(str(error))
    try:
        test_samples_have_truth(config)
    except ValueError as error:
        problems.append(str(error))

    if not labels:
        # UTOPIA has to cluster the spots itself, which needs every H&E and the UNI weights.
        without_he = [sample.name for sample in config.samples if sample.he is None]
        if without_he:
            problems.append(
                f"sample(s) {without_he} state no 'he' image and supply no cluster labels; give "
                f"each sample an H&E, or a {CLUSTER_PREFIX}<label>{CLUSTER_SUFFIX} file per sample"
            )
        weights = Path(config.features.model_dir) / "pytorch_model.bin"
        if not weights.exists():
            problems.append(
                f"'features.model_dir' must be the folder holding UNI's pytorch_model.bin; "
                f"{weights} does not exist"
            )

    if problems:
        raise ValueError("input validation failed:\n  " + "\n  ".join(problems))

    for role in EXTERNAL_ROLES:
        names = [sample.name for sample in config.samples_with_role(role)]
        print(f"[check] {role} sample(s): {', '.join(names)}")
    if labels:
        print(f"[check] cluster labels supplied by the user: {labels}; UNI is not run")
    else:
        correction = "on" if config.he_batch_effect_correction else "off"
        print(
            f"[check] no cluster labels supplied: UTOPIA clusters the spots on UNI features "
            f"(PCA {config.clustering.pca_dim}, k = {config.n_clusters}, Harmony {correction})"
        )
    spot_shape = config.sample_spot(config.samples[0].name).shape
    print(
        f"[check] maps are drawn on a {config.superpixel_size:g} um grid, each spot as a "
        f"{spot_shape}"
    )
    # The spot size is stated in microns; showing it in pixels of each H&E lets the user see at
    # once whether the value was meant in microns.
    for sample in config.samples:
        spot = config.sample_spot(sample.name)
        pixel_size_raw = config.sample_pixel_size_raw(sample.name)
        width_pixels = spot.crop_width_pixels(pixel_size_raw)
        print(
            f"[check]   {sample.name}: spot {spot.crop_width_um:g} um across = "
            f"{width_pixels:.1f} pixels of its H&E ({pixel_size_raw:g} um/pixel)"
        )


@dataclass
class SampleUnits:
    """One sample's spots, ready to be appended to a store."""

    name: str
    positions: np.ndarray  # (N, 2) int32 spot centres on the drawing grid
    radius: np.ndarray  # (N,) float32 drawing radius of each spot, in grid cells
    prediction: np.ndarray  # (N, G) float32 predicted values
    expression: np.ndarray | None  # (N, G) float32 measured values, or None
    clusters: dict[str, np.ndarray]  # cluster label -> (N,) user-supplied labels
    canvas_shape: tuple[int, int]  # (rows, columns) of the grid this sample is drawn on


def _canvas_shape(
    config: Config,
    sample: SampleSpec,
    image_shape: tuple[int, int] | None,
    positions_raw: np.ndarray,
    positions: np.ndarray,
    radius: float,
) -> tuple[int, int]:
    """Shape of the grid one sample is drawn on.

    With an H&E the grid covers the whole image, so it lines up with the tissue mask; without
    one it reaches just past the outermost spot.
    """
    if image_shape is None:
        reach = int(np.ceil(radius)) + 1
        return int(positions[:, 0].max()) + reach, int(positions[:, 1].max()) + reach

    outside = (positions_raw[:, 0] >= image_shape[0]) | (positions_raw[:, 1] >= image_shape[1])
    if outside.any():
        raise ValueError(
            f"sample '{sample.name}': {int(outside.sum())} of {len(positions_raw)} positions fall "
            f"outside its {image_shape} H&E; positions.npy must hold the (row, col) = (y, x) "
            "pixel of each spot centre on the raw H&E"
        )
    pixel_size_raw = config.sample_pixel_size_raw(sample.name)
    return tissue.grid_shape(image_shape, pixel_size_raw, config.superpixel_size)


def read_sample(
    config: Config, sample: SampleSpec, image_shape: tuple[int, int] | None
) -> SampleUnits:
    """Read one sample folder into the arrays its store needs."""
    files = sample_files(config, sample)
    pixel_size_raw = config.sample_pixel_size_raw(sample.name)
    positions_raw = np.load(files.positions).astype(np.int64)
    positions = raw_pixel_to_superpixel(positions_raw, pixel_size_raw, config.superpixel_size)
    # The drawing radius is half the spot's width, taken from microns to grid cells.
    half_width_um = config.sample_spot(sample.name).half_width_um
    radius = float(half_width_um / config.superpixel_size)

    expression = None
    if files.expression.exists():
        expression = np.load(files.expression).astype(np.float32)

    clusters: dict[str, np.ndarray] = {}
    for label, path in store_clusterings(files.directory).items():
        clusters[label] = np.load(path).astype(np.int16)

    return SampleUnits(
        name=sample.name,
        positions=positions,
        radius=np.full(len(positions), radius, dtype=np.float32),
        prediction=np.load(files.prediction).astype(np.float32),
        expression=expression,
        clusters=clusters,
        canvas_shape=_canvas_shape(config, sample, image_shape, positions_raw, positions, radius),
    )


def _remove_stale_arrays(directory: Path) -> None:
    """Delete the arrays an earlier import or clustering left in a store directory.

    A store that is rewritten with different spots must not keep cluster labels or measured
    values that belonged to the old ones.
    """
    if not directory.exists():
        return
    stale = list(directory.glob(f"{CLUSTER_PREFIX}*{CLUSTER_SUFFIX}"))
    stale.append(directory / "truth.npy")
    for path in stale:
        if path.exists():
            path.unlink()


def write_store(
    config: Config,
    directory: Path,
    units: Sequence[SampleUnits],
    genes: Sequence[str],
    canvas_shapes: dict[str, tuple[int, int]],
    with_truth: bool,
) -> int:
    """Write one calibration or prediction store from the given samples; return its unit count.

    ``sample_index`` counts over every sample of the run, in configuration order, so the two
    stores share one list of sample names and one list of canvases.
    """
    _remove_stale_arrays(directory)
    names = config.sample_names
    n_units = sum(len(sample.positions) for sample in units)

    pred = TargetStore.create(directory, "pred", list(genes), n_units, np.float32)
    pred_writer = pred.writer()
    truth_writer = None
    if with_truth:
        truth = TargetStore.create(directory, "truth", list(genes), n_units, np.float32)
        truth_writer = truth.writer()

    # Stores are gene-major, (genes, units), so each sample's (spots, genes) block is transposed.
    offset = 0
    for sample in units:
        stop = offset + len(sample.positions)
        pred_writer[:, offset:stop] = sample.prediction.T
        if truth_writer is not None:
            truth_writer[:, offset:stop] = sample.expression.T
        offset = stop
    pred_writer.flush()
    if truth_writer is not None:
        truth_writer.flush()

    sample_index = np.concatenate(
        [np.full(len(sample.positions), names.index(sample.name)) for sample in units]
    )
    write_store_units(
        directory,
        positions=np.concatenate([sample.positions for sample in units]),
        sample_index=sample_index,
        radius=np.concatenate([sample.radius for sample in units]),
        sample_names=names,
    )

    labels = sorted(units[0].clusters)
    for label in labels:
        stacked = np.concatenate([sample.clusters[label] for sample in units])
        np.save(directory / f"{CLUSTER_PREFIX}{label}{CLUSTER_SUFFIX}", stacked)

    meta = read_store_meta(directory)
    meta["canvas_shape"] = [list(canvas_shapes[name]) for name in names]
    meta["spot_shape"] = config.sample_spot(names[0]).shape
    meta["grid_um"] = float(config.superpixel_size)
    write_json(meta, directory / "meta.json")
    return n_units


def store_directory(config: Config, workspace: Workspace, role: str) -> Path:
    """Store an external role's samples are written to: calibration or predictions."""
    if role == "calibration":
        return workspace.calibration_dir(config.model, MODALITY)
    return workspace.prediction_dir(config.model, MODALITY)


def run(config: Config) -> None:
    """Run the import stage: validate the sample folders, mask the tissue, write both stores."""
    check(config)
    workspace = Workspace(config)
    workspace.prepare()
    print(
        f"[import] external model '{config.model}': positions are converted from raw H&E "
        f"pixels to a {config.superpixel_size:g} um drawing grid"
    )

    genes = [str(name) for name in read_json(sample_files(config, config.samples[0]).gene_names)]
    benchmark = test_samples_have_truth(config)

    units_by_name: dict[str, SampleUnits] = {}
    for sample in config.samples:
        image_shape = tissue.prepare(config, workspace, sample)
        units = read_sample(config, sample, image_shape)
        units_by_name[sample.name] = units
        print(
            f"[import]   {sample.name} ({sample.role}): {len(units.positions)} spots, drawn with "
            f"radius {float(units.radius[0]):.2f} cells on a {units.canvas_shape} canvas"
        )
    canvas_shapes = {name: units.canvas_shape for name, units in units_by_name.items()}

    for role in EXTERNAL_ROLES:
        role_units = [units_by_name[sample.name] for sample in config.samples_with_role(role)]
        # Calibration always carries truth; the test side only when the user supplied it.
        with_truth = role == "calibration" or benchmark
        directory = store_directory(config, workspace, role)
        n_units = write_store(config, directory, role_units, genes, canvas_shapes, with_truth)
        print(f"[import] {role}: {n_units} spots x {len(genes)} genes -> {directory}")

    labels = provided_cluster_labels(config)
    if labels:
        from .cluster import render_cluster_maps

        for label in labels:
            render_cluster_maps(config, workspace, label)
        print(f"[import] cluster labels supplied by the user: {labels}")
    else:
        print("[import] no cluster labels supplied; the features and cluster stages make them")
    if benchmark:
        print("[import] the test samples carry measured values, so the run is benchmarked")
