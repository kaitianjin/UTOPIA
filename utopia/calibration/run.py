"""Stage 4: build calibration predictions, then predict the test area.

Each spatial fold trains a model with that fold removed and predicts only the fold's centre, so
calibration predictions are held out; the full model predicts the test area. Predictions are stored
gene-major, ``(n_channels, n_units)`` float16, and do not depend on any inference-time setting.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from ..coordinates import load_positions
from ..geometry import Geometry, gather_units, limited_to_positions, positions_to_rows
from ..io import (
    UNIT_ARRAYS,
    TargetStore,
    create_memmap,
    open_memmap,
    read_json,
    write_store_units,
)
from ..models.default import predict as run_model
from ..models.default import train_classifier, train_regressor
from ..workspace import Workspace
from .folds import boxes_to_array, generate_folds, inside_box, paint_fold_ids

#: Genes are processed in blocks of this many columns when transposing between
#: the unit-major inputs and the gene-major stores, bounding peak memory.
GENE_BLOCK = 512
#: Upper quantile used to rescale each gene's counts into ``[0, 1]`` for training.
UPPER_QUANTILE = 99.99
#: Units copied per chunk when materialising counts, bounding the temporary buffer.
UNIT_BLOCK = 100_000


@dataclass
class UnitTable:
    """One role's usable units, flattened across samples."""

    sample_index: np.ndarray  # (N,) sample id per unit, non-decreasing
    positions: np.ndarray  # (N, 2) scale-1 superpixel position per unit
    value_rows: np.ndarray  # (N,) row in the sample's input value array, -1 if not listed there
    # (N,) whether ground truth exists for the unit. Genes: always, since a unit that is not
    # listed holds zero counts. Cell types: only where the unit is listed (annotated).
    evaluable: np.ndarray

    def __len__(self) -> int:
        """Number of units in the table."""
        return len(self.positions)

    def sample_slice(self, sample: int) -> slice:
        """Return the contiguous slice of rows belonging to ``sample``."""
        rows = np.flatnonzero(self.sample_index == sample)
        return slice(0, 0) if len(rows) == 0 else slice(int(rows[0]), int(rows[-1]) + 1)

    def spans_by_sample(self, geometries: Sequence[Geometry]) -> Iterator[tuple[Geometry, slice]]:
        """Yield each sample that has units together with its slice of rows, in sample order."""
        for sample, geometry in enumerate(geometries):
            span = self.sample_slice(sample)
            if span.stop > span.start:
                yield geometry, span


def _input_dir(config: Config, sample: str, modality: str) -> Path:
    """Input directory of one sample's modality."""
    return config.sample_dir(sample) / ("genes" if modality == "gene" else "cell_types")


def _channel_names(config: Config, modality: str) -> list[str]:
    """Read the channel names, requiring every sample to declare the same ones."""
    filename = "gene_names.json" if modality == "gene" else "ct_names.json"
    names: list[str] | None = None
    for sample in config.samples:
        current = read_json(_input_dir(config, sample.name, modality) / filename)
        if names is None:
            names = current
        elif current != names:
            raise ValueError(
                f"sample '{sample.name}' declares different {modality} names than "
                f"'{config.samples[0].name}'; all samples must share one channel list"
            )
    return names or []


def _predicted_channels(config: Config, modality: str) -> tuple[list[str], np.ndarray | None]:
    """The channels this run predicts and stores, and their columns in the samples' channel order.

    Columns are None when every input channel is predicted, which is always so for cell types.
    """
    channels = _channel_names(config, modality)
    requested = config.predicted_genes() if modality == "gene" else None
    if requested is None:
        return channels, None

    lookup = {name: column for column, name in enumerate(channels)}
    missing = sorted({name for name in requested if name not in lookup})
    if missing:
        raise ValueError(
            f"'predict_genes' names {len(missing)} gene(s) the samples do not declare, "
            f"e.g. {missing[:5]}; they must appear in gene_names.json"
        )
    if not requested:
        raise ValueError("'predict_genes' is empty; remove it to predict every gene instead")
    # The samples' order, not the request's, so store rows stay aligned with the
    # expression columns and the gather reads ascending columns.
    columns = np.array(sorted({lookup[name] for name in requested}), dtype=np.int64)
    return [channels[column] for column in columns], columns


def _build_table(
    config: Config, geometries: Sequence[Geometry], role: str, modality: str, restrict: bool
) -> UnitTable:
    """Collect the usable units of one role (``train``, ``calib`` or ``test``) across samples.

    The table holds every tissue superpixel of the role's region; ``restrict`` keeps only those
    with ground truth.
    """
    sample_index, positions = gather_units(geometries, role, modality if restrict else None)
    value_rows = np.full(len(positions), -1, dtype=np.int64)
    for index, geometry in enumerate(geometries):
        rows = np.flatnonzero(sample_index == index)
        if len(rows) == 0:
            continue
        directory = _input_dir(config, geometry.name, modality)
        source = load_positions(directory, config, geometry.name)
        lookup = positions_to_rows(source, geometry.lattice_shape)
        value_rows[rows] = lookup[positions[rows, 0], positions[rows, 1]]

    if limited_to_positions(modality):
        evaluable = value_rows >= 0
    else:
        evaluable = np.ones(len(positions), dtype=bool)
    return UnitTable(sample_index, positions, value_rows, evaluable)


def _gather_features(
    workspace: Workspace,
    config: Config,
    geometries: Sequence[Geometry],
    table: UnitTable,
    path: Path,
) -> np.memmap:
    """Copy the features of a unit table into one contiguous scratch array."""
    extractor = config.extractor
    first_features = np.load(workspace.features(geometries[0].name, extractor), mmap_mode="r")
    dimension = int(first_features.shape[1])
    out = create_memmap(path, (len(table), dimension), np.float16)
    for geometry, span in table.spans_by_sample(geometries):
        features = np.load(workspace.features(geometry.name, extractor), mmap_mode="r")
        rows = geometry.feature_rows(table.positions[span], config.features_cover_lattice)
        out[span] = features[rows]
    out.flush()
    return out


def _gather_counts(
    config: Config,
    geometries: Sequence[Geometry],
    table: UnitTable,
    n_genes: int,
    path: Path,
    columns: np.ndarray | None = None,
) -> np.memmap:
    """Copy the raw gene counts of a unit table into a ``(units, n_genes)`` uint16 scratch array.

    Units that ``genes/positions.npy`` does not list hold zero counts; ``columns`` selects the
    predicted genes.
    """
    out = create_memmap(path, (len(table), n_genes), np.uint16)
    for geometry, span in table.spans_by_sample(geometries):
        expression = np.load(
            _input_dir(config, geometry.name, "gene") / "expression.npy", mmap_mode="r"
        )
        for start in range(span.start, span.stop, UNIT_BLOCK):
            stop = min(start + UNIT_BLOCK, span.stop)
            rows = table.value_rows[start:stop]
            usable = rows >= 0
            block = np.zeros((stop - start, n_genes), dtype=np.uint16)
            if columns is None:
                block[usable] = expression[rows[usable]]
            else:
                # ``np.ix_`` selects the wanted columns during the read, so no
                # temporary holding every gene is built for the block.
                block[usable] = expression[np.ix_(rows[usable], columns)]
            out[start:stop] = block
    out.flush()
    return out


def _gene_thresholds(counts: np.ndarray, n_genes: int) -> np.ndarray:
    """Per-gene ``(lower, upper)`` count thresholds used to rescale counts into ``[0, 1]``.

    The lower threshold is zero by construction, so a prediction is mapped back to counts by
    multiplying by the upper threshold alone.
    """
    thresholds = np.zeros((n_genes, 2), dtype=np.float32)
    for start in range(0, n_genes, GENE_BLOCK):
        block = np.asarray(counts[:, start : start + GENE_BLOCK], dtype=np.float32)
        thresholds[start : start + block.shape[1], 1] = np.percentile(block, UPPER_QUANTILE, axis=0)
    return thresholds


def _scale_counts(counts: np.ndarray, thresholds: np.ndarray, path: Path) -> np.memmap:
    """Write the counts, clipped and rescaled into ``[0, 1]`` per gene, to a scratch array."""
    out = create_memmap(path, counts.shape, np.float16)
    for start in range(0, counts.shape[1], GENE_BLOCK):
        stop = min(start + GENE_BLOCK, counts.shape[1])
        block = np.asarray(counts[:, start:stop], dtype=np.float32)
        lower, upper = thresholds[start:stop, 0], thresholds[start:stop, 1]
        span = np.where(upper > lower, upper - lower, 1.0)
        out[:, start:stop] = ((np.clip(block, lower, upper) - lower) / span).astype(np.float16)
    out.flush()
    return out


@dataclass
class TrainingInputs:
    """One modality's training set, materialised in scratch and ready to fit on."""

    table: UnitTable
    channels: list[str]
    features: np.memmap  # (N, D) float16
    targets: np.memmap  # (N, G) float16 rescaled counts, or (N,) int64 cell-type labels
    hyper: dict[str, object]  # hyper-parameters handed to the trainer
    thresholds: np.ndarray | None = None  # (G, 2) gene rescaling thresholds; None for cell types
    columns: np.ndarray | None = None  # input columns of ``channels``; None when all are used

    @property
    def paths(self) -> tuple[str, str]:
        """Filenames of the feature and target arrays, for worker processes."""
        return str(self.features.filename), str(self.targets.filename)


def training_inputs(
    config: Config,
    workspace: Workspace,
    geometries: Sequence[Geometry],
    modality: str,
    scratch: Path,
    thresholds: np.ndarray | None = None,
) -> TrainingInputs:
    """Copy one modality's training features and targets into ``scratch``.

    Passing recorded ``thresholds`` keeps the gene rescaling of an earlier run; None recomputes it.
    """
    channels, columns = _predicted_channels(config, modality)
    table = _build_table(config, geometries, "train", modality, restrict=True)
    scratch.mkdir(parents=True, exist_ok=True)
    features = _gather_features(workspace, config, geometries, table, scratch / "train_x.npy")

    if modality == "gene":
        counts = _gather_counts(
            config, geometries, table, len(channels), scratch / "train_counts.npy", columns
        )
        if thresholds is None:
            thresholds = _gene_thresholds(counts, len(channels))
        targets = _scale_counts(counts, thresholds, scratch / "train_y.npy")
        del counts
        (scratch / "train_counts.npy").unlink()
        hyper: dict[str, object] = {
            "epochs": config.default.epochs,
            "batch_size": config.default.batch_size,
            "learning_rate": config.default.learning_rate,
            "dropout": config.default.dropout,
            "loss": config.default.loss,
            "beta": config.default.beta,
            "seed": config.default.seed,
        }
    else:
        targets = create_memmap(scratch / "train_y.npy", (len(table),), np.int64)
        for geometry, span in table.spans_by_sample(geometries):
            labels = np.load(_input_dir(config, geometry.name, "cell_type") / "labels.npy")
            targets[span] = labels[table.value_rows[span]]
        targets.flush()
        hyper = {
            "epochs": config.default.epochs,
            "batch_size": config.default.batch_size,
            "learning_rate": config.default.learning_rate,
            "dropout": max(config.default.dropout, 0.3),
            "seed": config.default.seed,
            "n_classes": len(channels),
        }
    return TrainingInputs(table, channels, features, targets, hyper, thresholds, columns)


def fit(
    modality: str, features: object, targets: object, rows: np.ndarray, hyper: dict[str, object]
) -> object:
    """Train one model of either modality on ``rows``; ``hyper`` is consumed."""
    if modality == "gene":
        return train_regressor(features, targets, rows, **hyper)
    return train_classifier(
        features, targets, rows, n_classes=int(hyper.pop("n_classes")), **hyper
    )


def _fold_worker(
    feature_path: str,
    target_path: str,
    calib_feature_path: str,
    train_rows: np.ndarray,
    calib_rows: np.ndarray,
    modality: str,
    hyper: dict[str, object],
    predict_batch_size: int,
    fold_id: int,
    queue: object,
) -> None:
    """Train one fold's model in a separate process and put ``(fold_id, predictions)`` on ``queue``.

    Inputs are opened as memory maps from their paths, so concurrent folds share one page cache.
    """
    predictions = _train_and_predict(
        feature_path,
        target_path,
        calib_feature_path,
        train_rows,
        calib_rows,
        modality,
        hyper,
        predict_batch_size,
    )
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    queue.put((fold_id, np.ascontiguousarray(predictions)))


def _train_and_predict(
    feature_path: str,
    target_path: str,
    calib_feature_path: str,
    train_rows: np.ndarray,
    calib_rows: np.ndarray,
    modality: str,
    hyper: dict[str, object],
    predict_batch_size: int,
) -> np.ndarray:
    """Train on ``train_rows`` and return float16 predictions for ``calib_rows``."""
    features = open_memmap(feature_path, mode="r")
    targets = open_memmap(target_path, mode="r")
    calib_features = open_memmap(calib_feature_path, mode="r")

    model = fit(modality, features, targets, train_rows, hyper)
    predictions = run_model(model, calib_features, calib_rows, batch_size=predict_batch_size)
    return predictions.astype(np.float16)


def _run_folds(
    jobs: list[tuple[int, np.ndarray, np.ndarray]],
    paths: tuple[str, str, str],
    modality: str,
    hyper: dict[str, object],
    store: TargetStore,
    parallel_processes: int,
    predict_batch_size: int,
) -> None:
    """Train every fold and write its predictions into the calibration store.

    ``jobs`` are ``(fold_id, train_rows, calib_rows)`` triples; ``parallel_processes`` folds train
    at once.
    """
    from tqdm import tqdm

    # A fold's calibration rows are also the store columns its predictions fill.
    columns_of = {fold_id: calib_rows for fold_id, _, calib_rows in jobs}
    writer = store.writer()
    progress = tqdm(total=len(jobs), desc="[calibrate] folds", unit="fold")

    if parallel_processes <= 1:
        for fold_id, train_rows, calib_rows in jobs:
            predictions = _train_and_predict(
                *paths, train_rows, calib_rows, modality, dict(hyper), predict_batch_size
            )
            writer[:, columns_of[fold_id]] = predictions.T
            progress.update(1)
    else:
        import torch.multiprocessing as multiprocessing

        multiprocessing.set_start_method("spawn", force=True)
        for start in range(0, len(jobs), parallel_processes):
            batch = jobs[start : start + parallel_processes]
            queue = multiprocessing.Queue()
            processes = []
            for fold_id, train_rows, calib_rows in batch:
                process = multiprocessing.Process(
                    target=_fold_worker,
                    args=(
                        *paths,
                        train_rows,
                        calib_rows,
                        modality,
                        dict(hyper),
                        predict_batch_size,
                        fold_id,
                        queue,
                    ),
                )
                process.start()
                processes.append(process)
            # Read every result before joining: a process blocks on put() until the
            # queue is drained, so joining first would deadlock.
            for _ in processes:
                fold_id, predictions = queue.get()
                writer[:, columns_of[fold_id]] = predictions.T
                progress.update(1)
            for process in processes:
                process.join()

    progress.close()
    writer.flush()


def _fold_jobs(
    config: Config,
    geometries: Sequence[Geometry],
    workspace: Workspace,
    train: UnitTable,
    calib: UnitTable,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], np.ndarray]:
    """Generate folds and work out what each one trains on and predicts.

    Returns the ``(fold_id, train_rows, calib_rows)`` jobs and the calibration rows no fold
    covers, which the full model predicts.
    """
    source = config.cluster_source
    fold_key = f"k{config.fold_n_clusters}_scale1"
    fold_train_rows: list[tuple[int, np.ndarray]] = []
    assigned = np.full(len(calib), -1, dtype=np.int64)
    saved: dict[str, np.ndarray] = {}
    next_id = 0

    for index, geometry in enumerate(geometries):
        with np.load(workspace.clusters(geometry.name, source)) as data:
            cluster_image = data[fold_key]
        observed = geometry.unit_mask("train", None)
        calib_span = calib.sample_slice(index)

        for roi_number, roi in enumerate(geometry.rois):
            if not {"train", "calib"} <= set(roi.roles):
                continue
            fold_boxes, centre_boxes = generate_folds(
                cluster_image,
                observed,
                roi.box,
                fold_widths=config.calibration.fold_widths,
                safe_boundary=config.calibration.safe_boundary,
            )
            saved[f"{geometry.name}_roi{roi_number}_fold"] = boxes_to_array(fold_boxes)
            saved[f"{geometry.name}_roi{roi_number}_centre"] = boxes_to_array(centre_boxes)
            print(f"[calibrate] {geometry.name} ROI {roi_number}: {len(fold_boxes)} folds")

            ids = paint_fold_ids(geometry.lattice_shape, centre_boxes, first_id=next_id)
            if calib_span.stop > calib_span.start:
                positions = calib.positions[calib_span]
                local = ids[positions[:, 0], positions[:, 1]]
                covered = local >= 0
                assigned[np.arange(calib_span.start, calib_span.stop)[covered]] = local[covered]

            for offset, box in enumerate(fold_boxes):
                excluded = (train.sample_index == index) & inside_box(train.positions, box)
                train_rows = np.flatnonzero(~excluded).astype(np.int64)
                fold_train_rows.append((next_id + offset, train_rows))
            next_id += len(fold_boxes)

    np.savez_compressed(workspace.folds(config.model), **saved)

    # A fold whose centre holds no calibration unit has nothing to predict and is skipped.
    jobs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for fold_id, train_rows in fold_train_rows:
        calib_rows = np.flatnonzero(assigned == fold_id).astype(np.int64)
        if len(calib_rows):
            jobs.append((fold_id, train_rows, calib_rows))
    return jobs, np.flatnonzero(assigned < 0).astype(np.int64)


def _drop_stale_arrays(directory: Path, keep: Sequence[str]) -> None:
    """Remove value arrays of a store directory that this run will not rewrite.

    The directory has one channel list, so a leftover array would be read against the wrong rows.
    """
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.npy")):
        if path.stem in UNIT_ARRAYS or path.stem.startswith("clusters_"):
            continue
        if path.stem not in keep:
            print(f"[calibrate]   dropping stale {path.name}; this run does not write it")
            path.unlink()


def _calibrate_modality(
    config: Config, workspace: Workspace, geometries: Sequence[Geometry], modality: str
) -> None:
    """Build calibration and test predictions for one modality."""
    print(f"[calibrate] {modality}: starting")

    scratch = workspace.scratch(config.model)
    inputs = training_inputs(config, workspace, geometries, modality, scratch)
    train, channels, hyper = inputs.table, inputs.channels, inputs.hyper
    # Unrestricted: a coarse unit sums its prediction over every tissue superpixel it covers,
    # so the stores must hold all of them, with or without ground truth.
    calib = _build_table(config, geometries, "calib", modality, restrict=False)
    test = _build_table(config, geometries, "test", modality, restrict=False)
    print(f"[calibrate]   {len(channels)} channel(s); units: {len(train)} train, "
          f"{len(calib)} calib, {len(test)} test")
    if modality == "gene":
        for role, table in (("train", train), ("calib", calib), ("test", test)):
            n_unlisted = int((table.value_rows < 0).sum())
            print(f"[calibrate]   {role}: {n_unlisted} of {len(table)} unit(s) are not listed in "
                  "genes/positions.npy and count as zero counts")

    calib_x = _gather_features(workspace, config, geometries, calib, scratch / "calib_x.npy")
    test_x = _gather_features(workspace, config, geometries, test, scratch / "test_x.npy")

    calib_dir = workspace.calibration_dir(config.model, modality)
    predict_dir = workspace.prediction_dir(config.model, modality)
    store_meta = {}
    if modality == "gene":
        store_meta["thresholds"] = {
            name: [float(low), float(high)]
            for name, (low, high) in zip(channels, inputs.thresholds)
        }

    sample_names = [geometry.name for geometry in geometries]
    _drop_stale_arrays(calib_dir, keep=("pred", "truth", "truth_labels"))
    _drop_stale_arrays(predict_dir, keep=("pred", "truth", "truth_labels") if config.benchmark
                       else ("pred",))
    calib_store = TargetStore.create(
        calib_dir, "pred", channels, len(calib), np.float16, store_meta
    )
    write_store_units(
        calib_dir,
        positions=calib.positions,
        sample_index=calib.sample_index,
        evaluable=calib.evaluable,
        sample_names=sample_names,
    )

    paths = (*inputs.paths, str(scratch / "calib_x.npy"))
    jobs, uncovered = _fold_jobs(config, geometries, workspace, train, calib)
    if jobs:
        _run_folds(
            jobs,
            paths,
            modality,
            hyper,
            calib_store,
            config.calibration.parallel_processes,
            config.default.predict_batch_size,
        )

    print("[calibrate]   training full model")
    full_model = fit(modality, inputs.features, inputs.targets, np.arange(len(train)), dict(hyper))

    if len(uncovered):
        print(f"[calibrate]   full model covers {len(uncovered)} calibration unit(s) "
              "outside any fold")
        predictions = run_model(
            full_model, calib_x, uncovered, batch_size=config.default.predict_batch_size
        )
        writer = calib_store.writer()
        writer[:, uncovered] = predictions.astype(np.float16).T
        writer.flush()

    print(f"[calibrate]   predicting {len(test)} test unit(s)")
    test_store = TargetStore.create(
        predict_dir, "pred", channels, len(test), np.float16, store_meta
    )
    writer = test_store.writer()
    for start in range(0, len(test), config.default.predict_batch_size):
        rows = np.arange(start, min(start + config.default.predict_batch_size, len(test)))
        writer[:, start : start + len(rows)] = run_model(
            full_model, test_x, rows, batch_size=config.default.predict_batch_size
        ).astype(np.float16).T
    writer.flush()
    write_store_units(
        predict_dir,
        positions=test.positions,
        sample_index=test.sample_index,
        evaluable=test.evaluable,
        sample_names=sample_names,
    )

    _write_ground_truth(
        config, geometries, calib, calib_dir, channels, modality, scratch, inputs.columns
    )
    if config.benchmark:
        _write_ground_truth(
            config, geometries, test, predict_dir, channels, modality, scratch, inputs.columns
        )

    del calib_x, test_x, inputs
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"[calibrate] {modality}: done")


def _write_ground_truth(
    config: Config,
    geometries: Sequence[Geometry],
    table: UnitTable,
    directory: Path,
    channels: Sequence[str],
    modality: str,
    scratch: Path,
    columns: np.ndarray | None = None,
) -> None:
    """Write the observed values of a unit table next to its predictions, with the same channels."""
    if modality == "gene":
        store = TargetStore.create(directory, "truth", channels, len(table), np.uint16)
        counts = _gather_counts(
            config, geometries, table, len(channels), scratch / "truth_counts.npy", columns
        )
        # The store is gene-major, so the unit-major counts are transposed block by block.
        writer = store.writer()
        for start in range(0, counts.shape[1], GENE_BLOCK):
            stop = min(start + GENE_BLOCK, counts.shape[1])
            writer[start:stop] = np.asarray(counts[:, start:stop]).T
        writer.flush()
        del counts
        (scratch / "truth_counts.npy").unlink()
    else:
        labels = np.full(len(table), -1, dtype=np.int16)
        for geometry, span in table.spans_by_sample(geometries):
            source = np.load(_input_dir(config, geometry.name, "cell_type") / "labels.npy")
            rows = table.value_rows[span]
            usable = rows >= 0
            labels[np.arange(span.start, span.stop)[usable]] = source[rows[usable]]
        np.save(directory / "truth_labels.npy", labels)


def run(config: Config) -> None:
    """Run the calibration stage for every configured modality."""
    workspace = Workspace(config)
    geometries = workspace.load_geometries()
    for modality in config.predict:
        _calibrate_modality(config, workspace, geometries, modality)
