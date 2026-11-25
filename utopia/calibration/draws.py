"""Stage 5: train and store the models the sampling-based methods draw from.

``mc_dropout`` makes K stochastic passes through one model, ``ensemble`` uses K models trained with
different seeds, and ``resampling`` uses K models trained on bootstrap resamples of the training
units. Inference reads the ``(targets, K, units)`` store of scale-1 count-scale draws built here.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from ..geometry import Geometry
from ..io import TargetStore, open_memmap, read_store_meta, write_json
from ..models.default import default_device, enable_dropout, rebuild
from ..workspace import Workspace
from .run import TrainingInputs, _predicted_channels, fit, training_inputs

#: Bundle each sampling method draws from; ``mc_dropout`` perturbs the single ``full`` model.
BUNDLE_OF_METHOD: dict[str, str] = {
    "mc_dropout": "full",
    "ensemble": "ensemble",
    "resampling": "resampling",
}


def _member_rows(bundle: str, n_rows: int, seed: int) -> np.ndarray:
    """Training rows of one bundle member.

    ``resampling`` draws ``n_rows`` rows with replacement; the other bundles use every row.
    """
    if bundle != "resampling":
        return np.arange(n_rows, dtype=np.int64)
    return np.random.default_rng(seed).integers(0, n_rows, size=n_rows, dtype=np.int64)


def _weights(model: nn.Module) -> dict[str, np.ndarray]:
    """A trained model's weights as numpy arrays, which can be passed safely between processes."""
    return {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}


def _member_worker(
    feature_path: str,
    target_path: str,
    rows: np.ndarray,
    modality: str,
    hyper: dict[str, object],
    member: int,
    queue: object,
) -> None:
    """Train one bundle member in its own process and put ``(member, weights)`` on ``queue``.

    Inputs are opened as memory maps from their paths, so concurrent members share one page cache.
    """
    model = fit(modality, open_memmap(feature_path, mode="r"), open_memmap(target_path, mode="r"),
                rows, hyper)
    queue.put((member, _weights(model)))


def _train_bundle(
    config: Config,
    modality: str,
    bundle: str,
    inputs: TrainingInputs,
    path: Path,
    parallel_processes: int,
) -> None:
    """Train every member of one bundle (``full``, ``ensemble`` or ``resampling``) and save it.

    The bundle is one ``.pt`` file at ``path`` holding every member's weights.
    """
    from tqdm import tqdm

    n_rows = len(inputs.table)
    if bundle == "full":
        seeds = [int(config.default.seed)]
    else:
        seeds = [config.sampling.seed + member for member in range(config.sampling.n_draws)]

    jobs = [(member, seed, _member_rows(bundle, n_rows, seed)) for member, seed in enumerate(seeds)]
    states: dict[int, dict[str, np.ndarray]] = {}
    progress = tqdm(total=len(jobs), desc=f"[sample] {modality} {bundle}", unit="model")

    def hyper_for(seed: int) -> dict[str, object]:
        """The training hyper-parameters with one member's seed."""
        return {**inputs.hyper, "seed": seed}

    if parallel_processes <= 1:
        for member, seed, rows in jobs:
            model = fit(modality, inputs.features, inputs.targets, rows, hyper_for(seed))
            states[member] = _weights(model)
            progress.update(1)
    else:
        import torch.multiprocessing as multiprocessing

        multiprocessing.set_start_method("spawn", force=True)
        feature_path, target_path = inputs.paths
        for start in range(0, len(jobs), parallel_processes):
            queue = multiprocessing.Queue()
            processes = []
            for member, seed, rows in jobs[start : start + parallel_processes]:
                process = multiprocessing.Process(
                    target=_member_worker,
                    args=(
                        feature_path, target_path, rows, modality, hyper_for(seed), member, queue
                    ),
                )
                process.start()
                processes.append(process)
            # Drain before joining: a worker blocks in put() until its result is read.
            for _ in processes:
                member, state = queue.get()
                states[member] = state
                progress.update(1)
            for process in processes:
                process.join()
    progress.close()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "bundle": bundle,
            "modality": modality,
            "seeds": seeds,
            "input_dim": int(inputs.features.shape[1]),
            "output_dim": len(inputs.channels),
            "dropout": float(inputs.hyper["dropout"]),
            "channels": list(inputs.channels),
            "state_dicts": [
                {name: torch.as_tensor(array) for name, array in states[member].items()}
                for member in range(len(jobs))
            ],
        },
        path,
    )
    print(f"[sample]   wrote {len(jobs)} model(s) to {path}")


def _needs_training(path: Path, channels: Sequence[str]) -> bool:
    """Whether a bundle is absent, or was trained for a channel list other than ``channels``."""
    if not path.exists():
        return True
    stored = torch.load(path, map_location="cpu")["channels"]
    if list(stored) == list(channels):
        return False
    print(f"[sample]   {path.name} was trained for {len(stored)} channel(s), the store now holds "
          f"{len(channels)}; retraining")
    return True


def run(config: Config) -> None:
    """Train the draw models every configured sampling method needs."""
    methods = config.sampling_methods
    if not methods:
        print("[sample] no sampling-based method configured; nothing to train")
        return

    workspace = Workspace(config)
    geometries = workspace.load_geometries()
    bundles = [BUNDLE_OF_METHOD[method] for method in methods]
    print(f"[sample] methods {methods} -> bundle(s) {bundles}")

    for modality in config.predict:
        # Reuse the rescaling the calibration stage recorded, so that a draw and
        # the stored point prediction of the same unit are on the same scale.
        meta = read_store_meta(workspace.calibration_dir(config.model, modality))
        channels = meta["channels"]
        # The channel list comes from the config and the thresholds from the store; if the
        # two differ, each gene would be rescaled by another gene's thresholds.
        current, _ = _predicted_channels(config, modality)
        if list(current) != list(channels):
            raise ValueError(
                f"{modality}: the calibration store holds {len(channels)} channel(s) but this "
                f"config asks for {len(current)}; 'predict_genes' changed since `calibrate` ran, "
                f"and a draw model must be trained on exactly what was calibrated. Re-run "
                f"`calibrate` first."
            )
        thresholds = None
        if modality == "gene":
            stored = meta["thresholds"]
            thresholds = np.array([stored[name] for name in channels], dtype=np.float32)

        missing = [
            bundle
            for bundle in bundles
            if _needs_training(workspace.draw_models(config.model, modality, bundle), channels)
        ]
        if not missing:
            print(f"[sample] {modality}: every bundle already present")
            continue

        scratch = workspace.scratch(config.model) / "draws"
        inputs = training_inputs(config, workspace, geometries, modality, scratch, thresholds)
        print(f"[sample] {modality}: {len(inputs.table)} training unit(s), "
              f"{config.sampling.n_draws} draw(s)")
        for bundle in missing:
            _train_bundle(
                config,
                modality,
                bundle,
                inputs,
                workspace.draw_models(config.model, modality, bundle),
                config.calibration.parallel_processes,
            )
        del inputs
        shutil.rmtree(scratch, ignore_errors=True)
    print("[sample] done")


class DrawModels:
    """The stored draw models of one method, evaluated one draw at a time.

    For ``mc_dropout`` (``stochastic``) a draw is a seeded dropout pass through the single model;
    otherwise each draw is a distinct model.
    """

    def __init__(self, config: Config, workspace: Workspace, modality: str, method: str) -> None:
        """Load the bundle backing ``method`` and rebuild its models on the default device."""
        bundle = torch.load(self.check(config, workspace, modality, method), map_location="cpu")
        dropout = bundle["dropout"] if config.sampling.dropout is None else config.sampling.dropout
        self.channels: list[str] = bundle["channels"]
        self.models: list[nn.Module] = [
            rebuild(modality, state, bundle["input_dim"], bundle["output_dim"], dropout)
            for state in bundle["state_dicts"]
        ]
        self.stochastic = method == "mc_dropout"
        if not self.stochastic and len(self.models) != config.sampling.n_draws:
            print(
                f"[infer]   note: {method} was trained with {len(self.models)} draw(s), not "
                f"sampling.n_draws={config.sampling.n_draws}; using the {len(self.models)} stored"
            )
        self.n_draws = config.sampling.n_draws if self.stochastic else len(self.models)
        self.seed = config.sampling.seed
        self.device = default_device()
        for model in self.models:
            model.to(self.device).eval()
            if self.stochastic:
                enable_dropout(model)

    @staticmethod
    def check(config: Config, workspace: Workspace, modality: str, method: str) -> Path:
        """Return the bundle path backing ``method``, raising if the stage never ran."""
        path = workspace.draw_models(config.model, modality, BUNDLE_OF_METHOD[method])
        if not path.exists():
            raise FileNotFoundError(
                f"method '{method}' needs {path}, which the 'sample' stage writes; "
                f"run `utopia run <config> --stage sample` first"
            )
        return path

    def channel_index(self, names: Sequence[str]) -> np.ndarray:
        """Output columns of the named channels."""
        lookup = {name: column for column, name in enumerate(self.channels)}
        return np.array([lookup[name] for name in names], dtype=np.int64)

    def draw(
        self, index: int, batches: Iterator[tuple[int, int, np.ndarray]], n_units: int,
        columns: np.ndarray
    ) -> np.ndarray:
        """Run draw ``index`` over the whole test area, keeping only the output ``columns``.

        ``batches`` yields ``(start, stop, features)`` blocks; the result is a
        ``(n_units, len(columns))`` float32 array.
        """
        model = self.models[0] if self.stochastic else self.models[index]
        if self.stochastic:
            torch.manual_seed(self.seed + index)
        selected = torch.as_tensor(columns, device=self.device)
        out = np.empty((n_units, len(columns)), dtype=np.float32)
        with torch.inference_mode():
            for start, stop, features in batches:
                x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=self.device)
                out[start:stop] = model(x).index_select(1, selected).float().cpu().numpy()
        return out


class TestFeatures:
    """The feature rows of the stored test units, read from the samples' feature arrays.

    Units are grouped by sample, so a batch is one read from one memory-mapped array.
    """

    def __init__(
        self,
        workspace: Workspace,
        config: Config,
        geometries: Sequence[Geometry],
        sample_index: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        """Open every sample's feature array and locate each unit's row in it."""
        self.arrays = [
            np.load(workspace.features(geometry.name, config.extractor), mmap_mode="r")
            for geometry in geometries
        ]
        self.sample_index = sample_index
        self.rows = np.empty(len(sample_index), dtype=np.int64)
        for index, geometry in enumerate(geometries):
            selected = np.flatnonzero(sample_index == index)
            if len(selected):
                self.rows[selected] = geometry.feature_rows(
                    positions[selected], config.features_cover_lattice
                )
        self.n_units = len(sample_index)
        self.dim = int(self.arrays[0].shape[1])

    def batches(self, batch_size: int) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield ``(start, stop, features)`` blocks covering every test unit once."""
        for index, array in enumerate(self.arrays):
            selected = np.flatnonzero(self.sample_index == index)
            if not len(selected):
                continue
            begin, end = int(selected[0]), int(selected[-1]) + 1
            for start in range(begin, end, batch_size):
                stop = min(start + batch_size, end)
                yield start, stop, array[self.rows[start:stop]]


@dataclass
class DrawTarget:
    """One target as draw generation sees it: a weighted sum of channels, without null or scale."""

    name: str  # becomes one row of the store
    channels: list[str]  # member channel names
    # (len(channels),) weight of each channel in the sum: its upper rescaling threshold
    # for a gene target (so the sum is on the count scale), 1 otherwise.
    weights: np.ndarray


def _stamp(path: Path) -> dict[str, object]:
    """Identify one input file by name, size and modification time."""
    return {"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}


def draw_signature(
    method: str,
    settings: dict[str, object],
    extractor: str,
    bundle: Path,
    predictions: Path,
    targets: Sequence[DrawTarget],
    n_units: int,
) -> dict[str, object]:
    """Everything a stored set of draws was generated from, used to decide whether to reuse it.

    ``settings`` holds the numbers that shape a draw (draw count, dropout, seed, batch size);
    nulls are left out because they do not change the stored numbers.
    """
    signature: dict[str, object] = {"method": method}
    signature.update(settings)
    signature["n_units"] = int(n_units)
    signature["extractor"] = extractor
    signature["bundle"] = _stamp(bundle)
    signature["predictions"] = _stamp(predictions)
    signature["targets"] = {
        target.name: {
            "channels": list(target.channels),
            "weights": [float(weight) for weight in target.weights],
        }
        for target in targets
    }
    return signature


def target_columns(
    channel_index: Callable[[Sequence[str]], np.ndarray], targets: Sequence[DrawTarget]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the channel columns the targets need and each target's positions among them."""
    per_target = [channel_index(target.channels) for target in targets]
    columns = np.unique(np.concatenate(per_target))
    offsets = [np.searchsorted(columns, rows) for rows in per_target]
    return columns, offsets


def check_fits_float16(target_name: str, values: np.ndarray) -> None:
    """Raise if a target's draws exceed what a float16 store holds.

    The cast would otherwise turn the overflow into ``inf`` silently, and the store be reused.
    """
    ceiling = float(np.finfo(np.float16).max)
    largest = float(values.max()) if len(values) else 0.0
    if largest > ceiling:
        raise ValueError(
            f"target '{target_name}' draws reach {largest:g}, past the {ceiling:g} a "
            f"float16 store holds; the members of a meta-gene target sum on the count "
            f"scale, so a group this large needs splitting into several targets"
        )


def generate(
    config: Config,
    workspace: Workspace,
    geometries: Sequence[Geometry],
    modality: str,
    method: str,
    sample_index: np.ndarray,
    positions: np.ndarray,
    targets: Sequence[DrawTarget],
) -> Path:
    """Run every draw over the test area and store what each target predicted.

    Writes a ``(targets, K, units)`` float16 store of scale-1 draws on the count scale and returns
    its directory; a stored set generated from the same inputs is reused.
    """
    directory = workspace.draw_predictions(config.model, modality, method)
    bundle = DrawModels.check(config, workspace, modality, method)
    predictions = workspace.prediction_dir(config.model, modality) / "pred.npy"
    n_units = int(len(sample_index))
    # ``mc_dropout`` consumes randomness batch by batch, so the batch size changes the draws.
    settings = {
        "n_draws": int(config.sampling.n_draws),
        "dropout": config.sampling.dropout,
        "seed": int(config.sampling.seed),
        "batch_size": int(config.sampling.batch_size),
    }
    signature = draw_signature(
        method, settings, config.extractor, bundle, predictions, targets, n_units
    )
    if read_store_meta(directory).get("draw_signature") == signature:
        print(f"[draws] {modality} {method}: reusing stored draws for "
              f"{len(targets)} target(s) in {directory}")
        return directory

    from tqdm import tqdm

    models = DrawModels(config, workspace, modality, method)
    features = TestFeatures(workspace, config, geometries, sample_index, positions)

    columns, offsets = target_columns(models.channel_index, targets)

    # Rewritten whole, so no target left over from an earlier list stays readable.
    shutil.rmtree(directory, ignore_errors=True)
    store = TargetStore.create(
        directory,
        "pred",
        [target.name for target in targets],
        n_units,
        np.float16,
        stack=models.n_draws,
    )
    writer = store.writer()

    for index in tqdm(range(models.n_draws), desc=f"[draws] {method}", unit="draw"):
        block = models.draw(
            index, features.batches(config.sampling.batch_size), features.n_units, columns
        )
        for row, (target, offset) in enumerate(zip(targets, offsets)):
            values = (block[:, offset] * target.weights).sum(axis=1)
            check_fits_float16(target.name, values)
            writer[row, index] = values.astype(np.float16)
    writer.flush()

    # The signature is written last, so an interrupted run is never reused.
    meta = read_store_meta(directory)
    meta["draw_signature"] = signature
    write_json(meta, directory / "meta.json")
    print(f"[draws] {modality} {method}: {models.n_draws} draw(s) over "
          f"{len(targets)} target(s) -> {directory}")
    return directory
