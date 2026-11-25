"""Training the baselines' bundles.

A bundle is one ``.pt`` file holding every trained member of one baseline at one scale, with the
scale, channels and rescaling thresholds recorded beside the weights.  ``raw``, ``mc_dropout``,
``resampling`` and the cell-type ensemble train through the pipeline's own
:func:`utopia.calibration.run.fit`; the two gene ensembles use the trainers in ``models.py``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..calibration.draws import _member_rows, _weights
from ..calibration.run import TrainingInputs, fit
from ..config import Config
from ..io import open_memmap
from .config import COUNT_SCALE_METHODS, BaselineConfig
from .models import train_heteroscedastic, train_negative_binomial


def member_seeds(config: Config, baselines: BaselineConfig, method: str) -> list[int]:
    """The training seed of every member of one baseline's bundle.

    ``raw`` and ``mc_dropout`` are single models that keep the pipeline's seed; the other bundles
    train member ``m`` at ``baselines.seed + m``.
    """
    if method in ("raw", "mc_dropout"):
        return [int(config.default.seed)]
    return [int(baselines.seed) + member for member in range(baselines.members(method))]


#: Trainer of each variance-bearing gene ensemble, keyed by the distribution its
#: members state; every other bundle trains through the pipeline's own ``fit``.
_TRAINERS = {
    "gaussian": train_heteroscedastic,
    "negative_binomial": train_negative_binomial,
}


def _distribution(modality: str, method: str) -> str | None:
    """The predictive distribution one member states, or None when it states none.

    ``"gaussian"`` for the gene ``deep_ensemble`` and ``"negative_binomial"`` for
    ``deep_ensemble_nb``; every other member states a point prediction or a class probability.
    """
    if modality != "gene":
        return None
    if method == "deep_ensemble":
        return "gaussian"
    if method == "deep_ensemble_nb":
        return "negative_binomial"
    return None


def _is_heteroscedastic(modality: str, method: str) -> bool:
    """Whether this member emits a per-unit variance beside its mean (the gene ensembles do)."""
    return _distribution(modality, method) is not None


def _fit_member(
    modality: str,
    method: str,
    features: Any,
    targets: Any,
    rows: np.ndarray,
    hyper: dict[str, object],
) -> torch.nn.Module:
    """Train one member, through the pipeline's trainer or a likelihood's own."""
    distribution = _distribution(modality, method)
    if distribution is None:
        return fit(modality, features, targets, rows, hyper)
    return _TRAINERS[distribution](features, targets, rows, **hyper)


def _member_worker(
    feature_path: str,
    target_path: str,
    rows: np.ndarray,
    modality: str,
    method: str,
    hyper: dict[str, object],
    member: int,
    queue: object,
) -> None:
    """Train one member in its own process and put ``(member, weights)`` on ``queue``.

    Inputs arrive as paths so the processes share one page cache, and the weights leave as numpy
    arrays rather than tensors.
    """
    model = _fit_member(
        modality,
        method,
        open_memmap(feature_path, mode="r"),
        open_memmap(target_path, mode="r"),
        rows,
        hyper,
    )
    queue.put((member, _weights(model)))


def _hyper_for(
    baselines: BaselineConfig, inputs: TrainingInputs, modality: str, method: str, seed: int
) -> dict[str, object]:
    """The hyper-parameters one member trains under: the pipeline's own, with this member's seed.

    The gene ensembles also get ``warmup_epochs`` and ``nll_beta`` for their likelihood trainers.
    """
    hyper = {**inputs.hyper, "seed": int(seed)}
    if _is_heteroscedastic(modality, method):
        hyper["warmup_epochs"] = int(baselines.warmup_epochs)
        hyper["nll_beta"] = float(baselines.nll_beta)
    return hyper


def train_bundle(
    config: Config,
    baselines: BaselineConfig,
    modality: str,
    method: str,
    scale: int,
    inputs: TrainingInputs,
    path: Path,
) -> float:
    """Train every member of one baseline and save the bundle to ``path``.

    ``inputs`` is the training set at ``scale``; returns the wall-clock seconds the bundle took.
    """
    from tqdm import tqdm

    seeds = member_seeds(config, baselines, method)
    jobs: list[tuple[int, int, np.ndarray]] = [
        (member, seed, _member_rows(method, len(inputs.table), seed))
        for member, seed in enumerate(seeds)
    ]
    states: dict[int, dict[str, np.ndarray]] = {}
    started = time.time()
    progress = tqdm(total=len(jobs), desc=f"[baselines] {modality} {method}", unit="model")
    processes = baselines.processes(config)

    if processes <= 1 or len(jobs) == 1:
        for member, seed, rows in jobs:
            model = _fit_member(
                modality,
                method,
                inputs.features,
                inputs.targets,
                rows,
                _hyper_for(baselines, inputs, modality, method, seed),
            )
            states[member] = _weights(model)
            progress.update(1)
    else:
        import torch.multiprocessing as multiprocessing

        multiprocessing.set_start_method("spawn", force=True)
        feature_path, target_path = inputs.paths
        for start in range(0, len(jobs), processes):
            queue = multiprocessing.Queue()
            running = []
            for member, seed, rows in jobs[start : start + processes]:
                process = multiprocessing.Process(
                    target=_member_worker,
                    args=(
                        feature_path,
                        target_path,
                        rows,
                        modality,
                        method,
                        _hyper_for(baselines, inputs, modality, method, seed),
                        member,
                        queue,
                    ),
                )
                process.start()
                running.append(process)
            # Drain before joining: a worker blocks in put() until its result is read.
            for _ in running:
                member, state = queue.get()
                states[member] = state
                progress.update(1)
            for process in running:
                process.join()
    progress.close()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": method,
            "modality": modality,
            "scale": int(scale),
            # Recorded because a bundle trained at another batch size is a differently
            # fitted model (see ``needs_training``).
            "batch_size": int(inputs.hyper["batch_size"]),
            "epochs": int(inputs.hyper["epochs"]),
            # The two gene ensembles write stores of the same shape that mean different things.
            "distribution": _distribution(modality, method),
            # Whether the target was the raw count rather than its [0, 1] rescaling.
            "count_scale": method in COUNT_SCALE_METHODS,
            # Puts a member's [0, 1] output back on the count scale; the thresholds differ
            # by scale, so they travel with the weights (None for a count-scale bundle).
            "thresholds": (
                None
                if inputs.thresholds is None
                else {
                    name: [float(low), float(high)]
                    for name, (low, high) in zip(inputs.channels, inputs.thresholds)
                }
            ),
            "heteroscedastic": _is_heteroscedastic(modality, method),
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
    elapsed = time.time() - started
    print(f"[baselines]   {len(jobs)} member(s) -> {path} in {elapsed:.1f}s")
    return elapsed


def bundle_meta(path: Path) -> dict[str, object]:
    """Everything but the weights of a stored bundle: scale, channels, thresholds and the rest."""
    stored = torch.load(path, map_location="cpu")
    return {key: value for key, value in stored.items() if key != "state_dicts"}


def needs_training(
    path: Path,
    channels: Sequence[str],
    n_members: int,
    scale: int,
    batch_size: int,
    distribution: str | None = None,
) -> bool:
    """Whether a bundle must be trained: it is absent, or does not match what the run asks for.

    The stored distribution, scale, batch size, channel list and member count are compared in
    that order, and the first mismatch is printed.
    """
    if not path.exists():
        return True
    stored = torch.load(path, map_location="cpu")
    if stored.get("distribution", None) != distribution:
        print(
            f"[baselines]   {path.name} states "
            f"{stored.get('distribution', 'unrecorded') or 'no distribution'}, this bundle states "
            f"{distribution or 'no distribution'}; retraining"
        )
        return True
    if int(stored.get("scale", -1)) != int(scale):
        print(
            f"[baselines]   {path.name} was trained at scale {stored.get('scale', 'unrecorded')}, "
            f"this bundle serves scale {scale}; retraining"
        )
        return True
    if int(stored.get("batch_size", -1)) != int(batch_size):
        print(
            f"[baselines]   {path.name} was trained at batch size "
            f"{stored.get('batch_size', 'unrecorded')}, this scale trains at {batch_size}; "
            "retraining"
        )
        return True
    if list(stored["channels"]) != list(channels):
        print(
            f"[baselines]   {path.name} was trained for {len(stored['channels'])} channel(s), the "
            f"store now holds {len(channels)}; retraining"
        )
        return True
    if len(stored["state_dicts"]) != n_members:
        print(
            f"[baselines]   {path.name} holds {len(stored['state_dicts'])} member(s), the config "
            f"asks for {n_members}; retraining"
        )
        return True
    return False
