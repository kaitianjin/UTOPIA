"""Walk a results tree into the axes a viewer can select over.

The index trusts the directories, not ``run_meta.json``, so a hand-copied subtree can be read:
sample, target, method and null come from the path, cluster label and scale from the keys inside
``result.npz``. ``run_meta.json`` is read only for the header shown on the page.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Arrays a ``result.npz`` can hold, keyed ``<name>_scale{s}`` or ``<name>_<label>_scale{s}``.
#: Names hold underscores themselves, so a key is matched against this list, not split at one.
RESULT_ARRAYS = (
    "pvals",
    "pvals_adj",
    "confidence",
    "level",
    "std",
    "lower",
    "upper",
    "prediction",
    "fdr",
    "fdp",
    "n_discoveries",
)
#: Every key ends in the scale it was computed at.
SCALE_PATTERN = re.compile(r"^(?P<head>.+)_scale(?P<scale>\d+)$")
#: A null directory is named after the value it tested against.
NULL_PATTERN = re.compile(r"^null_(?P<value>.+)$")
#: Folder ``utopia view`` writes its pages to by default, under the workspace beside ``results/``.
OUTPUT_DIR_NAME = "visualization"
#: Name of the directory a workspace keeps its model trees in.
RESULTS_DIR_NAME = "results"
#: Directories in a results tree that are neither models nor test samples: the pooled
#: diagnostics, and the viewer's output folder of a tree copied out of its workspace.
RESERVED = ("_pooled", OUTPUT_DIR_NAME)


@dataclass
class NullIndex:
    """One null value under one method, and the arrays stored against it."""

    value: str  # the null as it appears in the ``null_<value>`` directory name
    directory: Path
    # Array name -> the ``(label, scale)`` pairs it exists for; ``label`` is None for a method
    # whose answer does not use the clustering.
    keys: dict[str, list[tuple]] = field(default_factory=dict)

    @property
    def labels(self) -> list[str | None]:
        """Cluster labels this null's arrays are keyed by, in numeric order."""
        found = {label for pairs in self.keys.values() for label, _ in pairs}
        return _sorted_labels(found)

    @property
    def scales(self) -> list[int]:
        """Scales this null's arrays exist at."""
        return sorted({scale for pairs in self.keys.values() for _, scale in pairs})


@dataclass
class MethodIndex:
    """One uncertainty method's results for one target of one sample."""

    name: str
    nulls: dict[str, NullIndex] = field(default_factory=dict)  # null value -> its arrays


@dataclass
class TargetIndex:
    """One target's results for one sample: its own arrays, and its methods."""

    name: str
    directory: Path
    modality: str | None = None  # which ``units_<modality>_scale{s}.npz`` its arrays match
    scales: list[int] = field(default_factory=list)  # scales with a prediction file
    truth_scales: list[int] = field(default_factory=list)  # scales with a truth file
    calib_scales: list[int] = field(default_factory=list)  # scales with a calib_truth file
    methods: dict[str, MethodIndex] = field(default_factory=dict)


@dataclass
class SampleIndex:
    """One test sample's self-contained results subtree."""

    name: str
    # ``(modality, scale)`` -> that ``units_<modality>_scale{s}.npz``.
    units: dict[tuple, Path] = field(default_factory=dict)
    targets: dict[str, TargetIndex] = field(default_factory=dict)


@dataclass
class ModelIndex:
    """One model's whole results tree."""

    name: str
    samples: dict[str, SampleIndex] = field(default_factory=dict)
    # ``run_meta.json``, or an empty dict when the subtree was copied without it.
    header: dict[str, object] = field(default_factory=dict)


def parse_key(key: str) -> tuple | None:
    """Split a ``result.npz`` key into ``(name, cluster label, scale)``.

    The label is None for an unclustered key; the result is None for a key not in
    :data:`RESULT_ARRAYS`.
    """
    match = SCALE_PATTERN.match(key)
    if match is None:
        return None
    head, scale = match["head"], int(match["scale"])
    if head in RESULT_ARRAYS:
        return head, None, scale
    # Longest name first, so ``pvals_adj_k15`` is ``pvals_adj`` of ``k15``. The rest is the label,
    # which may hold underscores: an external run's label is the user's cluster file name.
    for name in sorted(RESULT_ARRAYS, key=len, reverse=True):
        prefix = name + "_"
        if head.startswith(prefix) and len(head) > len(prefix):
            return name, head[len(prefix):], scale
    return None


def _sorted_labels(labels: Iterable[str | None]) -> list[str | None]:
    """Order cluster labels by their trailing number when they carry one, None first."""

    def key(label: str | None) -> tuple:
        """Sort key of one label: None, then by trailing number, then by name."""
        if label is None:
            return (0, 0, "")
        digits = re.search(r"(\d+)$", label)
        return (1, int(digits.group(1)) if digits else 0, label)

    return sorted(labels, key=key)


def sorted_nulls(names: Iterable[str]) -> list[str]:
    """Order null names smallest first; names that are not numbers sort last, alphabetically."""

    def key(name: str) -> tuple:
        """Sort key of one null name: numbers by value, then every other name."""
        try:
            return (0, float(name), name)
        except ValueError:
            return (1, 0.0, name)

    return sorted(names, key=key)


def results_root(path: Path) -> Path:
    """Resolve what the user typed to the directory to index.

    A workspace resolves to its ``results/``; a results directory or a single model directory is
    returned as it is.
    """
    path = Path(path).resolve()
    if (path / RESULTS_DIR_NAME).is_dir():
        return path / RESULTS_DIR_NAME
    return path


def output_root(path: Path) -> Path:
    """Default output directory of the pages: ``<workspace>/visualization``, beside ``results/``.

    A tree that was copied out of its workspace has no workspace to write into, so its pages go
    beside its model trees.
    """
    root = results_root(Path(path))
    if is_model_dir(root):
        results_directory = root.parent
    else:
        results_directory = root
    if results_directory.name == RESULTS_DIR_NAME:
        workspace_directory = results_directory.parent
        return workspace_directory / OUTPUT_DIR_NAME
    return results_directory / OUTPUT_DIR_NAME


def is_model_dir(path: Path) -> bool:
    """Whether ``path`` is one model's tree rather than a tree of models."""
    if (path / "run_meta.json").exists():
        return True
    return any(
        child.is_dir()
        and child.name not in RESERVED
        and any(grandchild.is_dir() for grandchild in child.iterdir())
        and list(child.glob("units_*_scale*.npz"))
        for child in path.iterdir()
        if child.is_dir()
    )


def index_tree(path: Path) -> dict[str, ModelIndex]:
    """Index every model under a workspace, a ``results`` directory or one model directory.

    Returns model name -> :class:`ModelIndex`; models and samples without results are left out.
    """
    from ..io import read_json

    root = results_root(Path(path))
    if is_model_dir(root):
        directories = [root]
    else:
        directories = [
            child
            for child in sorted(root.iterdir())
            if child.is_dir() and child.name not in RESERVED
        ]

    indexed: dict[str, ModelIndex] = {}
    for directory in directories:
        header_path = directory / "run_meta.json"
        model = ModelIndex(
            name=directory.name,
            header=read_json(header_path) if header_path.exists() else {},
        )
        for sample_dir in sorted(directory.iterdir()):
            if not sample_dir.is_dir() or sample_dir.name in RESERVED:
                continue
            sample = _index_sample(sample_dir, model.header)
            if sample.targets:
                model.samples[sample.name] = sample
        if model.samples:
            indexed[model.name] = model
    return indexed


def _index_sample(directory: Path, header: dict[str, object]) -> SampleIndex:
    """Index one sample subtree: its unit files and every target below it."""
    sample = SampleIndex(name=directory.name)
    for path in sorted(directory.glob("units_*_scale*.npz")):
        match = re.match(r"^units_(?P<modality>.+)_scale(?P<scale>\d+)$", path.stem)
        if match:
            sample.units[(match["modality"], int(match["scale"]))] = path
    for target_dir in sorted(directory.iterdir()):
        if not target_dir.is_dir():
            continue
        target = _index_target(target_dir)
        if target.methods:
            sample.targets[target.name] = target
    _assign_modalities(sample, header)
    return sample


def _index_target(directory: Path) -> TargetIndex:
    """Index one target: the arrays it owns, and the methods applied to it."""
    target = TargetIndex(name=directory.name, directory=directory)
    for prefix, field_name in (
        ("prediction", "scales"),
        ("truth", "truth_scales"),
        ("calib_truth", "calib_scales"),
    ):
        scales = []
        for path in directory.glob(f"{prefix}_scale*.npy"):
            digits = re.match(rf"^{prefix}_scale(\d+)$", path.stem)
            if digits:
                scales.append(int(digits.group(1)))
        setattr(target, field_name, sorted(scales))

    for method_dir in sorted(directory.iterdir()):
        if not method_dir.is_dir():
            continue
        method = MethodIndex(name=method_dir.name)
        for null_dir in sorted(method_dir.iterdir()):
            match = NULL_PATTERN.match(null_dir.name) if null_dir.is_dir() else None
            if match is None or not (null_dir / "result.npz").exists():
                continue
            method.nulls[match["value"]] = _index_null(match["value"], null_dir)
        if method.nulls:
            target.methods[method.name] = method
    return target


def _index_null(value: str, directory: Path) -> NullIndex:
    """Read the key names of one ``result.npz`` without reading its arrays."""
    null = NullIndex(value=value, directory=directory)
    with np.load(directory / "result.npz") as data:
        names = list(data.files)
    for key in names:
        parsed = parse_key(key)
        if parsed is None:
            continue
        name, label, scale = parsed
        null.keys.setdefault(name, []).append((label, scale))
    return null


def _assign_modalities(sample: SampleIndex, header: dict[str, object]) -> None:
    """Set each target's modality, which decides the unit file its arrays are drawn against.

    It comes from ``run_meta.json`` when the tree has one, and otherwise from the unit file whose
    length matches the target's prediction.
    """
    declared = {
        str(name): str(entry.get("modality"))
        for name, entry in (header.get("targets") or {}).items()
        if isinstance(entry, dict)
    }
    lengths: dict[int, dict[str, int]] = {}
    for (modality, scale), path in sample.units.items():
        with np.load(path) as data:
            key = "positions" if "positions" in data.files else "evaluable"
            lengths.setdefault(scale, {})[modality] = int(len(data[key]))

    for target in sample.targets.values():
        for scale in target.scales:
            available = lengths.get(scale, {})
            named = declared.get(target.name)
            if named in available:
                target.modality = named
                break
            size = len(np.load(target.directory / f"prediction_scale{scale}.npy", mmap_mode="r"))
            matches = [name for name, count in available.items() if count == size]
            if matches:
                target.modality = matches[0]
                break


def axes(models: dict[str, ModelIndex]) -> dict[str, list]:
    """Collect the distinct values of every selectable axis in a tree."""
    samples, targets, methods, nulls, labels, scales = set(), set(), set(), set(), set(), set()
    for model in models.values():
        for sample in model.samples.values():
            samples.add(sample.name)
            for target in sample.targets.values():
                targets.add(target.name)
                scales.update(target.scales)
                for method in target.methods.values():
                    methods.add(method.name)
                    for null in method.nulls.values():
                        nulls.add(null.value)
                        labels.update(null.labels)
                        scales.update(null.scales)
    return {
        "models": sorted(models),
        "samples": sorted(samples),
        "targets": sorted(targets),
        "methods": sorted(methods),
        "nulls": sorted_nulls(nulls),
        "clusters": [label for label in _sorted_labels(labels) if label is not None],
        "scales": sorted(scales),
    }
