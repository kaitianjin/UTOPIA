"""Stage 6: turn stored predictions into p-values, confidence scores and maps.

For each target the stage reads the target's rows from the gene-major ``(channels, units)`` stores,
sums them into one value per unit, aggregates to each configured scale, computes p-values and
confidence with each configured method, and writes the results and maps. Targets are processed
one at a time, so peak memory is one target's arrays.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .aggregate import (
    coarse_shape,
    decode_lattice_keys,
    group_index,
    group_sum,
    lattice_keys,
)
from .clustering import BACKGROUND
from .config import CLUSTER_METHODS, MODALITIES, SAMPLING_METHODS, Config
from .coordinates import load_positions
from .geometry import Geometry, positions_to_rows
from .io import (
    TargetStore,
    load_mask,
    read_json,
    read_store_meta,
    read_store_units,
    read_yaml_as_text,
    save_image,
    store_clusterings,
    write_json,
)
from .uncertainty import (
    DrawAccumulator,
    UncertaintyResult,
    bh_adjust_grouped,
    conformal_cluster_pvals,
    conformal_marginal_levels,
    fdp_curve,
    pvals_to_confidence,
)
from .workspace import Workspace

#: Quantile of the calibration truth used to normalise a target's scale.
NORM_QUANTILE = 99.99
#: Quantile of the calibration truth used for the display colour scale alone.
DISPLAY_QUANTILE = 99.9
#: Percentiles of one slide's own values that bound the colour scale of an external model's maps;
#: its values can sit in a narrow band far from zero, which a scale from zero would paint flat.
SLIDE_RANGE_PERCENTILES = (0.1, 99.9)


@dataclass
class Target:
    """One inference target read from ``targets.yaml``."""

    name: str  # also the name of the target's results directory
    modality: str  # "gene" or "cell_type"
    channels: list[str]  # member gene or cell-type names
    null: float  # null value the target is tested against
    null_name: str = ""  # name of the null's results directory; defaults to the null value

    def __post_init__(self) -> None:
        """Name the null's results directory after the null value when no name was given."""
        if not self.null_name:
            self.null_name = f"{self.null:g}"


def _read_targets_file(path: Path) -> dict[str, object]:
    """Return the content of the YAML target list at ``path`` as a dict, refusing a JSON file."""
    path = Path(path)
    # The YAML reader would parse a JSON file without complaint, so it is refused by name.
    if path.suffix.lower() == ".json":
        raise ValueError(
            f"the target list {path} is a JSON file; the target list is YAML now, so write it "
            "as targets.yaml (see docs/DATA_STRUCTURE.md, section 1.4) and name that file "
            "under 'targets' in the config"
        )
    payload = read_yaml_as_text(path)

    # An empty file lists no targets.
    if payload is None or payload == "":
        return {}
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path} must start with the keys 'gene_groups' and / or 'ct_groups', but it holds "
            f"a {type(payload).__name__}"
        )
    return payload


def _member_names(entry: dict[str, object], key: str, path: Path) -> list[str]:
    """Return the gene or cell-type names listed under ``key`` of one target entry."""
    if not isinstance(entry, dict) or key not in entry:
        raise ValueError(f"every target in {path} needs a '{key}' list, but found: {entry}")
    members = entry[key]
    # A name written without the list brackets would otherwise be split into its letters.
    if not isinstance(members, list) or len(members) == 0:
        raise ValueError(
            f"'{key}' must be a list of names in {path}, for example '{key}: [NAME]', "
            f"but found: {members}"
        )
    for member in members:
        # A name holding a colon or a bracket is read by YAML as a mapping or a list.
        if isinstance(member, (dict, list)):
            raise ValueError(
                f"a name under '{key}' in {path} was not read as plain text: {member}; "
                "put the name in quotes"
            )
    return [str(member) for member in members]


def _null_of(group: dict[str, object], section: str, path: Path) -> tuple[float, str]:
    """Return the null value of one null-value group and the name it gives its results directory.

    The name is empty when the group states none; :class:`Target` then names it after the value.
    """
    if not isinstance(group, dict) or "null_number" not in group or "groups" not in group:
        raise ValueError(
            f"every item of '{section}' in {path} needs a 'null_number' and a 'groups' list, "
            f"but found: {group}"
        )
    try:
        null = float(group["null_number"])
    except (TypeError, ValueError):
        raise ValueError(
            f"'null_number' must be a number in {path}, but found: {group['null_number']}"
        ) from None

    null_name = group.get("null_name")
    if null_name is None:
        null_name = ""
    return null, str(null_name)


def load_targets(path: Path) -> list[Target]:
    """Read the gene and cell-type target groups from ``targets.yaml`` as a flat list of targets.

    Each null-value group lists the targets tested against that null; a target with several
    members must carry a ``name``.
    """
    payload = _read_targets_file(path)
    targets: list[Target] = []
    for section, modality, key in (
        ("gene_groups", "gene", "genes"),
        ("ct_groups", "cell_type", "cell_types"),
    ):
        # A section that is left out, or written with nothing under it, holds no targets.
        null_groups = payload.get(section) or []
        for group in null_groups:
            null, null_name = _null_of(group, section, path)
            for entry in group["groups"] or []:
                members = _member_names(entry, key, path)
                name = entry.get("name")
                if name is None or name == "":
                    if len(members) > 1:
                        raise ValueError(f"target group {members} needs a 'name' in {path}")
                    # A single gene or cell type is named after itself.
                    name = members[0]
                targets.append(
                    Target(
                        name=str(name),
                        modality=modality,
                        channels=members,
                        null=null,
                        null_name=null_name,
                    )
                )
    seen = [(target.name, target.null_name) for target in targets]
    clashes = sorted({pair for pair in seen if seen.count(pair) > 1})
    if clashes:
        raise ValueError(
            f"target(s) {clashes} appear twice under one null name in {path}; each would "
            "overwrite the other's results directory"
        )
    # A name may appear under several nulls (a sweep) but must have the same members each time:
    # the results directory and the draw-store row are keyed by name alone.
    defined: dict[str, Target] = {}
    for target in targets:
        first = defined.setdefault(target.name, target)
        if (first.modality, first.channels) != (target.modality, target.channels):
            raise ValueError(
                f"target '{target.name}' is defined twice in {path} with different members "
                f"({first.modality}:{first.channels} and {target.modality}:{target.channels}); "
                "one name must mean one thing, since the results directory is named after it"
            )
    return targets


@dataclass
class Store:
    """One calibration or prediction store: its per-unit arrays, clusterings and metadata.

    Every array besides the predictions is optional.
    """

    directory: Path
    positions: np.ndarray | None  # (N, 2) position of every unit; None means no maps
    sample_index: np.ndarray  # (N,) sample id of every unit; zeros when not supplied
    # (N,) whether the unit carries ground truth. The built-in model writes all True for
    # genes and the annotation mask for cell types; True everywhere when not supplied.
    evaluable: np.ndarray
    radius: np.ndarray | None  # drawing radius of the units; None for single-cell units
    clusterings: dict[str, np.ndarray]  # cluster label -> (N,) labels, from ``clusters_*.npy``
    meta: dict[str, object]  # the store's ``meta.json``
    n_units: int

    @classmethod
    def load(cls, directory: Path) -> "Store":
        """Read a store directory: its unit arrays, clusterings and metadata."""
        meta = read_store_meta(directory)
        arrays = read_store_units(directory)
        n_units = int(TargetStore.open(directory, "pred").n_units)
        clusterings = {
            label: np.load(path) for label, path in store_clusterings(directory).items()
        }
        return cls(
            directory=directory,
            positions=arrays.get("positions"),
            sample_index=arrays.get(
                "sample_index", np.zeros(n_units, dtype=np.int16)
            ).astype(np.int64),
            evaluable=arrays.get("evaluable", np.ones(n_units, dtype=bool)),
            radius=arrays.get("radius"),
            clusterings=clusterings,
            meta=meta,
            n_units=n_units,
        )

    @property
    def sample_names(self) -> list[str]:
        """Sample names in index order, defaulted when ``meta.json`` names none."""
        names = self.meta.get("samples")
        if names:
            return [str(name) for name in names]
        return [f"sample{index}" for index in range(int(self.sample_index.max()) + 1)]

    def has(self, name: str) -> bool:
        """Whether the store holds an array of the given base name."""
        return (self.directory / f"{name}.npy").exists()


def open_stores(workspace: Workspace, model: str, modality: str) -> dict[str, object]:
    """Open one modality's calibration and prediction stores, plus the gene truth stores.

    The test-side gene truth is opened only when the prediction store holds it.
    """
    calib_dir = workspace.calibration_dir(model, modality)
    predict_dir = workspace.prediction_dir(model, modality)
    calib, test = Store.load(calib_dir), Store.load(predict_dir)
    stores: dict[str, object] = {
        "calib": calib,
        "test": test,
        "calib_pred": TargetStore.open(calib_dir, "pred"),
        "test_pred": TargetStore.open(predict_dir, "pred"),
    }
    if modality == "gene":
        stores["calib_truth"] = TargetStore.open(calib_dir, "truth")
        if test.has("truth"):
            stores["test_truth"] = TargetStore.open(predict_dir, "truth")
    return stores


def external_modalities(workspace: Workspace, model: str) -> list[str]:
    """Modalities for which an external model wrote a calibration store."""
    return [
        modality
        for modality in MODALITIES
        if (workspace.calibration_dir(model, modality) / "pred.npy").exists()
    ]


def check_stores(config: Config) -> None:
    """Validate an external model's calibration and prediction stores, raising on any problem.

    It cannot check that the calibration predictions were held out; that is up to their author.
    """
    workspace = Workspace(config)
    modalities = external_modalities(workspace, config.model)
    if not modalities:
        raise ValueError(
            f"no store under {workspace.calibration_dir(config.model, 'gene').parent}; "
            f"write one with export_scripts/utopia_store.py (docs/DATA_STRUCTURE.md, section 1.2)"
        )

    problems: list[str] = []
    for modality in modalities:
        calib_dir = workspace.calibration_dir(config.model, modality)
        predict_dir = workspace.prediction_dir(config.model, modality)
        if not (predict_dir / "pred.npy").exists():
            problems.append(
                f"{modality}: a calibration store exists but {predict_dir}/pred.npy does not"
            )
            continue
        calib, test = Store.load(calib_dir), Store.load(predict_dir)
        calib_pred = TargetStore.open(calib_dir, "pred")
        test_pred = TargetStore.open(predict_dir, "pred")

        if calib_pred.channels != test_pred.channels:
            problems.append(f"{modality}: the two stores name different channels")
        truth_name = "truth_labels" if modality == "cell_type" else "truth"
        if not calib.has(truth_name):
            problems.append(f"{modality}: {calib_dir}/{truth_name}.npy is required to calibrate")
        for store, label in ((calib, "calibration"), (test, "prediction")):
            problems.extend(_store_problems(store, label, modality))
        missing = set(calib.clusterings) ^ set(test.clusterings)
        if missing:
            problems.append(
                f"{modality}: clustering(s) {sorted(missing)} appear in one store only; "
                "UTOPIA needs the same labels on both sides"
            )

    if problems:
        raise ValueError("store validation failed:\n  " + "\n  ".join(problems))
    print(f"[check] external model '{config.model}': {', '.join(modalities)} store(s) validated")
    print(f"[check] method(s): {', '.join(config.methods)}; BH scope {config.inference.bh_scope}")


def _store_problems(store: Store, label: str, modality: str) -> list[str]:
    """List the problems found in one store's predictions, optional arrays and metadata."""
    problems: list[str] = []
    values = np.load(store.directory / "pred.npy", mmap_mode="r")
    if values.ndim != 2:
        return [f"{modality} {label}: pred.npy must be (channels, units), got {values.shape}"]
    # Checking about 10,000 evenly spaced units is enough to catch a broken store.
    step = max(1, values.shape[1] // 10000)
    sampled = np.asarray(values[:, ::step], np.float32)
    if not np.isfinite(sampled).all():
        problems.append(f"{modality} {label}: pred.npy holds NaN or inf, which cannot be ranked")
    for name, array in (
        ("positions", store.positions),
        ("evaluable", store.evaluable),
        ("radius", store.radius),
        *[(f"clusters_{key}", found) for key, found in store.clusterings.items()],
    ):
        if array is not None and np.ndim(array) and len(array) != store.n_units:
            problems.append(
                f"{modality} {label}: {name}.npy has {len(array)} entries but pred.npy has "
                f"{store.n_units} units"
            )
    names = store.meta.get("samples")
    canvas = store.meta.get("canvas_shape")
    if canvas is not None and names is not None and len(canvas) != len(names):
        problems.append(
            f"{modality} {label}: canvas_shape lists {len(canvas)} shapes for {len(names)} samples"
        )
    if (
        names is not None
        and len(store.sample_index)
        and int(store.sample_index.max()) >= len(names)
    ):
        problems.append(
            f"{modality} {label}: sample_index reaches past the {len(names)} named samples"
        )
    if canvas is not None and store.positions is not None:
        for index, shape in enumerate(canvas):
            rows = store.sample_index == index
            if rows.any() and (store.positions[rows] >= np.array(shape)).any():
                problems.append(
                    f"{modality} {label}: sample {index} has positions outside its canvas"
                )
    return problems


@dataclass
class ScaleIndex:
    """Grouping of a store's units into the coarse units reported at one scale.

    At scale 1, and always for an external store, the grouping is the identity.
    """

    inverse: np.ndarray  # (N,) coarse-unit id per store unit
    n_groups: int  # number of coarse units
    counts: np.ndarray  # (n_groups,) number of store units in each coarse unit
    sample_index: np.ndarray  # (n_groups,) sample id of each coarse unit
    positions: np.ndarray | None  # (n_groups, 2) position of each coarse unit, or None
    rows: np.ndarray  # (n_groups,) store row of each coarse unit at scale 1, else -1
    keys: np.ndarray | None = None  # (n_groups,) sorted lattice keys above scale 1, else None


def build_scale_index(store: Store, scale: int, span: int) -> ScaleIndex:
    """Group a store's scale-1 units into the ``scale`` x ``scale`` blocks that contain them.

    ``span`` is a bound on both lattice dimensions, used to build one integer key per block.
    """
    if scale == 1:
        return identity_index(store)
    keys = lattice_keys(store.sample_index, store.positions, scale, span)
    unique, inverse, counts = group_index(keys)
    sample_index, positions = decode_lattice_keys(unique, scale, span)
    rows = np.full(len(unique), -1, dtype=np.int64)
    return ScaleIndex(
        inverse, len(unique), counts, sample_index.astype(np.int64), positions, rows, unique
    )


def identity_index(store: Store) -> ScaleIndex:
    """Report one result per store unit, in store row order."""
    n_units = store.n_units
    rows = np.arange(n_units, dtype=np.int64)
    return ScaleIndex(
        inverse=rows,
        n_groups=n_units,
        counts=np.ones(n_units, dtype=np.int64),
        sample_index=store.sample_index,
        positions=store.positions,
        rows=rows,
    )


def _aggregate(values: np.ndarray, index: ScaleIndex) -> np.ndarray:
    """Sum per-unit values within each coarse unit."""
    if index.n_groups == len(values):
        return np.asarray(values, dtype=np.float64)
    return group_sum(values, index.inverse, index.n_groups)


@dataclass
class OffTissue:
    """One role's measured superpixels outside the tissue mask, and the coarse unit each joins.

    The stores hold tissue superpixels only; these supply the rest of a coarse unit's gene count.
    """

    # Per sample: its ``expression.npy`` memory map, gene name -> column, and the input rows of
    # its kept superpixels.
    sources: list[tuple[np.ndarray, dict[str, int], np.ndarray]]
    # Scale -> coarse unit each kept superpixel joins, -1 where it joins none.
    inverse: dict[int, np.ndarray]

    def counts(self, channels: Sequence[str]) -> np.ndarray:
        """Summed raw count of ``channels`` at every kept superpixel."""
        per_sample = []
        for expression, columns, rows in self.sources:
            channel_columns = [columns[name] for name in channels]
            block = np.asarray(expression[np.ix_(rows, channel_columns)], dtype=np.float64)
            per_sample.append(block.sum(axis=1))
        return np.concatenate(per_sample)

    def add(self, block_values: np.ndarray, counts: np.ndarray, scale: int) -> np.ndarray:
        """Add the superpixel ``counts`` to one scale's coarse-unit values; scale 1 has none."""
        inverse = self.inverse.get(scale)
        if inverse is None:
            return block_values
        hit = inverse >= 0
        added = np.bincount(inverse[hit], weights=counts[hit], minlength=len(block_values))
        return block_values + added


def _off_tissue(
    config: Config,
    geometries: Sequence[Geometry],
    role: str,
    indexes: dict[int, ScaleIndex],
    span: int,
) -> OffTissue | None:
    """Collect one role's measured non-tissue superpixels that fall in a block with tissue.

    Returns None when there is nothing to add: only scale 1, or a sample without ``genes/`` input.
    """
    coarse = {}
    for scale, index in indexes.items():
        if index.keys is not None and len(index.keys):
            coarse[scale] = index
    directories = [config.sample_dir(geometry.name) / "genes" for geometry in geometries]
    every_sample_has_counts = all(
        (directory / "expression.npy").exists() for directory in directories
    )
    if not coarse or not geometries or not every_sample_has_counts:
        return None

    # Only a superpixel listed in the input's own positions can have a count to add; an unlisted
    # one holds zero counts.
    lookups = [
        positions_to_rows(
            load_positions(directory, config, geometry.name), geometry.lattice_shape
        )
        for geometry, directory in zip(geometries, directories)
    ]
    found = [
        np.argwhere(geometry.role_region(role) & ~geometry.tissue_mask & (lookup >= 0))
        for geometry, lookup in zip(geometries, lookups)
    ]
    sample_index = np.concatenate(
        [np.full(len(units), sample, dtype=np.int64) for sample, units in enumerate(found)]
    )
    positions = np.concatenate(found)
    inverse: dict[int, np.ndarray] = {}
    kept = np.zeros(len(positions), dtype=bool)
    for scale, index in coarse.items():
        keys = lattice_keys(sample_index, positions, scale, span)
        slot = np.minimum(np.searchsorted(index.keys, keys), len(index.keys) - 1)
        hit = index.keys[slot] == keys
        inverse[scale] = np.where(hit, slot, -1)
        kept |= hit

    sources = []
    for sample, (directory, lookup) in enumerate(zip(directories, lookups)):
        mine = positions[kept & (sample_index == sample)]
        names = read_json(directory / "gene_names.json")
        sources.append((
            np.load(directory / "expression.npy", mmap_mode="r"),
            {name: column for column, name in enumerate(names)},
            lookup[mine[:, 0], mine[:, 1]].astype(np.int64),
        ))
    return OffTissue(sources, {scale: values[kept] for scale, values in inverse.items()})


def add_off_tissue(
    stores: dict[str, object],
    config: Config,
    geometries: Sequence[Geometry],
    grids: dict[int, ScaleGrid],
    span: int,
) -> dict[str, int]:
    """Store each role's :class:`OffTissue` in ``stores`` under ``<role>_off_tissue``.

    Returns role -> the number of off-tissue superpixels that join a coarse unit.
    """
    added: dict[str, int] = {}
    roles = (("calib", "calib_truth"), ("test", "test_truth"))
    for role, truth in roles:
        if truth not in stores:
            continue
        indexes = {scale: getattr(grid, role) for scale, grid in grids.items()}
        found = _off_tissue(config, geometries, role, indexes, span)
        if found is not None:
            stores[f"{role}_off_tissue"] = found
            added[role] = int(sum(len(rows) for _, _, rows in found.sources))
    return added


@dataclass
class Clustering:
    """One clustering's labels for the calibration and test units of one scale."""

    calib: np.ndarray  # cluster label of each calibrating unit
    test: np.ndarray  # cluster label of each test unit
    labels: Sequence[int] | None  # label values to process; None takes them from ``calib``
    sizes: np.ndarray | None  # units carrying each label; None uses calib plus test units


def _lattice_cluster_labels(
    workspace: Workspace,
    source: str,
    sample_names: Sequence[str],
    key: str,
    index: ScaleIndex,
) -> np.ndarray:
    """Look up each coarse unit's histology cluster label on the cluster images."""
    labels = np.empty(index.n_groups, dtype=np.int16)
    for sample, name in enumerate(sample_names):
        rows = index.sample_index == sample
        if not rows.any():
            continue
        with np.load(workspace.clusters(name, source)) as data:
            image = data[key]
        positions = index.positions[rows]
        labels[rows] = image[positions[:, 0], positions[:, 1]]
    return labels


def _lattice_cluster_sizes(
    workspace: Workspace, source: str, sample_names: Sequence[str], key: str, n_clusters: int
) -> np.ndarray:
    """Count the tissue units carrying each cluster label, over all samples.

    The ``(n_clusters,)`` counts are the denominator of the cluster coverage rule.
    """
    sizes = np.zeros(n_clusters, dtype=np.int64)
    for name in sample_names:
        with np.load(workspace.clusters(name, source)) as data:
            image = data[key]
        labels = image[image != BACKGROUND].astype(np.int64)
        sizes += np.bincount(labels, minlength=n_clusters)[:n_clusters]
    return sizes


@dataclass
class ScaleGrid:
    """Everything about one scale that does not depend on the target or method."""

    scale: int  # always 1 for an external store
    calib: ScaleIndex
    test: ScaleIndex
    keep: np.ndarray  # (n_calib_groups,) calibration units carrying truth; these calibrate
    evaluable: np.ndarray  # (n_test_groups,) test units carrying truth; a benchmark scores these
    # Cluster label -> clustering, with the calibration side restricted to ``keep``.
    clusters: dict[str, Clustering] = field(default_factory=dict)


@dataclass
class TargetScale:
    """One target's values at one scale, divided by ``divisor`` onto the scale of the null."""

    fit_y: np.ndarray  # measured calibration values, restricted to ``ScaleGrid.keep``
    fit_yhat: np.ndarray  # predicted calibration values on the same units
    test_yhat: np.ndarray  # predicted test values
    test_y: np.ndarray | None  # measured test values; None without test truth
    calib_y: np.ndarray  # measured values of all calibration units, painted on the ROI overlay
    divisor: float  # brings a raw summed value to the scale the null is stated on
    display: float  # further divisor used only when drawing, so a map falls in [0, 1]


@dataclass
class TargetPlan:
    """A target's per-scale values, prepared once and reused by every method."""

    target: Target
    scales: dict[int, TargetScale]


def _build_grid(
    config: Config,
    workspace: Workspace,
    sample_names: Sequence[str],
    calib: Store,
    test: Store,
    scale: int,
    span: int,
) -> ScaleGrid:
    """Group both stores at one scale and look up their cluster labels."""
    if config.is_external:
        calib_index, test_index = identity_index(calib), identity_index(test)
    else:
        calib_index = build_scale_index(calib, scale, span)
        test_index = build_scale_index(test, scale, span)
    # A calibration block calibrates when any of its children carries truth; its value still sums
    # over the whole block, so it stands on the same scale as a test block.
    keep = _aggregate(calib.evaluable.astype(np.float32), calib_index) > 0
    evaluable = _aggregate(test.evaluable.astype(np.float32), test_index) > 0

    grid = ScaleGrid(
        scale=scale, calib=calib_index, test=test_index, keep=keep, evaluable=evaluable
    )
    if not any(method in CLUSTER_METHODS for method in config.methods):
        return grid

    if config.is_external:
        # The author's own clusterings, whose labels appear verbatim in the output keys.
        for label, calib_labels in calib.clusterings.items():
            grid.clusters[label] = Clustering(
                calib=calib_labels[keep],
                test=test.clusterings[label],
                labels=None,
                sizes=None,
            )
        return grid

    source = config.cluster_source
    for n_clusters in config.n_clusters:
        key = f"k{n_clusters}_scale{scale}"
        grid.clusters[f"k{n_clusters}"] = Clustering(
            calib=_lattice_cluster_labels(workspace, source, sample_names, key, calib_index)[keep],
            test=_lattice_cluster_labels(workspace, source, sample_names, key, test_index),
            labels=range(n_clusters),
            sizes=_lattice_cluster_sizes(workspace, source, sample_names, key, n_clusters),
        )
    return grid


def _target_values(
    store: TargetStore, target: Target, weights: np.ndarray | None
) -> np.ndarray:
    """Reduce a target's channels to one value per unit: their (optionally weighted) sum."""
    rows = store.read(target.channels)
    if weights is None:
        return rows.sum(axis=0)
    return (rows * weights[:, None]).sum(axis=0)


def target_weights(store: TargetStore, target: Target) -> np.ndarray:
    """The weight of each member channel in a target's sum.

    A gene whose store records rescaling thresholds is weighted by its upper threshold, so the sum
    is on the count scale; every other channel has weight 1.
    """
    thresholds = store.meta.get("thresholds")
    if target.modality == "gene" and thresholds is not None:
        missing = [name for name in target.channels if name not in thresholds]
        if missing:
            # ``predict_genes`` may store a subset, so a correctly spelled gene can simply not
            # have been predicted; a bare KeyError would not say why.
            raise KeyError(
                f"target '{target.name}' needs gene(s) {missing}, which the store at "
                f"{store.directory} does not hold -- it was written for {len(thresholds)} "
                f"gene(s). Add them to 'predict_genes' and re-run `calibrate`, or drop them "
                f"from the target list."
            )
        return np.array([thresholds[name][1] for name in target.channels], dtype=np.float32)
    return np.ones(len(target.channels), dtype=np.float32)


def _measured_values(target: Target, stores: dict[str, object], role: str) -> np.ndarray:
    """One target's measured values on the ``calib`` or ``test`` side, one per scale-1 unit.

    A gene target sums its measured channels; a cell-type target is the 0/1 indicator that the
    unit's true cell type is one of its members.
    """
    if target.modality == "gene":
        return _target_values(stores[f"{role}_truth"], target, None)
    store: Store = stores[role]
    labels = np.load(store.directory / "truth_labels.npy")
    members = stores[f"{role}_pred"].channel_index(target.channels)
    return np.isin(labels, members).astype(np.float32)


def _norm(config: Config, target: Target, fit_y: np.ndarray) -> tuple[float, float]:
    """Return one target's ``(divisor, display)`` divisors at one scale from its calibration truth.

    The built-in model's gene targets are divided by an upper quantile of those values; cell-type
    and external targets are not rescaled, and a quantile only sets their display divisor.
    """
    rescale = target.modality == "gene" and not config.is_external
    norm = float(np.percentile(fit_y, NORM_QUANTILE)) if len(fit_y) else 1.0
    divisor = max(1.0, norm) if rescale else 1.0
    shown = float(np.percentile(fit_y / divisor, DISPLAY_QUANTILE)) if len(fit_y) else 1.0
    display = 1.0 if rescale else max(np.finfo(np.float32).tiny, shown)
    return divisor, display


def _build_plan(
    config: Config,
    target: Target,
    stores: dict[str, object],
    grids: dict[int, ScaleGrid],
) -> TargetPlan:
    """Read one target's channels and reduce them to per-scale value arrays.

    Above scale 1 a gene target's measured value also includes the off-tissue superpixels of each
    coarse unit when ``stores`` carries them; its prediction sums tissue superpixels only.
    """
    calib_pred: TargetStore = stores["calib_pred"]
    test_pred: TargetStore = stores["test_pred"]
    test: Store = stores["test"]
    gene = target.modality == "gene"

    weights = target_weights(calib_pred, target)
    calib_yhat_1 = _target_values(calib_pred, target, weights)
    test_yhat_1 = _target_values(test_pred, target, weights)
    calib_y_1 = _measured_values(target, stores, "calib")
    # The calibration side always carries truth; the test side only when it was supplied.
    if gene:
        test_has_truth = stores.get("test_truth") is not None
    else:
        test_has_truth = test.has("truth_labels")
    test_y_1 = _measured_values(target, stores, "test") if test_has_truth else None
    calib_off: OffTissue | None = stores.get("calib_off_tissue") if gene else None
    test_off: OffTissue | None = stores.get("test_off_tissue") if gene else None
    calib_off_counts = None if calib_off is None else calib_off.counts(target.channels)
    test_off_counts = (
        None if test_off is None or test_y_1 is None else test_off.counts(target.channels)
    )

    scales: dict[int, TargetScale] = {}
    for scale, grid in grids.items():
        calib_y = _aggregate(calib_y_1, grid.calib)
        if calib_off_counts is not None:
            calib_y = calib_off.add(calib_y, calib_off_counts, scale)
        calib_yhat = _aggregate(calib_yhat_1, grid.calib)
        test_yhat = _aggregate(test_yhat_1, grid.test)
        test_y = _aggregate(test_y_1, grid.test) if test_y_1 is not None else None
        if test_off_counts is not None:
            test_y = test_off.add(test_y, test_off_counts, scale)
        fit_y, fit_yhat = calib_y[grid.keep], calib_yhat[grid.keep]
        divisor, display = _norm(config, target, fit_y)
        scales[scale] = TargetScale(
            fit_y=fit_y / divisor,
            fit_yhat=fit_yhat / divisor,
            test_yhat=test_yhat / divisor,
            test_y=None if test_y is None else test_y / divisor,
            calib_y=calib_y / divisor,
            divisor=divisor,
            display=display,
        )
    return TargetPlan(target=target, scales=scales)


def _bh_groups(config: Config, index: ScaleIndex) -> np.ndarray | None:
    """Group id per test unit that the Benjamini-Hochberg correction runs within.

    This is the unit's sample for the per-sample scopes, and None for a single family.
    """
    if config.inference.bh_scope in ("sample_cluster", "sample"):
        return index.sample_index
    return None


def _confidence(config: Config, pvals_adj: np.ndarray) -> np.ndarray:
    """Adjusted p-values on the 0-to-1 display scale between the configured anchors."""
    return pvals_to_confidence(
        pvals_adj, p_min=config.inference.p_min, p_max=config.inference.p_max
    )


def _cluster_result(
    config: Config,
    target: Target,
    values: TargetScale,
    grid: ScaleGrid,
    clustering: Clustering,
) -> UncertaintyResult:
    """Mondrian split-conformal p-values, conditioned on histology cluster."""
    groups = _bh_groups(config, grid.test)
    # The ``sample`` and ``global`` scopes ask for a family the method's own per-cluster
    # adjustment does not describe, so the correction is retaken over the raw p-values.
    per_cluster = config.inference.bh_scope in ("sample_cluster", "cluster")
    pvals, pvals_adj = conformal_cluster_pvals(
        values.fit_y,
        values.fit_yhat,
        clustering.calib,
        values.test_yhat,
        clustering.test,
        null=target.null,
        clusters=clustering.labels,
        cluster_sizes=clustering.sizes,
        min_calib_units=config.inference.min_calib_units,
        # The batch-effect settings may differ by scale, so each is resolved at this grid's scale.
        correct_batch_effect=config.inference.at_scale("correct_batch_effect", grid.scale),
        magnitude=config.inference.at_scale("batch_effect_magnitude", grid.scale),
        aggregate=config.inference.at_scale("batch_effect_aggregate", grid.scale),
        min_null_fraction=float(
            config.inference.at_scale("batch_effect_null_fraction", grid.scale)
        ),
        bh_groups=groups if per_cluster else None,
    )
    if not per_cluster:
        pvals_adj = bh_adjust_grouped(pvals, groups)
    return UncertaintyResult(
        prediction=values.test_yhat,
        pvals=pvals,
        pvals_adj=pvals_adj,
        confidence=_confidence(config, pvals_adj),
    )


def _marginal_result(
    config: Config, target: Target, values: TargetScale, grid: ScaleGrid
) -> UncertaintyResult:
    """Interval-inversion split-conformal levels, with no clustering."""
    level, pvals, pvals_adj = conformal_marginal_levels(
        values.fit_y,
        values.fit_yhat,
        values.test_yhat,
        null=target.null,
        bh_groups=_bh_groups(config, grid.test),
    )
    return UncertaintyResult(
        prediction=values.test_yhat,
        pvals=pvals,
        pvals_adj=pvals_adj,
        confidence=_confidence(config, pvals_adj),
        level=level,
    )


def _method_result(
    config: Config,
    method: str,
    plan: TargetPlan,
    grid: ScaleGrid,
    clustering: Clustering | None,
    draws: dict[int, DrawAccumulator] | None,
) -> UncertaintyResult:
    """Apply one uncertainty method to one target at one scale.

    ``clustering`` is used by the cluster methods, and ``draws`` (scale -> accumulator) by the
    sampling methods.
    """
    values = plan.scales[grid.scale]
    if method == "none":
        return UncertaintyResult(prediction=values.test_yhat)
    if method in CLUSTER_METHODS:
        return _cluster_result(config, plan.target, values, grid, clustering)
    if method == "conformal_marginal":
        return _marginal_result(config, plan.target, values, grid)
    return draws[grid.scale].result(
        coverage=config.sampling.coverage,
        p_min=config.inference.p_min,
        p_max=config.inference.p_max,
        bh_groups=_bh_groups(config, grid.test),
    )


def slide_range(values: np.ndarray) -> tuple[float, float]:
    """Colour range of one slide's values: their 0.1th and 99.9th percentiles.

    The upper bound is kept above the lower one, so constant values still give a usable range.
    """
    if not len(values):
        return 0.0, 1.0
    low, high = np.percentile(np.asarray(values, dtype=np.float64), SLIDE_RANGE_PERCENTILES)
    low, high = float(low), float(high)
    if high <= low:
        high = low + 1e-6
    return low, high


def _shown(config: Config, values: np.ndarray, display: float) -> np.ndarray:
    """Bring one slide's values of one kind onto the 0-to-1 colour scale of a magnitude map.

    The built-in model divides by the target's display divisor, which every slide shares; an
    external model's values are normalised within the slide, between their :func:`slide_range`.
    """
    if not config.is_external:
        return values / display
    low, high = slide_range(values)
    return (np.asarray(values, dtype=np.float64) - low) / (high - low)


@dataclass
class Canvas:
    """Where one sample's units are drawn, and how big each of them is."""

    shape: tuple[int, int]  # (rows, columns)
    radius: float  # drawing radius of the sample's units; 0 draws each unit as one cell
    square: bool = False  # draw each unit as a square of half-width ``radius``, not a disk
    tissue: np.ndarray | None = None  # (rows, columns) tissue mask drawn behind the units


def _canvas_tissue(workspace: Workspace, sample: str, shape: tuple[int, int]) -> np.ndarray | None:
    """Tissue mask the ``import`` stage made for an external sample, or None if it made none.

    The mask is only used when it has exactly the canvas shape, so a store written by hand next
    to an unrelated mask is drawn without one.
    """
    path = workspace.tissue_mask(sample)
    if not path.exists():
        return None
    mask = load_mask(path)
    if mask.shape != tuple(shape):
        return None
    return mask


def _canvases(
    workspace: Workspace,
    geometries: Sequence[Geometry],
    store: Store,
    sample_names: Sequence[str],
    scale: int,
) -> dict[str, Canvas]:
    """The canvas each sample's maps are drawn on: a built-in sample's lattice at this scale.

    An external sample uses the ``canvas_shape`` its ``meta.json`` declares, or else the bounding
    box of its positions, with the declared ``spot_shape`` and the workspace's tissue mask if any.
    """
    if geometries:
        return {
            geometry.name: Canvas(coarse_shape(geometry.lattice_shape, scale), 0.0)
            for geometry in geometries
        }
    if store.positions is None:
        return {}

    declared = store.meta.get("canvas_shape")
    square = store.meta.get("spot_shape") == "square"
    # ``radius.npy`` is either one value for the whole store or one per unit.
    radii = (
        None
        if store.radius is None
        else np.broadcast_to(np.atleast_1d(store.radius).ravel(), (len(store.sample_index),))
    )
    canvases: dict[str, Canvas] = {}
    for index, name in enumerate(sample_names):
        rows = store.sample_index == index
        radius = 0.0 if radii is None or not rows.any() else float(radii[rows].max())
        if declared is not None:
            shape = (int(declared[index][0]), int(declared[index][1]))
        elif rows.any():
            extent = store.positions[rows].max(axis=0) + int(np.ceil(radius)) + 1
            shape = (int(extent[0]), int(extent[1]))
        else:
            continue
        canvases[name] = Canvas(
            shape, radius, square=square, tissue=_canvas_tissue(workspace, name, shape)
        )
    return canvases


def _render(
    canvas: Canvas,
    positions: np.ndarray,
    values: np.ndarray,
    path: Path,
    colour: str,
    overlay: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Write one sample's map of per-unit ``values``, which are on a 0-to-1 display scale.

    ``colour`` is ``"confidence"`` or ``"magnitude"``; ``overlay`` is an optional pair of
    calibration positions and measured values painted over the ROIs.
    """
    from . import plot

    image, drawn = plot.paint(values, positions, canvas.shape, canvas.radius, canvas.square)
    truth_image, truth_drawn = None, None
    if overlay is not None and len(overlay[0]):
        truth_image, truth_drawn = plot.paint(
            overlay[1], overlay[0], canvas.shape, canvas.radius, canvas.square
        )

    renderer = plot.confidence_map if colour == "confidence" else plot.magnitude_map
    save_image(renderer(image, drawn, truth_image, truth_drawn, tissue=canvas.tissue), path)


@dataclass
class SampleView:
    """One test sample's rows within a scale's test and calibration units."""

    name: str
    rows: np.ndarray  # (n,) indices of this sample's coarse test units
    canvas: Canvas | None
    positions: np.ndarray | None  # (n, 2) positions of those test units, or None
    calib_rows: np.ndarray  # (m,) indices of this sample's calibration units
    calib_positions: np.ndarray | None  # (m, 2) their positions, or None


def _sample_views(
    grid: ScaleGrid, canvases: dict[str, Canvas], sample_names: Sequence[str]
) -> list[SampleView]:
    """Split a scale's coarse units by test sample; samples with none are skipped."""
    views = []
    for index, name in enumerate(sample_names):
        rows = np.flatnonzero(grid.test.sample_index == index)
        if not len(rows):
            continue
        calib_rows = np.flatnonzero(grid.calib.sample_index == index)
        views.append(
            SampleView(
                name=name,
                rows=rows,
                canvas=canvases.get(name),
                positions=None if grid.test.positions is None else grid.test.positions[rows],
                calib_rows=calib_rows,
                calib_positions=(
                    None if grid.calib.positions is None else grid.calib.positions[calib_rows]
                ),
            )
        )
    return views


def _write_units(
    workspace: Workspace,
    config: Config,
    modality: str,
    grid: ScaleGrid,
    views: Sequence[SampleView],
) -> None:
    """Write the per-(sample, modality, scale) arrays every target shares."""
    for view in views:
        arrays: dict[str, np.ndarray] = {
            "evaluable": grid.evaluable[view.rows],
            "row_index": grid.test.rows[view.rows],
        }
        if view.positions is not None:
            arrays["positions"] = view.positions.astype(np.int32)
        if view.calib_positions is not None:
            arrays["calib_positions"] = view.calib_positions.astype(np.int32)
        if view.canvas is not None:
            arrays["canvas_shape"] = np.array(view.canvas.shape, dtype=np.int32)
            arrays["radius"] = np.array(view.canvas.radius, dtype=np.float32)
            arrays["square"] = np.array(view.canvas.square, dtype=bool)
            if view.canvas.tissue is not None:
                # Kept with the units so the viewer can paint the tissue behind the spots.
                arrays["tissue_mask"] = np.asarray(view.canvas.tissue, dtype=bool)
        for label, clustering in grid.clusters.items():
            arrays[f"clusters_{label}"] = clustering.test[view.rows].astype(np.int16)
        path = workspace.units_file(config.model, view.name, modality, grid.scale)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)


def _write_target_arrays(
    workspace: Workspace,
    config: Config,
    plan: TargetPlan,
    grid: ScaleGrid,
    views: Sequence[SampleView],
) -> dict[str, dict[str, list[float]]]:
    """Write one scale's point prediction, test truth and calibration truth of one target.

    An external run also gets a map of the test truth and returns each slide's colour ranges,
    sample -> ``prediction`` / ``truth`` / ``calib_truth`` -> ``[low, high]``; a built-in run
    returns an empty mapping.
    """
    values = plan.scales[grid.scale]
    ranges: dict[str, dict[str, list[float]]] = {}
    for view in views:
        directory = workspace.target_dir(config.model, view.name, plan.target.name)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = f"scale{grid.scale}"
        np.save(
            directory / f"prediction_{suffix}.npy",
            values.test_yhat[view.rows].astype(np.float32),
        )
        if values.test_y is not None:
            np.save(directory / f"truth_{suffix}.npy", values.test_y[view.rows].astype(np.float32))
        if len(view.calib_rows):
            np.save(
                directory / f"calib_truth_{suffix}.npy",
                values.calib_y[view.calib_rows].astype(np.float32),
            )
        if not config.is_external:
            continue

        slide_ranges = {"prediction": list(slide_range(values.test_yhat[view.rows]))}
        if values.test_y is not None:
            slide_ranges["truth"] = list(slide_range(values.test_y[view.rows]))
        if len(view.calib_rows):
            slide_ranges["calib_truth"] = list(slide_range(values.calib_y[view.calib_rows]))
        ranges[view.name] = slide_ranges

        # The prediction map is drawn per method; the measured truth has one map per target.
        if config.inference.visualize and view.canvas is not None and values.test_y is not None:
            _render(
                view.canvas,
                view.positions,
                _shown(config, values.test_y[view.rows], values.display),
                directory / f"truth_{suffix}.png",
                colour="magnitude",
            )
    return ranges


def _write_method(
    config: Config,
    workspace: Workspace,
    plan: TargetPlan,
    grids: dict[int, ScaleGrid],
    views: dict[int, Sequence[SampleView]],
    method: str,
    draws: dict[int, DrawAccumulator] | None,
) -> None:
    """Write one target's results under one uncertainty method, sample by sample.

    Arrays of the cluster methods are keyed ``<label>_scale{s}`` and all others ``scale{s}``; every
    scale goes into the same ``result.npz``.
    """
    target = plan.target
    names = {view.name for scale in views for view in views[scale]}
    collected: dict[str, dict[str, np.ndarray]] = {name: {} for name in names}

    for scale, grid in grids.items():
        values = plan.scales[scale]
        labels: list[str | None] = list(grid.clusters) if method in CLUSTER_METHODS else [None]
        prediction = draws[scale].mean if method in SAMPLING_METHODS else values.test_yhat
        for view in views[scale]:
            method_dir = workspace.method_dir(config.model, view.name, target.name, method)
            method_dir.mkdir(parents=True, exist_ok=True)
            if method in SAMPLING_METHODS:
                # The only prediction that belongs to the method and not the target: a mean over
                # draws, not the model's output.
                np.save(
                    method_dir / f"prediction_scale{scale}.npy",
                    prediction[view.rows].astype(np.float32),
                )
            if config.inference.visualize and view.canvas is not None:
                _render(
                    view.canvas,
                    view.positions,
                    _shown(config, prediction[view.rows], values.display),
                    method_dir / f"prediction_scale{scale}.png",
                    colour="magnitude",
                    overlay=_overlay(config, view, values),
                )

        for label in labels:
            clustering = None if label is None else grid.clusters[label]
            result = _method_result(config, method, plan, grid, clustering, draws)
            suffix = f"scale{scale}" if label is None else f"{label}_scale{scale}"

            for view in views[scale]:
                null_dir = workspace.null_dir(
                    config.model, view.name, target.name, method, target.null_name
                )
                _write_view_result(
                    config, target, view, grid, values, result, scale, suffix, null_dir,
                    collected[view.name], config.inference.visualize,
                )

            if (
                len(views[scale]) > 1
                and values.test_y is not None
                and result.pvals_adj is not None
            ):
                pooled = workspace.pooled_null_dir(
                    config.model, target.name, method, target.null_name
                )
                _write_pooled_curve(
                    target, grid, values, result, pooled / f"fdp_curve_{suffix}.png"
                )

    for name, arrays in collected.items():
        null_dir = workspace.null_dir(config.model, name, target.name, method, target.null_name)
        null_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(null_dir / "result.npz", **arrays)


def _write_view_result(
    config: Config,
    target: Target,
    view: SampleView,
    grid: ScaleGrid,
    values: TargetScale,
    result: UncertaintyResult,
    scale: int,
    suffix: str,
    null_dir: Path,
    arrays: dict[str, np.ndarray],
    visualize: bool,
) -> None:
    """Add one sample's arrays of one result to ``arrays``, keyed by ``suffix``.

    With ``visualize``, the sample's FDP curve and confidence map are drawn into ``null_dir``.
    """
    from . import plot

    for name, array in result.arrays().items():
        if name != "prediction":
            arrays[f"{name}_{suffix}"] = array[view.rows].astype(np.float32)

    if values.test_y is not None and result.pvals_adj is not None:
        scored = view.rows[grid.evaluable[view.rows]]
        fdr, fdp, discoveries = fdp_curve(
            result.pvals_adj[scored], values.test_y[scored], target.null
        )
        arrays[f"fdr_scale{scale}"] = fdr
        arrays[f"fdp_{suffix}"] = fdp
        arrays[f"n_discoveries_{suffix}"] = discoveries
        if visualize:
            plot.save_fdp_curve(fdr, fdp, discoveries, null_dir / f"fdp_curve_{suffix}.png")

    if visualize and view.canvas is not None and result.confidence is not None:
        _render(
            view.canvas,
            view.positions,
            result.confidence[view.rows],
            null_dir / f"confidence_{suffix}.png",
            colour="confidence",
            overlay=_overlay(config, view, values),
        )


def _write_pooled_curve(
    target: Target, grid: ScaleGrid, values: TargetScale, result: UncertaintyResult, path: Path
) -> None:
    """Draw the FDP curve of every evaluable test unit at once, across samples.

    The pooled curve is a comparison, not the guarantee, which holds per sample.
    """
    from . import plot

    fdr, fdp, discoveries = fdp_curve(
        result.pvals_adj[grid.evaluable], values.test_y[grid.evaluable], target.null
    )
    plot.save_fdp_curve(fdr, fdp, discoveries, path)


def _overlay(
    config: Config, view: SampleView, values: TargetScale
) -> tuple[np.ndarray, np.ndarray] | None:
    """Calibration positions and display-scaled measured values for this sample's ROI overlay.

    Returns None when the sample has no calibration units or no positions.
    """
    if not len(view.calib_rows) or view.calib_positions is None:
        return None
    shown = _shown(config, values.calib_y[view.calib_rows], values.display)
    return view.calib_positions, shown


def _file_digest(path: Path) -> dict[str, object]:
    """Size and SHA-256 of one store file, recorded so a results tree identifies its inputs."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _write_run_meta(
    config: Config,
    workspace: Workspace,
    stores: dict[str, Path],
    targets: dict[str, dict[str, object]],
    display: dict[str, dict[str, float]],
    nulls: dict[str, list[dict[str, object]]],
    cluster_labels: Sequence[str],
    samples: Sequence[str],
    draw_stores: dict[str, dict[str, object]],
    off_tissue: dict[str, dict[str, int]],
    display_range: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
) -> None:
    """Write ``run_meta.json``: the settings, stores, targets, nulls and display divisors of a run.

    Entries of an existing file are kept unless this run overwrites them; ``display_range`` holds
    the per-slide colour ranges of an external run's maps, keyed target, sample, scale and kind.
    """
    path = workspace.run_meta(config.model)
    existing = read_json(path) if path.exists() else {}
    display = {**existing.get("display_divisor", {}), **display}
    display_range = {**existing.get("display_range", {}), **display_range}
    meta = {
        "model": config.model,
        "mode": config.mode,
        "methods": config.methods,
        "bh_scope": config.inference.bh_scope,
        "cluster_labels": list(cluster_labels),
        "scales": list(config.scales),
        "samples": list(samples),
        "confidence_scale": {"p_min": config.inference.p_min, "p_max": config.inference.p_max},
        # Resolved per scale, so the file says what was applied, not the mapping it came from.
        "batch_effect": {
            str(scale): {
                name: config.inference.at_scale(name, scale)
                for name in config.inference.PER_SCALE
            }
            for scale in config.scales
        },
        "min_calib_units": config.inference.min_calib_units,
        "display_divisor": display,
        "display_range": display_range,
        "targets": {
            name: {**entry, "nulls": nulls[name]} for name, entry in targets.items()
        },
        "stores": {
            key: {"path": str(value), **_file_digest(value)} for key, value in stores.items()
        },
        "draw_stores": draw_stores,
        "off_tissue_truth": off_tissue,
    }
    write_json({**existing, **meta}, path)


def _prepare_draws(
    config: Config,
    workspace: Workspace,
    geometries: Sequence[Geometry],
    modality: str,
    sampling: Sequence[str],
    targets: Sequence[Target],
    stores: dict[str, object],
) -> dict[str, TargetStore]:
    """Generate or reuse each sampling method's draw store, before the target loop begins.

    Returns method -> its ``(targets, K, units)`` store; a target under several nulls is one row.
    """
    if not sampling:
        return {}

    from .calibration import draws as draw_stage

    test: Store = stores["test"]
    seen: dict[str, Target] = {}
    for target in targets:
        seen.setdefault(target.name, target)
    specs = [
        draw_stage.DrawTarget(
            name=target.name,
            channels=list(target.channels),
            weights=target_weights(stores["calib_pred"], target),
        )
        for target in seen.values()
    ]
    found: dict[str, TargetStore] = {}
    for method in sampling:
        directory = draw_stage.generate(
            config,
            workspace,
            geometries,
            modality,
            method,
            test.sample_index,
            test.positions,
            specs,
        )
        found[method] = TargetStore.open(directory, "pred")
    return found


def _target_draws(
    store: TargetStore, target: Target, plan: TargetPlan, grids: dict[int, ScaleGrid]
) -> dict[int, DrawAccumulator]:
    """Reduce one target's stored draws to an accumulator at each scale.

    Each draw is summed to the scale and divided by the target's divisor before it is compared
    with the null.
    """
    draws = store.read([target.name])[0]
    found: dict[int, DrawAccumulator] = {}
    for scale, grid in grids.items():
        divisor = plan.scales[scale].divisor
        accumulator = DrawAccumulator(null=target.null, n_units=grid.test.n_groups)
        for draw in draws:
            accumulator.update(_aggregate(draw, grid.test) / divisor)
        found[scale] = accumulator
    return found


def run(config: Config) -> None:
    """Run every configured uncertainty method over every target, one at a time."""
    workspace = Workspace(config)
    geometries = [] if config.is_external else workspace.load_geometries()
    span = max((max(geometry.lattice_shape) for geometry in geometries), default=1)
    targets = load_targets(config.targets_path)
    methods = config.methods
    sampling = [method for method in methods if method in SAMPLING_METHODS]
    modalities = (
        external_modalities(workspace, config.model) if config.is_external else config.predict
    )
    print(f"[infer] model '{config.model}' ({config.mode}); method(s): {', '.join(methods)}")

    # What run_meta.json needs, kept as plain numbers so that no target's arrays have to outlive
    # its turn through the loop.
    meta_by_target: dict[str, dict[str, object]] = {}
    display_by_target: dict[str, dict[str, float]] = {}
    # External only: target -> sample -> scale -> kind -> [low, high], the per-slide colour ranges.
    range_by_target: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    nulls_by_target: dict[str, list[dict[str, object]]] = {}
    cluster_labels: list[str] = []
    digests: dict[str, Path] = {}
    drawn_stores: dict[str, dict[str, object]] = {}
    off_tissue: dict[str, dict[str, int]] = {}
    sample_names: list[str] = []

    for modality in modalities:
        selected = [target for target in targets if target.modality == modality]
        if not selected:
            continue
        stores = open_stores(workspace, config.model, modality)
        calib, test = stores["calib"], stores["test"]
        names = [geometry.name for geometry in geometries] or test.sample_names
        sample_names = sample_names or names
        digests[f"calibration/{modality}"] = calib.directory / "pred.npy"
        digests[f"predictions/{modality}"] = test.directory / "pred.npy"

        print(f"[infer] {modality}: {len(selected)} target(s)")
        grids = {
            scale: _build_grid(config, workspace, names, calib, test, scale, span)
            for scale in config.scales
        }
        canvases = {
            scale: _canvases(workspace, geometries, test, names, scale) for scale in grids
        }
        views = {
            scale: _sample_views(grid, canvases[scale], names) for scale, grid in grids.items()
        }
        if modality == "gene" and not config.is_external:
            off_tissue[modality] = add_off_tissue(stores, config, geometries, grids, span)
            added = ", ".join(f"{n} {role}" for role, n in off_tissue[modality].items())
            print(f"[infer] gene: coarse truth adds off-tissue superpixels ({added})" if added else
                  "[infer] gene: coarse truth sums tissue superpixels only (no genes/ input)")

        # Done before anything is written, so a missing draw bundle fails here and not half-way
        # through a results tree.
        draw_stores = _prepare_draws(
            config, workspace, geometries, modality, sampling, selected, stores
        )
        # Each draw store is recorded by path, size and the signature in its own meta.json; it
        # is too large to hash on every run.
        for method, store in draw_stores.items():
            drawn_stores[f"{modality}/{method}"] = {
                "path": str(store.path),
                "bytes": store.path.stat().st_size,
                "signature": store.meta.get("draw_signature"),
            }

        for scale, grid in grids.items():
            _write_units(workspace, config, modality, grid, views[scale])
            cluster_labels = sorted(set(cluster_labels) | set(grid.clusters))

        for target in selected:
            plan = _build_plan(config, target, stores, grids)
            meta_by_target[target.name] = {
                "modality": target.modality,
                "channels": list(target.channels),
            }
            display_by_target[target.name] = {
                str(scale): float(values.display) for scale, values in plan.scales.items()
            }
            entry = {"name": target.null_name, "value": target.null}
            recorded = nulls_by_target.setdefault(target.name, [])
            if entry not in recorded:
                recorded.append(entry)

            for scale, grid in grids.items():
                slide_ranges = _write_target_arrays(workspace, config, plan, grid, views[scale])
                for sample_name, kinds in slide_ranges.items():
                    per_sample = range_by_target.setdefault(target.name, {})
                    per_sample.setdefault(sample_name, {})[str(scale)] = kinds
            for method in methods:
                drawn = (
                    _target_draws(draw_stores[method], target, plan, grids)
                    if method in SAMPLING_METHODS
                    else None
                )
                _write_method(config, workspace, plan, grids, views, method, drawn)
                del drawn
            # Free this target's arrays before the next target is read.
            del plan
        print(f"[infer] {modality}: {len(methods)} method(s) x {len(selected)} target(s) written")

    _write_run_meta(
        config,
        workspace,
        digests,
        meta_by_target,
        display_by_target,
        nulls_by_target,
        cluster_labels,
        sample_names,
        drawn_stores,
        off_tissue,
        range_by_target,
    )
    print(f"[infer] wrote {workspace.results_root(config.model)}")
