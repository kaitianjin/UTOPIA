"""Building one scale's own units: mean features in, summed counts out.

A coarse unit at scale ``s`` is the ``s x s`` block of scale-1 superpixels that share a parent cell
in the lattice, the grouping :func:`utopia.inference.build_scale_index` reports at.  Its feature
vector is the mean over its superpixels and its target the sum of their counts, so each gene
baseline can train one model per scale; an edge block is taken with the children it has.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..aggregate import (
    block_mean_features,
    decode_lattice_keys,
    group_index,
    lattice_keys,
)
from ..calibration.run import (
    TrainingInputs,
    UnitTable,
    _build_table,
    _gather_counts,
    _gather_features,
    _gene_thresholds,
    _predicted_channels,
    _scale_counts,
    training_inputs,
)
from ..config import Config
from ..geometry import Geometry
from ..io import create_memmap, open_memmap
from ..workspace import Workspace

#: Scale-1 rows reduced per pass when coarsening a memory-mapped array.  Bounds
#: the float32 temporary rather than the output, which is ``1 / s^2`` the size.
CHUNK = 200_000
#: Smallest batch :func:`coarse_batch_size` will shrink to, so that a very coarse
#: scale over a small ROI still trains on batches rather than near-singletons.
MIN_BATCH = 64


def coarse_batch_size(config: Config, scale: int) -> int:
    """Training batch size at ``scale``: scale 1's over ``scale^2``, never below :data:`MIN_BATCH`.

    A coarse training set has about ``1 / scale^2`` the rows, so shrinking the batch with it keeps
    the gradient steps per epoch the same at every scale.
    """
    base = int(config.default.batch_size)
    return max(MIN_BATCH, base // (int(scale) * int(scale)))


@dataclass
class CoarseGrouping:
    """How scale-1 units group into the units reported at one scale."""

    scale: int  # coarsening factor; 1 leaves the units alone
    inverse: np.ndarray  # (N,) coarse-unit id of every scale-1 unit
    n_groups: int  # number of coarse units
    counts: np.ndarray  # (n_groups,) children each coarse unit actually has
    sample_index: np.ndarray  # (n_groups,) sample id of each coarse unit
    positions: np.ndarray  # (n_groups, 2) coarse lattice position of each coarse unit

    @property
    def is_identity(self) -> bool:
        """Whether every coarse unit is one scale-1 unit, in the original order."""
        return self.scale == 1


def group_units(
    sample_index: np.ndarray, positions: np.ndarray, scale: int, span: int
) -> CoarseGrouping:
    """Group scale-1 units into their parent cells at ``scale``.

    ``span`` must bound both lattice dimensions and be the value the test side is given
    (:func:`utopia.inference.build_scale_index`); too small a bound silently merges distinct blocks.
    """
    sample_index = np.asarray(sample_index, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    if scale == 1:
        rows = np.arange(len(sample_index), dtype=np.int64)
        return CoarseGrouping(
            scale=1,
            inverse=rows,
            n_groups=len(rows),
            counts=np.ones(len(rows), dtype=np.int64),
            sample_index=sample_index,
            positions=positions,
        )
    keys = lattice_keys(sample_index, positions, scale, span)
    unique, inverse, counts = group_index(keys)
    coarse_samples, coarse_positions = decode_lattice_keys(unique, scale, span)
    return CoarseGrouping(
        scale=scale,
        inverse=inverse,
        n_groups=len(unique),
        counts=counts,
        sample_index=coarse_samples.astype(np.int64),
        positions=coarse_positions,
    )


def _accumulate(out: np.ndarray, ids: np.ndarray, block: np.ndarray) -> None:
    """Add one chunk of rows, ``block``, into the rows of ``out`` that ``ids`` assigns them to."""
    order = np.argsort(ids, kind="stable")
    ordered = ids[order]
    starts = np.flatnonzero(np.concatenate(([True], ordered[1:] != ordered[:-1])))
    out[ordered[starts]] += np.add.reduceat(block[order], starts, axis=0)


def coarse_sum(
    values: np.ndarray, grouping: CoarseGrouping, out: np.ndarray, chunk: int = CHUNK
) -> np.ndarray:
    """Sum a ``(N, D)`` array over each coarse unit, chunk by chunk, into ``out``.

    ``values`` may be a memory map larger than memory; ``out`` is overwritten in place and returned.
    """
    out[...] = 0
    for start in range(0, values.shape[0], chunk):
        stop = min(start + chunk, values.shape[0])
        _accumulate(
            out,
            grouping.inverse[start:stop],
            np.asarray(values[start:stop], dtype=np.float32),
        )
    return out


def coarse_mean(
    values: np.ndarray, grouping: CoarseGrouping, out: np.ndarray, chunk: int = CHUNK
) -> np.ndarray:
    """Average a ``(N, D)`` array over each coarse unit into ``out``, as :func:`coarse_sum` sums."""
    if out.dtype == np.float32:
        sums = coarse_sum(values, grouping, out, chunk)
        sums /= grouping.counts[:, None]
        return sums
    # A float16 destination would lose precision in the running sum, so the sum is
    # accumulated in float32 and cast once at the end.
    sums = coarse_sum(
        values, grouping, np.zeros((grouping.n_groups, values.shape[1]), dtype=np.float32), chunk
    )
    sums /= grouping.counts[:, None]
    out[...] = sums.astype(out.dtype)
    return out


def lattice_block_features(
    workspace: Workspace, config: Config, geometries: Sequence[Geometry], grouping: CoarseGrouping
) -> np.ndarray:
    """Mean feature of every coarse unit over all of its superpixels, ``(n_groups, D)`` float32.

    Used with the ``very_slow`` extractor, whose feature files hold every superpixel of the
    lattice; with the other extractors the mean is over a unit's tissue superpixels instead.
    """
    first = np.load(workspace.features(geometries[0].name, config.extractor), mmap_mode="r")
    values = np.zeros((grouping.n_groups, first.shape[1]), dtype=np.float32)
    for index, geometry in enumerate(geometries):
        selected = np.flatnonzero(grouping.sample_index == index)
        if len(selected) == 0:
            continue
        features = np.load(workspace.features(geometry.name, config.extractor), mmap_mode="r")
        values[selected] = block_mean_features(
            features, geometry.lattice_shape, grouping.scale, grouping.positions[selected]
        )
    return values


class CoarseFeatures:
    """The mean feature vector of every coarse test unit, ready to predict on.

    Offers the same ``n_units`` / ``dim`` / :meth:`batches` interface as
    :class:`utopia.calibration.draws.TestFeatures`, and is built only above scale 1.
    """

    def __init__(
        self,
        fine: object,
        grouping: CoarseGrouping,
        chunk: int = CHUNK,
        lattice_values: np.ndarray | None = None,
    ) -> None:
        """Average the ``fine`` features over each coarse unit, or take ``lattice_values`` as is."""
        self.n_units = int(grouping.n_groups)
        self.dim = int(fine.dim)
        if lattice_values is not None:
            # Already averaged over every superpixel of each block (``very_slow``).
            self.values = lattice_values
            return
        self.values = np.zeros((self.n_units, self.dim), dtype=np.float32)
        for start, stop, block in fine.batches(chunk):
            _accumulate(
                self.values,
                grouping.inverse[start:stop],
                np.asarray(block, dtype=np.float32),
            )
        self.values /= np.maximum(grouping.counts, 1)[:, None]

    def batches(self, batch_size: int) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield ``(start, stop, features)`` blocks covering every coarse unit once."""
        for start in range(0, self.n_units, batch_size):
            stop = min(start + batch_size, self.n_units)
            yield start, stop, self.values[start:stop]


def coarse_training_inputs(
    config: Config,
    workspace: Workspace,
    geometries: Sequence[Geometry],
    modality: str,
    scale: int,
    span: int,
    scratch: Path,
    thresholds: np.ndarray | None = None,
    rescale: bool = True,
) -> TrainingInputs:
    """Assemble the ROI training set at one scale: per block, mean features and summed counts.

    Scale 1 with ``rescale`` is the pipeline's own ``training_inputs``; otherwise (genes only) the
    ``[0, 1]`` thresholds come from this scale's block sums, or the targets stay raw counts.
    """
    if scale == 1 and rescale:
        return training_inputs(config, workspace, geometries, modality, scratch, thresholds)
    if modality != "gene":
        raise ValueError(
            f"a {modality} training set has no coarse form: its truth is a count of member "
            "superpixels and its model is a classifier over scale-1 labels, which the "
            "Poisson-binomial tail already reads at any scale"
        )

    channels, columns = _predicted_channels(config, modality)
    table = _build_table(config, geometries, "train", modality, restrict=True)
    grouping = group_units(table.sample_index, table.positions, scale, span)
    scratch.mkdir(parents=True, exist_ok=True)

    if config.features_cover_lattice:
        block_features = lattice_block_features(workspace, config, geometries, grouping)
        features = create_memmap(scratch / "train_x.npy", block_features.shape, np.float16)
        features[...] = block_features.astype(np.float16)
        features.flush()
        del block_features
    else:
        fine_x = _gather_features(workspace, config, geometries, table, scratch / "fine_x.npy")
        features = create_memmap(
            scratch / "train_x.npy", (grouping.n_groups, fine_x.shape[1]), np.float16
        )
        coarse_mean(fine_x, grouping, features)
        features.flush()
        del fine_x
        (scratch / "fine_x.npy").unlink()

    fine_counts = _gather_counts(
        config, geometries, table, len(channels), scratch / "fine_counts.npy", columns
    )
    # A raw-count set accumulates straight into the array the trainer reads.
    counts_path = scratch / ("coarse_counts.npy" if rescale else "train_y.npy")
    counts = create_memmap(counts_path, (grouping.n_groups, len(channels)), np.float32)
    coarse_sum(fine_counts, grouping, counts)
    counts.flush()
    del fine_counts
    (scratch / "fine_counts.npy").unlink()

    if rescale:
        thresholds = _gene_thresholds(counts, len(channels))
        targets = _scale_counts(counts, thresholds, scratch / "train_y.npy")
        del counts
        counts_path.unlink()
    else:
        thresholds = None
        targets = counts

    hyper = {
        "epochs": config.default.epochs,
        "batch_size": coarse_batch_size(config, scale),
        "learning_rate": config.default.learning_rate,
        "dropout": config.default.dropout,
        "loss": config.default.loss,
        "beta": config.default.beta,
        "seed": config.default.seed,
    }
    coarse_table = UnitTable(
        sample_index=grouping.sample_index,
        positions=grouping.positions,
        value_rows=np.arange(grouping.n_groups, dtype=np.int64),
        evaluable=np.ones(grouping.n_groups, dtype=bool),
    )
    return TrainingInputs(
        coarse_table, channels, open_memmap(scratch / "train_x.npy", mode="r"),
        targets, hyper, thresholds, columns,
    )
