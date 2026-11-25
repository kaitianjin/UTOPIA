"""Every path the pipeline reads or writes.

No other module builds paths by hand. Names carry their provenance (extractor, model, method,
null value), so several configurations can coexist in one workspace.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .geometry import Geometry


class Workspace:
    """Resolves workspace paths for one :class:`~utopia.config.Config`."""

    def __init__(self, config: Config) -> None:
        """Root every path at the configured workspace directory."""
        self.config = config
        self.root = config.workspace

    def sample_dir(self, sample: str) -> Path:
        """Directory holding one sample's processed image, geometry and features."""
        return self.root / "samples" / sample

    def he(self, sample: str) -> Path:
        """Rescaled, padded H&E image at ``superpixel_size / 16`` microns per pixel."""
        return self.sample_dir(sample) / "he.png"

    def tissue_mask(self, sample: str) -> Path:
        """Superpixel-resolution tissue mask produced by HistoSweep.

        For an external sample it is on the grid the maps are drawn on instead.
        """
        return self.sample_dir(sample) / "tissue_mask.png"

    def geometry(self, sample: str) -> Path:
        """Tissue mask, ROI boxes, roles and measured-region masks."""
        return self.sample_dir(sample) / "geometry.npz"

    def features(self, sample: str, extractor: str) -> Path:
        """Feature array, ``(N, D)`` float16, one row per tissue superpixel in mask order.

        The ``very_slow`` extractor instead writes one row per lattice superpixel, row-major; an
        external run stores ``(N, 1024)`` rows, one per spot in ``positions.npy`` order.
        """
        return self.sample_dir(sample) / f"features_{extractor}.npy"

    def features_meta(self, sample: str, extractor: str) -> Path:
        """Provenance of the feature array next to it."""
        return self.sample_dir(sample) / f"features_{extractor}.json"

    def clusters(self, sample: str, source: str) -> Path:
        """Dense cluster-label images, keyed ``k{n}_scale{s}``; grown incrementally."""
        return self.sample_dir(sample) / f"clusters_{source}.npz"

    def cluster_map(self, sample: str, source: str, key: str) -> Path:
        """Rendered cluster-label map for one ``k{n}_scale{s}`` key."""
        return self.sample_dir(sample) / f"clusters_{source}_{key}.png"

    def load_geometries(self) -> list[Geometry]:
        """Load every sample's geometry, in configuration order."""
        return [Geometry.load(self.geometry(name)) for name in self.config.sample_names]

    def clustering_model(self, source: str, scale: int, n_clusters: int) -> Path:
        """Fitted PCA + KMeans pair, reusable for transferring labels to new data."""
        return self.root / "models" / "clustering" / f"{source}_scale{scale}_k{n_clusters}.pkl"

    # Everything the three draw-based methods need sits under one tree, apart from the default
    # path; a workspace that never asks for them never grows this directory.
    def other_uq_dir(self, model: str, modality: str) -> Path:
        """Root of one modality's sampling-method apparatus."""
        return self.root / "other_UQ" / model / modality

    def draw_models(self, model: str, modality: str, kind: str) -> Path:
        """``.pt`` bundle of trained draw models; ``kind`` is full, ensemble or resampling."""
        return self.other_uq_dir(model, modality) / "models" / f"{kind}.pt"

    def draw_predictions(self, model: str, modality: str, method: str) -> Path:
        """Store directory of one sampling method's K draws, ``(targets, K, units)`` at scale 1."""
        return self.other_uq_dir(model, modality) / method

    def calibration_dir(self, model: str, modality: str) -> Path:
        """Directory of calibration predictions and truth for one modality."""
        return self.root / "calibration" / model / modality

    def prediction_dir(self, model: str, modality: str) -> Path:
        """Directory of test-area predictions for one modality."""
        return self.root / "predictions" / model / modality

    def folds(self, model: str) -> Path:
        """Cross-validation fold boxes used to build in-sample calibration data."""
        return self.root / "calibration" / model / "folds.npz"

    def scratch(self, model: str) -> Path:
        """Directory for large intermediates that are deleted once a stage finishes."""
        return self.root / "calibration" / model / "scratch"

    # A test sample sits above the target because the multiplicity correction is per sample:
    # ``results/<model>/<sample>/`` is one self-contained subtree.
    def results_root(self, model: str) -> Path:
        """Root of one model's results tree."""
        return self.root / "results" / model

    def run_meta(self, model: str) -> Path:
        """What produced a results tree: stores and hashes, methods, BH scope."""
        return self.results_root(model) / "run_meta.json"

    def sample_results(self, model: str, sample: str) -> Path:
        """One test sample's self-contained results subtree."""
        return self.results_root(model) / sample

    def units_file(self, model: str, sample: str, modality: str, scale: int) -> Path:
        """Positions, evaluation mask and cluster labels shared by every target."""
        return self.sample_results(model, sample) / f"units_{modality}_scale{scale}.npz"

    def target_dir(self, model: str, sample: str, target: str) -> Path:
        """Directory of one target's per-sample results, above the methods."""
        return self.sample_results(model, sample) / target

    def method_dir(self, model: str, sample: str, target: str, method: str) -> Path:
        """Directory of one target's results under one uncertainty method."""
        return self.target_dir(model, sample, target) / method

    def null_dir(self, model: str, sample: str, target: str, method: str, null: str) -> Path:
        """Sub-directory for one null, given by name; the point prediction sits above it."""
        return self.method_dir(model, sample, target, method) / f"null_{null}"

    def pooled_null_dir(self, model: str, target: str, method: str, null: str) -> Path:
        """Where the across-slide diagnostic goes; a comparison, not the guarantee."""
        return self.results_root(model) / "_pooled" / target / method / f"null_{null}"

    def prepare(self) -> None:
        """Create the workspace root directory."""
        self.root.mkdir(parents=True, exist_ok=True)
