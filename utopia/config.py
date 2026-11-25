"""Typed configuration loaded from a single YAML file.

``mode`` is ``builtin`` (measurements and an H&E, all stages) or ``external`` (predictions made
by another model). An external run either lists ``samples`` for UTOPIA to import, or names none
and reads calibration and prediction stores written elsewhere.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import yaml

PathLike = Union[str, Path]

#: The one model UTOPIA implements itself; every other model is external. The name is also the
#: directory its calibration, predictions and results are stored under.
BUILTIN_MODEL = "default_model"
#: Side length of a scale-1 superpixel, in pixels of the processed H&E.
SUPERPIXEL = 16
#: Coordinate systems ``positions.npy`` and the ROIs may be written in: pixels of the raw (not
#: yet rescaled) H&E image, or indices on the scale-1 superpixel lattice.
COORDINATE_TYPES = ("raw_pixel", "superpixel")
#: The coordinate system assumed when a configuration does not name one.
DEFAULT_COORDINATE_TYPE = "raw_pixel"
#: Uncertainty methods, in the order results are reported in.
SUPPORTED_METHODS = (
    "none",
    "mc_dropout",
    "ensemble",
    "resampling",
    "conformal_marginal",
    "UTOPIA",
)
#: Methods whose confidence comes from predictive draws; they need the ``sample`` stage.
SAMPLING_METHODS = ("mc_dropout", "ensemble", "resampling")
#: Methods that depend on the histology clustering; reported once per cluster count.
CLUSTER_METHODS = ("UTOPIA",)
#: Method reported when ``inference.methods`` names none.
DEFAULT_METHOD = "UTOPIA"
#: Modalities that can be predicted.
MODALITIES = ("gene", "cell_type")
#: Feature extractors: CATCH window fusion (fast), one pass per tissue superpixel (slow), or
#: one pass per superpixel of the whole lattice (very_slow).
EXTRACTORS = ("fast", "slow", "very_slow")
#: Name each extractor's feature files are stored under.
EXTRACTOR_NAMES = {"fast": "uni_catch", "slow": "uni_slow", "very_slow": "uni_very_slow"}
#: Extractors whose feature files hold every lattice superpixel in row-major order, rather
#: than the tissue superpixels alone.
LATTICE_EXTRACTORS = ("very_slow",)
#: What a run is given: measurements and an H&E, or somebody else's stores.
MODES = ("builtin", "external")
#: Methods an external store can support: the three that need no network.
EXTERNAL_METHODS = ("none", "conformal_marginal", "UTOPIA")
#: Roles a sample of an external run can take: its spots calibrate, or they are tested.
EXTERNAL_ROLES = ("calibration", "test")
#: Shapes a spot of an external model can have on the H&E.
SPOT_SHAPES = ("circle", "square")
#: Name the per-spot UNI features of an external run are stored under.
EXTERNAL_EXTRACTOR = "uni_spot"
#: Suffix added to that name when Harmony corrected the features before clustering.
HARMONY_SUFFIX = "_harmony"
#: Sections an external run may state when it lists ``samples`` (section 1.2 of the data guide).
EXTERNAL_SAMPLE_SECTIONS = (
    "samples",
    "pixel_size_raw",
    "superpixel_size",
    "n_clusters",
    "tissue_mask",
    "features",
    "clustering",
    "spot",
    "he_batch_effect_correction",
)
#: Keys of the ``features`` section an external run reads; the others describe the built-in
#: extractors.
EXTERNAL_FEATURE_KEYS = (
    "model_dir",
    "batch_size",
    "num_workers",
    "device",
    "harmony_iterations",
    "random_state",
)
#: Sections only an external run that lists ``samples`` reads; refused for the built-in model.
EXTERNAL_ONLY_SECTIONS = ("spot", "he_batch_effect_correction")
#: Default drawing grid of an external run, in microns per canvas cell.
EXTERNAL_SUPERPIXEL_SIZE = 4.0
#: Default PCA width before KMeans in an external run.
EXTERNAL_PCA_DIM = 100
#: Config sections that only describe how the built-in model predicts; refused in external mode.
BUILTIN_SECTIONS = (
    "samples",
    "pixel_size_raw",
    "superpixel_size",
    "coordinate_type",
    "scales",
    "n_clusters",
    "predict",
    "predict_genes",
    "benchmark",
    "tissue_mask",
    "features",
    "clustering",
    "default",
    "calibration",
    "sampling",
)
#: Families a Benjamini-Hochberg correction may be taken over. The default corrects within each
#: (sample x cluster x target) group, so adding a slide never changes another slide's calls.
BH_SCOPES = ("sample_cluster", "sample", "cluster", "global")


@dataclass
class RoiSpec:
    """One region of interest, in the units named by ``coordinate_type`` as the user wrote it."""

    top_left: tuple[int, int]  # (row, col) of the top-left corner
    height: int  # number of rows the ROI spans
    width: int  # number of columns the ROI spans
    roles: tuple[str, ...]  # subset of {"train", "calib"}; both when omitted


@dataclass
class SpotSpec:
    """Footprint of one spot of an external model, in microns.

    A circle is given by its ``radius`` and a square by its ``width``. Both are physical sizes,
    so one value serves H&E images of different resolutions; ``crop_width_pixels`` converts the
    footprint to the pixels of one image.
    """

    shape: str = "circle"  # "circle" or "square", see SPOT_SHAPES
    radius: float | None = None  # circle only: radius in microns
    width: float | None = None  # square only: side length in microns

    @property
    def crop_width_um(self) -> float:
        """Side of the square that encloses the spot, in microns."""
        if self.shape == "circle":
            return 2.0 * float(self.radius)
        return float(self.width)

    @property
    def half_width_um(self) -> float:
        """Distance from the spot centre to the edge of its footprint, in microns."""
        return self.crop_width_um / 2.0

    def crop_width_pixels(self, pixel_size_raw: float) -> float:
        """Side of the square that encloses the spot, in pixels of a raw H&E.

        ``pixel_size_raw`` is the size of one pixel of that H&E, in microns.
        """
        return self.crop_width_um / float(pixel_size_raw)


@dataclass
class SampleSpec:
    """One sample: an H&E image plus its ROIs (built-in) or its role (external)."""

    name: str
    # H&E file name inside the sample's input directory. None is allowed for an external
    # sample only: its maps are then drawn without a tissue mask.
    he: str | None
    rois: list[RoiSpec] = field(default_factory=list)
    pixel_size_raw: float | None = None  # microns per pixel of this H&E; None = the global value
    role: str | None = None  # external only: "calibration" or "test", see EXTERNAL_ROLES
    # External only: the ``spot`` keys this sample states itself, which override the global ones.
    spot: dict[str, Any] = field(default_factory=dict)


#: Keys a ``samples`` entry may carry; anything else is rejected at load time.
SAMPLE_KEYS = ("name", "he", "rois", "pixel_size_raw")
#: Keys a ``samples`` entry of an external run may carry.
EXTERNAL_SAMPLE_KEYS = ("name", "role", "he", "pixel_size_raw", "spot")
#: Keys the ``spot`` of one external sample may carry, overriding the global ``spot`` section.
SAMPLE_SPOT_KEYS = ("radius", "width")


@dataclass
class TissueMaskConfig:
    """HistoSweep tissue-masking parameters."""

    density_thresh: float = 100.0  # colour-histogram count below which a superpixel is background
    min_size: int = 50  # smallest tissue island kept, in superpixels
    clean_background: bool = True  # whether to remove islands smaller than min_size
    glcm_levels: int = 64  # grey levels used by the texture refinement


@dataclass
class FeatureConfig:
    """UNI feature extraction settings."""

    # "fast": CATCH window fusion; "slow": one pass per tissue superpixel; "very_slow": one pass
    # per superpixel of the whole lattice.
    method: str = "fast"
    # Correct slide-to-slide batch effects; "auto" = out-of-sample runs (more than one sample).
    harmony: bool | str = "auto"
    harmony_iterations: int = 20  # maximum Harmony iterations
    model_dir: str = ""  # directory holding the UNI weights, pytorch_model.bin
    batch_size: int = 64
    num_workers: int = 8
    device: str = "cuda"
    pca: int | None = 100  # width kept per stream; None keeps the raw 1024 + 1024 dimensions
    # Fit the PCA on every n-th tissue window (fast) or tissue superpixel (slow, very_slow);
    # None = every window (fast) or every 16th superpixel.
    fit_patch_stride: int | None = None
    fit_tokens_per_patch: int = 49  # fast only: local tokens drawn per window for the PCA (1-196)
    tissue_windows: bool = True  # fast only: encode just the windows that overlap the tissue mask
    random_state: int = 42


@dataclass
class ClusteringConfig:
    """Histology clustering (PCA + KMeans over the extracted features)."""

    source: str | None = None  # name the cluster files are stored under; None = extractor name
    pca_dim: int = 80  # PCA width fitted per scale before KMeans
    random_state: int = 42


@dataclass
class DefaultConfig:
    """Hyper-parameters of the built-in feed-forward predictor."""

    epochs: int = 20
    batch_size: int = 8096
    learning_rate: float = 1e-3
    dropout: float = 0.1
    loss: str = "amse"  # "amse" (asymmetric, under-prediction weighted by beta) or "rmse"
    beta: float = 2.0  # weight on the squared error where the prediction is below the target
    seed: int = 42
    predict_batch_size: int = 32768  # units per forward pass when predicting


@dataclass
class CalibrationConfig:
    """Spatial cross-validation used to build calibration predictions."""

    # Fold side lengths to try, coarsest to finest, in superpixels.
    fold_widths: list[int] = field(default_factory=lambda: [200, 100, 50])
    # Guard band in superpixels between a fold's held-out centre and the training data around it.
    safe_boundary: int = 16
    # Cluster count fold generation checks coverage with; None = n_clusters[0].
    fold_n_clusters: int | None = None
    parallel_processes: int = 8  # folds (or draw models) trained concurrently; 1 trains in-process


@dataclass
class SamplingConfig:
    """Draw-based uncertainty: how many draws, and how they are produced."""

    n_draws: int = 100  # predictive draws K; the count p-value has resolution 1 / K
    coverage: float = 0.95  # central coverage of the reported Gaussian predictive interval
    dropout: float | None = None  # mc_dropout rate; None reuses the training rate
    seed: int = 1000  # base seed; draw k uses seed + k
    batch_size: int = 32768  # units per forward pass when generating a draw


@dataclass
class InferenceConfig:
    """Conformal inference and reporting options.

    Each batch-effect setting takes one value, or a ``{scale: value}`` mapping naming every scale.
    """

    methods: list[str] | None = None  # uncertainty methods to report; None = UTOPIA alone
    bh_scope: str = "sample_cluster"  # Benjamini-Hochberg family, one of BH_SCOPES
    # Estimate a per-cluster prediction shift and absorb it before ranking.
    correct_batch_effect: bool | dict[Any, bool] = True
    # Quantile grid the shift is measured on, or a fixed number used as the tolerance.
    batch_effect_magnitude: str | float | dict[Any, str | float] = "medium"
    # How the per-quantile shifts become one tolerance: "max" or "mean".
    batch_effect_aggregate: str | dict[Any, str] = "max"
    # Estimate a shift only in clusters where more than this fraction of calibration units is null.
    batch_effect_null_fraction: float | dict[Any, float] = 0.1
    # Calibration units a cluster needs to be tested; None = max(200, 5% of the cluster).
    min_calib_units: int | None = None
    p_min: float = 0.01  # adjusted p-value shown as a confidence of 1
    p_max: float = 0.5  # adjusted p-value shown as a confidence of 0
    visualize: bool = True  # whether to render maps and FDP curves

    #: Settings that may be stated once or once per scale.
    PER_SCALE = (
        "correct_batch_effect",
        "batch_effect_magnitude",
        "batch_effect_aggregate",
        "batch_effect_null_fraction",
    )

    def at_scale(self, name: str, scale: int) -> Any:
        """Value of a :data:`PER_SCALE` setting at ``scale``, whether stated once or per scale."""
        value = getattr(self, name)
        if not isinstance(value, dict):
            return value
        for key in (scale, str(scale)):
            if key in value:
                return value[key]
        raise KeyError(f"inference.{name} names no value for scale {scale}")


@dataclass
class Config:
    """The complete run configuration."""

    root: Path  # directory of the config file; relative paths resolve against it
    workspace: Path  # directory holding every generated artifact
    # Pixel size of the input H&E, microns. A sample's own pixel_size_raw takes precedence;
    # None is allowed when every sample states its own.
    pixel_size_raw: float | None = 0.5
    superpixel_size: float = 8.0  # physical side of one scale-1 superpixel, microns
    # Units of positions.npy and of the ROIs: "raw_pixel" or "superpixel", see COORDINATE_TYPES.
    coordinate_type: str = DEFAULT_COORDINATE_TYPE
    samples: list[SampleSpec] = field(default_factory=list)
    # Superpixel scales; a unit at scale s is s x s superpixels, so scale 1 = superpixel_size.
    scales: list[int] = field(default_factory=lambda: [1])
    n_clusters: list[int] = field(default_factory=lambda: [15])  # histology cluster counts
    predict: list[str] = field(default_factory=lambda: ["gene"])  # modalities, see MODALITIES
    benchmark: bool = False  # ground truth exists for the test area (enables FDP diagnostics)
    mode: str = "builtin"  # "builtin" or "external", see MODES
    model: str = BUILTIN_MODEL  # "default_model", or the external stores' directory name
    targets: str = "targets.yaml"  # inference target list, relative to root
    # Genes ``calibrate`` predicts and stores: a list, a JSON file of names, or None for all.
    predict_genes: str | list[str] | None = None
    tissue_mask: TissueMaskConfig = field(default_factory=TissueMaskConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    default: DefaultConfig = field(default_factory=DefaultConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    # External only: the spot footprint shared by every sample that does not state its own.
    spot: SpotSpec | None = None
    # External only: Harmony-correct the spot features across samples before clustering.
    he_batch_effect_correction: bool = False

    # -- derived ------------------------------------------------------------ #
    @property
    def pixel_size(self) -> float:
        """Pixel size the H&E is rescaled to, in microns: ``superpixel_size / SUPERPIXEL``."""
        return self.superpixel_size / SUPERPIXEL

    def sample_pixel_size_raw(self, sample: str) -> float:
        """Microns per pixel of the named sample's raw H&E; its own value overrides the global."""
        spec = self.sample(sample)
        if spec.pixel_size_raw is not None:
            return spec.pixel_size_raw
        if self.pixel_size_raw is not None:
            return self.pixel_size_raw
        raise ValueError(
            f"sample '{sample}' has no pixel_size_raw: set a global 'pixel_size_raw' or a "
            f"'pixel_size_raw' in the sample's entry under 'samples'"
        )

    def rescale_factor(self, sample: str) -> float:
        """Factor the named sample's raw H&E is resized by; above 1 the image is up-scaled."""
        return self.sample_pixel_size_raw(sample) / self.pixel_size

    def sample(self, name: str) -> SampleSpec:
        """The configured sample called ``name``."""
        for spec in self.samples:
            if spec.name == name:
                return spec
        raise KeyError(f"no sample named '{name}' in the config; available: {self.sample_names}")

    @property
    def is_external(self) -> bool:
        """Whether the predictions were made outside UTOPIA."""
        return self.mode == "external"

    @property
    def imports_samples(self) -> bool:
        """Whether this is an external run that lists sample folders for UTOPIA to import."""
        return self.is_external and bool(self.samples)

    def samples_with_role(self, role: str) -> list[SampleSpec]:
        """The external samples whose role is ``role``, in configuration order."""
        return [sample for sample in self.samples if sample.role == role]

    def sample_spot(self, name: str) -> SpotSpec:
        """Spot footprint of the named external sample; its own ``spot`` keys take precedence."""
        merged: dict[str, Any] = {}
        if self.spot is not None:
            merged = {"shape": self.spot.shape, "radius": self.spot.radius, "width": self.spot.width}
        merged.update(self.sample(name).spot)
        return SpotSpec(**merged)

    @property
    def external_extractor(self) -> str:
        """Name the clustering of an external run is stored under, carrying how it was made."""
        if self.he_batch_effect_correction:
            return EXTERNAL_EXTRACTOR + HARMONY_SUFFIX
        return EXTERNAL_EXTRACTOR

    @property
    def sample_names(self) -> list[str]:
        """Sample names in configuration order, which fixes sample indices."""
        return [sample.name for sample in self.samples]

    @property
    def methods(self) -> list[str]:
        """Uncertainty methods to report, in :data:`SUPPORTED_METHODS` order."""
        chosen = self.inference.methods or [DEFAULT_METHOD]
        return [method for method in SUPPORTED_METHODS if method in chosen]

    @property
    def sampling_methods(self) -> list[str]:
        """Configured methods that need draw models built by the ``sample`` stage."""
        return [method for method in self.methods if method in SAMPLING_METHODS]

    @property
    def regime(self) -> str:
        """``"out_of_sample"`` when the run has more than one sample, else ``"in_sample"``.

        The ROI roles do not matter: on a single sample, an ROI kept for calibration alone is
        still a part of the slide the model was trained on.
        """
        return "out_of_sample" if len(self.samples) > 1 else "in_sample"

    @property
    def harmony_enabled(self) -> bool:
        """Whether Harmony corrects the features; needs at least two samples."""
        requested = self.features.harmony
        if requested == "auto":
            requested = self.regime == "out_of_sample"
        return bool(requested) and len(self.samples) > 1

    @property
    def extractor(self) -> str:
        """Name the features are stored under, carrying how they were made."""
        name = EXTRACTOR_NAMES[self.features.method]
        return f"{name}_harmony" if self.harmony_enabled else name

    @property
    def features_cover_lattice(self) -> bool:
        """Whether the feature files hold every lattice superpixel instead of tissue only."""
        return self.features.method in LATTICE_EXTRACTORS

    @property
    def cluster_source(self) -> str:
        """Name the cluster files are stored under: ``clustering.source`` or the extractor name."""
        return self.clustering.source or self.extractor

    @property
    def fold_n_clusters(self) -> int:
        """Cluster count used by fold generation to check cluster coverage."""
        return self.calibration.fold_n_clusters or self.n_clusters[0]

    def sample_dir(self, name: str) -> Path:
        """Input directory of the named sample."""
        return self.root / "samples" / name

    @property
    def targets_path(self) -> Path:
        """Resolved path of the inference target list."""
        return self.root / self.targets

    def predicted_genes(self) -> list[str] | None:
        """Return the gene names ``calibrate`` should store, or None for all of them.

        A string ``predict_genes`` is read as a JSON list file relative to the config directory.
        """
        requested = self.predict_genes
        if requested is None:
            return None
        if isinstance(requested, str):
            import json

            with open(self.root / requested) as handle:
                requested = json.load(handle)
            if not isinstance(requested, list):
                raise ValueError(
                    f"'predict_genes' file {self.root / self.predict_genes} must hold a JSON list "
                    "of gene names"
                )
        return [str(name) for name in requested]


def _build_dataclass(cls: Any, payload: dict[str, Any] | None, section: str) -> Any:
    """Instantiate a config dataclass from a YAML mapping, rejecting unknown keys."""
    payload = dict(payload or {})
    unknown = set(payload) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown key(s) in config section '{section}': {sorted(unknown)}")
    return cls(**payload)


def _optional_float(value: Any) -> float | None:
    """Convert a YAML number to a float, keeping None for a key that was left out."""
    if value is None:
        return None
    return float(value)


def _build_samples(payload: Sequence[dict[str, Any]]) -> list[SampleSpec]:
    """Turn the ``samples`` YAML list into :class:`SampleSpec` objects."""
    samples = []
    for entry in payload:
        unknown = sorted(set(entry) - set(SAMPLE_KEYS))
        if unknown:
            raise ValueError(
                f"unknown key(s) {unknown} in the samples entry {entry.get('name', '?')!r}; "
                f"available: {list(SAMPLE_KEYS)}"
            )
        rois = [
            RoiSpec(
                top_left=(int(roi["top_left"][0]), int(roi["top_left"][1])),
                height=int(roi["height"]),
                width=int(roi["width"]),
                roles=tuple(roi.get("roles", ["train", "calib"])),
            )
            for roi in entry.get("rois", [])
        ]
        name = str(entry["name"])
        he = str(entry.get("he", "he_raw.png"))
        pixel_size_raw = _optional_float(entry.get("pixel_size_raw"))
        samples.append(SampleSpec(name=name, he=he, rois=rois, pixel_size_raw=pixel_size_raw))
    return samples


def _build_external_samples(payload: Sequence[dict[str, Any]]) -> list[SampleSpec]:
    """Turn the ``samples`` YAML list of an external run into :class:`SampleSpec` objects."""
    samples = []
    for entry in payload:
        unknown = sorted(set(entry) - set(EXTERNAL_SAMPLE_KEYS))
        if unknown:
            raise ValueError(
                f"unknown key(s) {unknown} in the samples entry {entry.get('name', '?')!r} of an "
                f"external run; available: {list(EXTERNAL_SAMPLE_KEYS)}"
            )
        name = str(entry["name"])

        he = entry.get("he")
        if he is not None:
            he = str(he)

        pixel_size_raw = _optional_float(entry.get("pixel_size_raw"))

        role = entry.get("role")
        if role is not None:
            role = str(role)

        # The shape is shared by the whole run; a sample may only state its own size.
        spot = dict(entry.get("spot") or {})
        unknown_spot = sorted(set(spot) - set(SAMPLE_SPOT_KEYS))
        if unknown_spot:
            raise ValueError(
                f"unknown key(s) {unknown_spot} in the 'spot' of sample '{name}'; a sample may "
                f"state {list(SAMPLE_SPOT_KEYS)}, and the shape is set once in the top-level 'spot'"
            )

        samples.append(
            SampleSpec(name=name, he=he, pixel_size_raw=pixel_size_raw, role=role, spot=spot)
        )
    return samples


def _load_external_samples(
    payload: dict[str, Any], root: Path, workspace: Path, inference: InferenceConfig
) -> Config:
    """Build the config of an external run that lists sample folders to import."""
    refused = [
        key
        for key in BUILTIN_SECTIONS
        if key in payload and key not in EXTERNAL_SAMPLE_SECTIONS
    ]
    if refused:
        raise ValueError(
            f"config section(s) {refused} describe how the built-in model predicts, which an "
            f"external run does not do; remove them (see docs/DATA_STRUCTURE.md, section 1.3)"
        )

    features_payload = dict(payload.get("features") or {})
    unused = sorted(set(features_payload) - set(EXTERNAL_FEATURE_KEYS))
    if unused:
        raise ValueError(
            f"'features' key(s) {unused} belong to the built-in extractors; an external run reads "
            f"{list(EXTERNAL_FEATURE_KEYS)} only, and corrects batch effects with the top-level "
            "'he_batch_effect_correction'"
        )

    # An external run clusters on PCA-100 of the spot features unless told otherwise.
    clustering_payload = dict(payload.get("clustering") or {})
    clustering_payload.setdefault("pca_dim", EXTERNAL_PCA_DIM)

    correction = payload.get("he_batch_effect_correction", False)
    if not isinstance(correction, bool):
        raise ValueError("'he_batch_effect_correction' must be true or false")

    spot = None
    if payload.get("spot") is not None:
        spot = _build_dataclass(SpotSpec, payload["spot"], "spot")

    return Config(
        root=root,
        workspace=workspace,
        pixel_size_raw=_optional_float(payload.get("pixel_size_raw")),
        superpixel_size=float(payload.get("superpixel_size", EXTERNAL_SUPERPIXEL_SIZE)),
        samples=_build_external_samples(payload["samples"]),
        scales=[1],
        n_clusters=[int(count) for count in payload.get("n_clusters", [15])],
        mode="external",
        model=str(payload["model"]),
        targets=str(payload.get("targets", "targets.yaml")),
        tissue_mask=_build_dataclass(TissueMaskConfig, payload.get("tissue_mask"), "tissue_mask"),
        features=_build_dataclass(FeatureConfig, features_payload, "features"),
        clustering=_build_dataclass(ClusteringConfig, clustering_payload, "clustering"),
        inference=inference,
        spot=spot,
        he_batch_effect_correction=correction,
    )


def load_config(path: PathLike) -> Config:
    """Read, resolve and validate the YAML configuration at ``path``."""
    path = Path(path).resolve()
    with open(path) as handle:
        payload = yaml.safe_load(handle) or {}

    _check_top_level(payload, path)
    root = path.parent
    mode = str(payload.get("mode", "builtin"))
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}'; available: {list(MODES)}")
    workspace = (root / payload.get("workspace", "workspace")).resolve()
    inference = _build_dataclass(InferenceConfig, payload.get("inference"), "inference")

    if mode == "external":
        if "model" not in payload:
            raise ValueError(
                "external mode needs 'model': the name the external model's stores and results "
                "are kept under"
            )
        if payload.get("samples"):
            # The user lists sample folders; UTOPIA imports them into the two stores.
            config = _load_external_samples(payload, root, workspace, inference)
            validate_config(config)
            return config

        # No samples: the two stores were written elsewhere, so every section describing how
        # predictions, features or clusters are made is refused by name.
        present = [
            key for key in BUILTIN_SECTIONS + EXTERNAL_ONLY_SECTIONS if key in payload
        ]
        if present:
            raise ValueError(
                f"config section(s) {present} need 'samples': an external run without samples "
                f"only reads stores written elsewhere, so remove them or list the samples to "
                f"import (see docs/DATA_STRUCTURE.md, section 1.2)"
            )
        config = Config(
            root=root,
            workspace=workspace,
            mode=mode,
            model=str(payload["model"]),
            targets=str(payload.get("targets", "targets.yaml")),
            scales=[1],
            inference=inference,
        )
        validate_config(config)
        return config

    external_only = [key for key in EXTERNAL_ONLY_SECTIONS if key in payload]
    if external_only:
        raise ValueError(
            f"config section(s) {external_only} are read in mode 'external' only; the built-in "
            "model corrects H&E batch effects with 'features.harmony'"
        )

    # The global pixel_size_raw may be left out when every sample states its own.
    config = Config(
        root=root,
        workspace=workspace,
        pixel_size_raw=_optional_float(payload.get("pixel_size_raw")),
        superpixel_size=float(payload.get("superpixel_size", 8.0)),
        coordinate_type=str(payload.get("coordinate_type", DEFAULT_COORDINATE_TYPE)),
        samples=_build_samples(payload["samples"]),
        scales=[int(scale) for scale in payload.get("scales", [1])],
        n_clusters=[int(count) for count in payload.get("n_clusters", [15])],
        predict=[str(modality) for modality in payload.get("predict", ["gene"])],
        benchmark=bool(payload.get("benchmark", False)),
        mode=mode,
        model=str(payload.get("model", BUILTIN_MODEL)),
        targets=str(payload.get("targets", "targets.yaml")),
        predict_genes=payload.get("predict_genes"),
        tissue_mask=_build_dataclass(TissueMaskConfig, payload.get("tissue_mask"), "tissue_mask"),
        features=_build_dataclass(FeatureConfig, payload.get("features"), "features"),
        clustering=_build_dataclass(ClusteringConfig, payload.get("clustering"), "clustering"),
        default=_build_dataclass(DefaultConfig, payload.get("default"), "default"),
        calibration=_build_dataclass(CalibrationConfig, payload.get("calibration"), "calibration"),
        sampling=_build_dataclass(SamplingConfig, payload.get("sampling"), "sampling"),
        inference=inference,
    )
    validate_config(config)
    _warn_if_upscaling(config)
    return config


def _warn_if_upscaling(config: Config) -> None:
    """Warn when the requested superpixel size makes ``preprocess`` enlarge an H&E.

    Samples that share a raw pixel size share one warning.
    """
    tolerance = 1e-9
    upscaled: dict[float, list[str]] = {}
    for sample in config.samples:
        pixel_size_raw = config.sample_pixel_size_raw(sample.name)
        if config.pixel_size < pixel_size_raw - tolerance:
            upscaled.setdefault(pixel_size_raw, []).append(sample.name)

    for pixel_size_raw, names in upscaled.items():
        smallest = pixel_size_raw * SUPERPIXEL
        factor = pixel_size_raw / config.pixel_size
        warnings.warn(
            f"superpixel_size={config.superpixel_size:g} um needs an H&E of "
            f"{config.pixel_size:g} um/pixel (superpixel_size / {SUPERPIXEL}), finer than "
            f"pixel_size_raw={pixel_size_raw:g} um/pixel of sample(s) {names}. The image will be "
            f"up-scaled by {factor:.3g}x, which adds no real detail. "
            f"Use superpixel_size >= {smallest:g} to avoid up-scaling.",
            stacklevel=3,
        )


#: Every key the top level of a config may carry; anything else is rejected at load time.
TOP_LEVEL_KEYS = frozenset(BUILTIN_SECTIONS) | frozenset(EXTERNAL_ONLY_SECTIONS) | {
    "workspace",
    "mode",
    "model",
    "targets",
    "inference",
}


def _check_top_level(payload: dict[str, Any], path: Path) -> None:
    """Raise if the config's top level carries a key UTOPIA does not read."""
    unknown = sorted(set(payload) - TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(
            f"unknown top-level key(s) {unknown} in {path}; available: "
            f"{sorted(TOP_LEVEL_KEYS)}"
        )


def _validate_per_scale(config: Config) -> None:
    """Check that every per-scale batch-effect setting names exactly the reported scales."""
    inference = config.inference
    for name in InferenceConfig.PER_SCALE:
        value = getattr(inference, name)
        if not isinstance(value, dict):
            continue
        named = {int(key) for key in value}
        missing = sorted(set(config.scales) - named)
        spare = sorted(named - set(config.scales))
        if missing or spare:
            raise ValueError(
                f"inference.{name} is stated per scale but names {sorted(named)} while the run "
                f"reports at {list(config.scales)}"
                + (f"; missing {missing}" if missing else "")
                + (f"; unused {spare}" if spare else "")
            )
    for scale in config.scales:
        aggregate = inference.at_scale("batch_effect_aggregate", scale)
        if aggregate not in ("max", "mean"):
            raise ValueError(
                f"inference.batch_effect_aggregate at scale {scale} is '{aggregate}'; "
                "available: ['max', 'mean']"
            )
        fraction = float(inference.at_scale("batch_effect_null_fraction", scale))
        if not 0.0 <= fraction < 1.0:
            raise ValueError(
                f"inference.batch_effect_null_fraction at scale {scale} is {fraction}; it is a "
                "fraction of the calibration units and must lie in [0, 1)"
            )


def _validate_pixel_sizes(config: Config) -> None:
    """Check that every sample ends up with a positive ``pixel_size_raw``."""
    if config.pixel_size_raw is not None and config.pixel_size_raw <= 0:
        raise ValueError("'pixel_size_raw' must be a positive number of microns per pixel")
    for sample in config.samples:
        if sample.pixel_size_raw is not None and sample.pixel_size_raw <= 0:
            raise ValueError(
                f"sample '{sample.name}': 'pixel_size_raw' must be a positive number of microns "
                "per pixel"
            )
    missing_pixel_size = [sample.name for sample in config.samples if sample.pixel_size_raw is None]
    if config.pixel_size_raw is None and missing_pixel_size:
        raise ValueError(
            f"sample(s) {missing_pixel_size} have no pixel_size_raw: set a global "
            "'pixel_size_raw', or a 'pixel_size_raw' in each of their entries under 'samples'"
        )


def _validate_spot(config: Config, sample: str) -> None:
    """Check the resolved spot footprint of one external sample."""
    spot = config.sample_spot(sample)
    if spot.shape not in SPOT_SHAPES:
        raise ValueError(
            f"sample '{sample}': unknown spot shape '{spot.shape}'; available: {list(SPOT_SHAPES)}"
        )
    if spot.shape == "circle":
        needed, value, other = "radius", spot.radius, spot.width
    else:
        needed, value, other = "width", spot.width, spot.radius
    if value is None:
        raise ValueError(
            f"sample '{sample}': a {spot.shape} spot needs 'spot.{needed}', in microns; state "
            "it in the top-level 'spot' section or in the sample's own 'spot'"
        )
    if other is not None:
        unused = "width" if needed == "radius" else "radius"
        raise ValueError(
            f"sample '{sample}': a {spot.shape} spot is sized by 'spot.{needed}', so "
            f"'spot.{unused}' must be left out"
        )
    if float(value) <= 0:
        raise ValueError(f"sample '{sample}': 'spot.{needed}' must be a positive number of microns")


def _validate_samples_and_grid(config: Config) -> None:
    """Check what both modes read: sample names, pixel sizes, superpixel size and cluster counts."""
    names = config.sample_names
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate sample names: {names}")
    _validate_pixel_sizes(config)
    if config.superpixel_size <= 0:
        raise ValueError("'superpixel_size' must be a positive number of microns")
    if any(count < 2 for count in config.n_clusters):
        raise ValueError("'n_clusters' values must be at least 2")


def _validate_external_samples(config: Config) -> None:
    """Check the sections only an external run with sample folders reads."""
    for sample in config.samples:
        if sample.role not in EXTERNAL_ROLES:
            raise ValueError(
                f"sample '{sample.name}': 'role' is {sample.role!r}; every sample of an external "
                f"run must state one of {list(EXTERNAL_ROLES)}"
            )
    for role in EXTERNAL_ROLES:
        if not config.samples_with_role(role):
            raise ValueError(f"an external run needs at least one sample with role '{role}'")

    _validate_samples_and_grid(config)
    if config.clustering.pca_dim < 1:
        raise ValueError("'clustering.pca_dim' must be a positive number of dimensions")
    for sample in config.samples:
        _validate_spot(config, sample.name)


def validate_config(config: Config) -> None:
    """Raise ``ValueError`` if the configuration cannot be executed as written."""
    requested = config.inference.methods or []
    unsupported = [method for method in requested if method not in SUPPORTED_METHODS]
    if unsupported:
        raise ValueError(
            f"unknown uncertainty method(s) {unsupported}; available: {list(SUPPORTED_METHODS)}"
        )
    if config.inference.methods is not None and not config.inference.methods:
        raise ValueError("'inference.methods' is empty; remove it to use the default instead")
    if config.inference.bh_scope not in BH_SCOPES:
        raise ValueError(
            f"unknown inference.bh_scope '{config.inference.bh_scope}'; available: {list(BH_SCOPES)}"
        )
    _validate_per_scale(config)

    if config.is_external:
        # The sampling methods re-run or retrain the network, which a store of finished
        # predictions cannot support.
        needs_network = [method for method in requested if method not in EXTERNAL_METHODS]
        if needs_network:
            raise ValueError(
                f"method(s) {needs_network} re-run the network and so are available for the "
                f"built-in '{BUILTIN_MODEL}' model only, not the external '{config.model}'"
            )
        if config.model == BUILTIN_MODEL:
            raise ValueError(
                f"'{BUILTIN_MODEL}' names the built-in model; give an external run's stores their "
                "own model name"
            )
        if config.imports_samples:
            _validate_external_samples(config)
        return

    if config.model != BUILTIN_MODEL:
        raise ValueError(
            f"model '{config.model}' is not built in; predictions made outside UTOPIA enter as a "
            f"store with mode 'external' (docs/DATA_STRUCTURE.md, section 1.2), and the built-in model is "
            f"'{BUILTIN_MODEL}'"
        )
    _validate_samples_and_grid(config)
    if config.coordinate_type not in COORDINATE_TYPES:
        raise ValueError(
            f"unknown coordinate_type '{config.coordinate_type}'; available: "
            f"{list(COORDINATE_TYPES)}"
        )
    if config.sampling.n_draws < 2:
        raise ValueError("'sampling.n_draws' must be at least 2")
    if not 0.0 < config.sampling.coverage < 1.0:
        raise ValueError("'sampling.coverage' must lie strictly between 0 and 1")
    unknown_modalities = [item for item in config.predict if item not in MODALITIES]
    if unknown_modalities:
        raise ValueError(f"unknown predict modality {unknown_modalities}; available: {list(MODALITIES)}")
    if not config.predict:
        raise ValueError("'predict' must list at least one modality")
    if any(scale < 1 for scale in config.scales):
        raise ValueError("'scales' must be positive integers")
    if config.fold_n_clusters not in config.n_clusters:
        raise ValueError(
            f"calibration.fold_n_clusters={config.fold_n_clusters} must be one of n_clusters="
            f"{config.n_clusters}, so that fold generation reuses a clustering that is computed"
        )
    if not config.samples:
        raise ValueError("'samples' must list at least one sample")

    features = config.features
    if features.method not in EXTRACTORS:
        raise ValueError(
            f"features.method '{features.method}' is unknown; available: {list(EXTRACTORS)}"
        )
    if features.harmony not in (True, False, "auto"):
        raise ValueError("features.harmony must be true, false or 'auto'")
    if not isinstance(features.tissue_windows, bool):
        raise ValueError("features.tissue_windows must be true or false")
    if features.fit_patch_stride is not None and (
        not isinstance(features.fit_patch_stride, int) or features.fit_patch_stride < 1
    ):
        raise ValueError("features.fit_patch_stride must be a positive integer, or null for the default")
    tokens = features.fit_tokens_per_patch
    if not isinstance(tokens, int) or not 1 <= tokens <= 196:
        raise ValueError("features.fit_tokens_per_patch must be an integer from 1 to 196")
    if config.regime == "out_of_sample":
        if features.method != "fast":
            raise ValueError(
                "out-of-sample runs (more than one sample) must use features.method 'fast'; "
                "the slow and very_slow extractors are offered for in-sample runs only"
            )
        if features.harmony is False:
            raise ValueError(
                "out-of-sample runs must correct batch effects; leave features.harmony at "
                "'auto' or set it to true"
            )
    if features.harmony is True and not config.harmony_enabled:
        raise ValueError("features.harmony needs at least two samples to have batches to correct")
    if config.harmony_enabled and features.method != "fast":
        raise ValueError("Harmony corrects patch-level embeddings, which only the fast extractor has")

    all_roles = set()
    for sample in config.samples:
        roles = {role for roi in sample.rois for role in roi.roles}
        unknown_roles = roles - {"train", "calib"}
        if unknown_roles:
            raise ValueError(f"sample '{sample.name}': unknown ROI role(s) {sorted(unknown_roles)}")
        all_roles.update(roles)
    if "train" not in all_roles:
        raise ValueError("at least one ROI must have the 'train' role")
    if "calib" not in all_roles:
        raise ValueError("at least one ROI must have the 'calib' role")
