"""What a baselines run is configured with, and where it writes.

The baselines take the same YAML the pipeline run used plus one ``baselines:`` block, written in a
``<config>.baselines.yaml`` sidecar or as a top-level key of the config itself.  Results go into
the run's own ``results/`` tree as ``<prefix><method>`` rows; trained bundles, draw stores and the
cost record go under ``baseline_UQ/<model>/``.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import SUPPORTED_METHODS, Config, PathLike, load_config
from ..workspace import Workspace

#: The baselines, in report order.  ``deep_ensemble`` is the Gaussian-likelihood ensemble of
#: Lakshminarayanan et al.; ``deep_ensemble_nb`` puts a negative binomial on the raw counts.
METHODS = ("raw", "deep_ensemble", "deep_ensemble_nb", "mc_dropout", "resampling")
#: Suffix of the row that feeds a baseline's own mean into cluster-conditional
#: split conformal, so only the source of the null distribution changes.
CONFORMAL_SUFFIX = "_conformal"
#: The classic conformal baseline: the point prediction ranked against held-out
#: calibration data with one pooled residual distribution (UTOPIA without clusters).
CONFORMAL_MARGINAL = "conformal_marginal"
#: Methods whose confidence comes from a spread over members or passes; ``raw``
#: is the floor and has none.
UNCERTAIN_METHODS = ("deep_ensemble", "deep_ensemble_nb", "mc_dropout", "resampling")
#: Methods whose gene p-value is the unadjusted count ``p_i = (1/M) * #{m : yhat_{m,i} <= nu}``
#: and whose interval is the empirical quantiles of the same M draws.
COUNT_PVALUE_METHODS = ("mc_dropout", "resampling")
#: Methods trained against the raw counts rather than their ``[0, 1]`` rescaling,
#: so their output is already on the count scale and every channel weight is 1.
COUNT_SCALE_METHODS = ("deep_ensemble_nb",)
#: Methods that exist for genes alone.  A cell-type target is a count of labelled
#: superpixels, whose distribution the Poisson-binomial tail already states exactly.
GENE_ONLY_METHODS = ("deep_ensemble_nb",)
#: Methods trained at scale 1 alone and read at coarser scales by summing the
#: scale-1 prediction over the block; the other methods train per scale.
AGGREGATED_METHODS = ("raw",)


@dataclass
class BaselineConfig:
    """The ``baselines:`` block of the run's YAML."""

    # Which of METHODS to run.
    methods: list[str] = field(default_factory=lambda: list(METHODS))
    # M, the members of deep_ensemble and deep_ensemble_nb.
    n_members: int = 5
    # M, the bootstrap members of resampling; its count p-value resolves to 1 / M.
    n_resamples: int = 200
    # K, the stochastic forward passes of mc_dropout; its count p-value resolves to 1 / K.
    n_passes: int = 200
    # Cluster counts to report on, among those the run computed; None uses fold generation's.
    n_clusters: list[int] | None = None
    # Leading epochs an ensemble member trains on a mean-only objective before beta-NLL.
    warmup_epochs: int = 5
    # beta-NLL exponent; 0 recovers plain NLL.
    nll_beta: float = 0.5
    # Base seed: member m trains at seed + m, and MC-dropout pass k runs at seed + k.
    seed: int = 2000
    # Dropout probability during mc_dropout's passes; None reuses the rate trained with.
    dropout: float | None = None
    # Units per forward pass when generating draws.
    batch_size: int = 32768
    # Members trained concurrently; None follows the run's calibration.parallel_processes.
    parallel_processes: int | None = None
    # Residual variance is estimated per histology "cluster", or once per target ("marginal").
    residual_variance: str = "cluster"
    # Clusters calibrated by fewer units fall back to the marginal estimate.
    residual_min_units: int = 200
    # Also report each baseline's mean through cluster-conditional split conformal.
    conformalize: bool = False
    # Also report the point prediction through marginal split conformal (no clustering).
    conformal_marginal: bool = True
    # Prefix of every row's directory name; "" is refused where it would overwrite a pipeline row.
    prefix: str = "baseline_"
    # Central coverage levels the interval check reports at.
    coverage: list[float] = field(default_factory=lambda: [0.9, 0.95])
    # Nominal FDR levels the per-cluster breakdown reports at.
    fdr_grid: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2])
    # Write images beside the arrays; None follows the run's own inference.visualize.
    visualize: bool | None = None

    def members(self, method: str) -> int:
        """Draws ``method`` produces: ``n_passes``, ``n_resamples``, ``n_members``, or 1 for raw."""
        if method == "mc_dropout":
            return int(self.n_passes)
        if method == "resampling":
            return int(self.n_resamples)
        return 1 if method == "raw" else int(self.n_members)

    def clusters(self, config: Config) -> list[int]:
        """Cluster counts to condition on, defaulted from the run's own."""
        return list(self.n_clusters) if self.n_clusters else [config.fold_n_clusters]

    def processes(self, config: Config) -> int:
        """Members trained concurrently, defaulted from the run's ``parallel_processes``."""
        if self.parallel_processes is not None:
            return int(self.parallel_processes)
        return int(config.calibration.parallel_processes)

    def rows(self) -> list[str]:
        """Every method name the results tree will carry, in report order.

        The chosen methods come first, then ``conformal_marginal`` (if on and ``raw`` runs), then
        a ``<method>_conformal`` row per uncertainty-bearing method when ``conformalize`` is set.
        """
        chosen = [method for method in METHODS if method in self.methods]
        rows = list(chosen)
        if self.conformal_marginal and "raw" in chosen:
            rows.append(CONFORMAL_MARGINAL)
        if self.conformalize:
            rows += [
                method + CONFORMAL_SUFFIX for method in chosen if method in UNCERTAIN_METHODS
            ]
        return rows

    def label(self, row: str) -> str:
        """The directory one row is written to, inside the shared results tree."""
        return f"{self.prefix}{row}"

    def labels(self) -> list[str]:
        """Every directory name this run will add to the results tree."""
        return [self.label(row) for row in self.rows()]

    def visualizes(self, config: Config) -> bool:
        """Whether to render maps and curves, defaulting to what the run does."""
        if self.visualize is None:
            return bool(config.inference.visualize)
        return bool(self.visualize)


#: Recorded in ``run_meta.json`` and printed at the start of every run that asks
#: for the conformalised rows, because those rows are an approximation.
CONFORMAL_CAVEAT = (
    "The '*_conformal' rows rank a baseline's own mean prediction against the "
    "*pipeline's* held-out fold predictions, which the 'calibrate' stage already "
    "produced: a baseline's members are trained on every training row, so they "
    "have no held-out score of their own without re-running cross-validation M "
    "times over. Both sides come from the same architecture and the same "
    "training set, and the per-cluster batch-effect tolerance absorbs the mean "
    "shift between them, but the exchangeability conformal validity rests on is "
    "not exactly satisfied. The row shows what calibration buys a baseline's "
    "score; it is not itself a validity claim."
)


def _validate(config: Config, baselines: BaselineConfig) -> None:
    """Raise ``ValueError`` if the baselines block cannot be executed as written."""
    unknown = [method for method in baselines.methods if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown baseline method(s) {unknown}; available: {list(METHODS)}")
    if not baselines.methods:
        raise ValueError("'baselines.methods' is empty; remove it to run all of them instead")
    if config.is_external:
        raise ValueError(
            "the baselines train and re-run the network, so they are available for the built-in "
            "model alone, not for an external store"
        )
    if baselines.n_members < 2:
        raise ValueError("'baselines.n_members' must be at least 2")
    if baselines.n_resamples < 2:
        raise ValueError("'baselines.n_resamples' must be at least 2")
    if baselines.n_passes < 2:
        raise ValueError("'baselines.n_passes' must be at least 2")
    if baselines.residual_variance not in ("cluster", "marginal"):
        raise ValueError(
            f"'baselines.residual_variance' is '{baselines.residual_variance}'; available: "
            "['cluster', 'marginal']"
        )
    missing = sorted(set(baselines.clusters(config)) - set(config.n_clusters))
    if missing:
        raise ValueError(
            f"'baselines.n_clusters' names {missing}, which the run did not cluster at; "
            f"available: {list(config.n_clusters)}"
        )
    if any(not 0.0 < level < 1.0 for level in baselines.coverage):
        raise ValueError("'baselines.coverage' levels must lie strictly between 0 and 1")
    if any(character in baselines.prefix for character in "/\\ "):
        raise ValueError(
            f"'baselines.prefix' is {baselines.prefix!r}; it names a directory, so it cannot hold "
            "a path separator or a space"
        )
    # Checked against every supported pipeline method, not only those this run reports:
    # the rows share the pipeline's results tree, which gains rows over time.
    clashes = sorted(set(baselines.labels()) & set(SUPPORTED_METHODS))
    if clashes:
        raise ValueError(
            f"baseline row(s) {clashes} would be written to the same directory as the pipeline "
            f"method(s) of the same name and overwrite them; set 'baselines.prefix' (currently "
            f"{baselines.prefix!r}) to something that keeps them apart"
        )


def load(path: PathLike, baselines_path: PathLike | None = None) -> tuple[Config, BaselineConfig]:
    """Read a run's YAML and its ``baselines:`` block, returning ``(config, baselines)``.

    The block is the config's own top-level key or a separate file (``baselines_path``, by default
    the ``<config>.baselines.yaml`` sidecar), and having both is an error.
    """
    path = Path(path).resolve()
    with open(path) as handle:
        payload: dict[str, Any] = yaml.safe_load(handle) or {}
    inline = payload.pop("baselines", None) or {}

    if baselines_path:
        beside = Path(baselines_path)
    else:
        beside = path.parent / f"{path.stem}.baselines.yaml"
    external: dict[str, Any] = {}
    if beside.exists():
        with open(beside) as handle:
            loaded = yaml.safe_load(handle) or {}
        # A sidecar is a bare mapping of options or carries the same ``baselines:`` heading.
        external = loaded.get("baselines", loaded) or {}
    if inline and external:
        raise ValueError(
            f"a 'baselines:' block appears both in {path} and in {beside}; keep one of them"
        )
    section = inline or external
    known = set(BaselineConfig.__dataclass_fields__)
    unknown = sorted(set(section) - known)
    if unknown:
        raise ValueError(f"unknown key(s) in config section 'baselines': {unknown}")

    if inline:
        # The pipeline's loader rejects unknown keys, so it parses a copy without the block;
        # the copy sits in a temporary directory, so relative paths are re-anchored afterwards.
        with tempfile.TemporaryDirectory() as scratch:
            stripped = Path(scratch) / path.name
            with open(stripped, "w") as handle:
                yaml.safe_dump(payload, handle)
            config = load_config(stripped)
        config.root = path.parent
        config.workspace = (path.parent / str(payload.get("workspace", "workspace"))).resolve()
    else:
        config = load_config(path)

    baselines = BaselineConfig(**section)
    _validate(config, baselines)
    return config, baselines


class BaselineWorkspace(Workspace):
    """The run's workspace, plus the places a baselines run keeps its apparatus.

    Results stay in the shared ``results/`` tree; trained bundles, draw stores and the cost record
    go under ``baseline_UQ/``, so deleting that directory resets the experiment.
    """

    def baselines_root(self, model: str) -> Path:
        """Root of one model's baseline apparatus: ``baseline_UQ/<model>``."""
        return self.root / "baseline_UQ" / model

    def baselines_dir(self, model: str, modality: str) -> Path:
        """Root of one modality's baseline apparatus: models and draws."""
        return self.baselines_root(model) / modality

    def baseline_models(self, model: str, modality: str, method: str, scale: int) -> Path:
        """Bundle file of the trained members backing one baseline at one scale."""
        return self.baselines_dir(model, modality) / "models" / f"{method}_scale{scale}.pt"

    def baseline_draws(self, model: str, modality: str, method: str, scale: int) -> Path:
        """Store directory of one baseline's raw predictions at one scale."""
        return self.baselines_dir(model, modality) / method / f"scale{scale}"

    def baseline_scratch(self, model: str, modality: str, scale: int, kind: str) -> Path:
        """Where one scale's training arrays are materialised while a bundle fits.

        ``kind`` is ``"rescaled"`` for ``[0, 1]`` targets or ``"counts"`` for raw counts; the two
        get separate directories so neither overwrites the other's ``train_y.npy``.
        """
        return self.scratch(model) / "baselines" / f"{modality}_scale{scale}_{kind}"

    def cost(self, model: str) -> Path:
        """Path of the JSON recording the wall-clock time of every fit and every draw pass."""
        return self.baselines_root(model) / "cost.json"
