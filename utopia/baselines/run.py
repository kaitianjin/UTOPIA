"""Driver of the standard UQ baselines that UTOPIA's own method is compared against.

``features``, ``cluster`` and ``calibrate`` must have run with ground truth over the test area.
Each row is written to ``results/<model>/<sample>/<target>/<prefix><method>/`` with a pipeline
method's files; bundles and draws go under ``baseline_UQ/``.
"""

from __future__ import annotations

import copy
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from statistics import NormalDist

import numpy as np

from ..calibration.draws import DrawTarget
from ..calibration.run import _predicted_channels
from ..config import Config
from ..inference import (
    Clustering,
    ScaleGrid,
    Store,
    Target,
    TargetPlan,
    _build_grid,
    _build_plan,
    _canvases,
    _cluster_result,
    _marginal_result,
    _overlay,
    _render,
    _sample_views,
    _write_pooled_curve,
    _write_view_result,
    add_off_tissue,
    load_targets,
    open_stores,
)
from ..inference import _write_target_arrays as _pipeline_target_arrays
from ..inference import _write_units as _pipeline_units
from ..io import TargetStore, read_json, write_json
from ..uncertainty import (
    UncertaintyResult,
    bh_adjust_grouped,
    pvals_to_confidence,
)
from . import coarse
from . import draws as draw_stage
from . import train as train_stage
from .config import (
    AGGREGATED_METHODS,
    CONFORMAL_CAVEAT,
    CONFORMAL_MARGINAL,
    CONFORMAL_SUFFIX,
    COUNT_PVALUE_METHODS,
    COUNT_SCALE_METHODS,
    GENE_ONLY_METHODS,
    METHODS,
    UNCERTAIN_METHODS,
    BaselineConfig,
    BaselineWorkspace,
)
from .stats import (
    block_layout,
    block_sums,
    coverage_and_width,
    draw_interval,
    gaussian_pvals,
    mixture,
    nb_interval,
    nb_pvals,
    pad_blocks,
    poisson_binomial_tail,
    residual_variance,
    sort_draws,
    sun_cai_adjust,
)


@dataclass
class MethodSource:
    """Where one reported row's numbers come from."""

    name: str  # the row's name, which is what the logic branches on
    label: str  # the directory the row is written to: ``baselines.prefix`` + name
    base: str  # the baseline whose draw stores the row reads
    # None to state the model's own predictive tail; "cluster" or "marginal" to rank the
    # baseline's mean with conformal instead, within histology clusters or pooled.
    conformal: str | None
    # Scale -> that scale's draw store; rows trained at scale 1 only hold ``{1: store}``.
    stores: dict[int, TargetStore]
    heteroscedastic: bool  # whether a store's stack is means followed by variances
    # "gaussian" (rescaled target), "negative_binomial" (raw counts), or None for a point
    # prediction; it decides the tail and the interval.
    distribution: str | None = None

    @property
    def counts(self) -> bool:
        """Whether this row's members are on the raw count scale already."""
        return self.distribution == "negative_binomial"

    def store_scale(self, scale: int) -> int:
        """The scale of the store this row reads when reporting at ``scale``.

        That is ``scale`` itself when the row has a store there, and its smallest scale (scale 1)
        otherwise.
        """
        return scale if scale in self.stores else min(self.stores)


def _bh_groups(
    config: Config, grid: ScaleGrid, clustering: Clustering | None
) -> np.ndarray | None:
    """The family each test unit's multiplicity correction is taken within.

    One group id per test unit following the run's own ``bh_scope``, or None for one global family.
    """
    scope = config.inference.bh_scope
    if scope == "global":
        return None
    sample = np.asarray(grid.test.sample_index, dtype=np.int64)
    if scope == "sample" or clustering is None:
        return sample
    cluster = np.asarray(clustering.test, dtype=np.int64)
    if scope == "cluster":
        return cluster
    return sample * (int(cluster.max()) + 1) + cluster


def _labels(config: Config, source: MethodSource, grid: ScaleGrid) -> list[str | None]:
    """Cluster labels one row's arrays are keyed by, or ``[None]`` when it has none."""
    if not grid.clusters:
        return [None]
    if source.conformal == "marginal":
        return [None]
    if source.conformal == "cluster":
        return list(grid.clusters)
    if source.base == "raw":
        return [None]
    if source.base in COUNT_PVALUE_METHODS:
        # No cluster enters an unadjusted count p-value, so keying these rows by a
        # clustering would write the same numbers again under a cluster prefix.
        return [None]
    if config.inference.bh_scope in ("sample_cluster", "cluster"):
        return list(grid.clusters)
    return [None]


def _residual_term(
    baselines: BaselineConfig,
    plan: TargetPlan,
    grids: dict[int, ScaleGrid],
    scale: int,
    label: str | None,
) -> float | np.ndarray:
    """The observation-noise variance a spread-only method lacks, estimated at one scale.

    It is the calibration residual variance on the count scale: a float, or one value per cluster
    when it is estimated within clusters.
    """
    values = plan.scales[scale]
    to_counts = float(values.divisor) ** 2
    if label is None or baselines.residual_variance == "marginal":
        return residual_variance(values.fit_yhat, values.fit_y) * to_counts
    clustering = grids[scale].clusters[label]
    n_groups = int(max(clustering.calib.max(initial=0), clustering.test.max(initial=0))) + 1
    return residual_variance(
        values.fit_yhat,
        values.fit_y,
        np.asarray(clustering.calib, dtype=np.int64),
        n_groups,
        baselines.residual_min_units,
    ) * to_counts


def _per_unit_residual(
    residual: float | np.ndarray, n_units: int, clustering: Clustering | None
) -> np.ndarray:
    """Spread the count-scale residual estimate onto this scale's test units, one value each."""
    if np.ndim(residual) == 0 or clustering is None:
        return np.full(n_units, float(np.mean(residual)), dtype=np.float64)
    return np.asarray(residual)[np.asarray(clustering.test, dtype=np.int64)]


def _conformal_result(
    config: Config,
    target: Target,
    values: object,
    grid: ScaleGrid,
    source: MethodSource,
    prediction: np.ndarray,
    clustering: Clustering | None,
) -> UncertaintyResult:
    """One row that ranks a baseline's mean against the pipeline's held-out calibration data.

    The ranking is taken within the unit's histology cluster (``cluster``) or against one pooled
    residual distribution (``marginal``).
    """
    scored = replace(values, test_yhat=prediction)
    if source.conformal == "marginal":
        return _marginal_result(config, target, scored, grid)
    return _cluster_result(config, target, scored, grid, clustering)


def _regression_result(
    config: Config,
    target: Target,
    plan: TargetPlan,
    grid: ScaleGrid,
    source: MethodSource,
    members: tuple[np.ndarray, np.ndarray | None],
    residual: float | np.ndarray,
    clustering: Clustering | None,
) -> UncertaintyResult:
    """One gene or meta-gene row at one scale, from its ``(means, variances or None)`` members.

    Members read from a scale-1 store are first summed over this scale's blocks; the tail and the
    interval then follow the row's kind (point, conformal, draw count, negative binomial, Gaussian).
    """
    values = plan.scales[grid.scale]
    divisor = values.divisor
    mu, var = members
    if np.shape(mu)[1] != grid.test.n_groups:
        inverse, n_blocks = grid.test.inverse, grid.test.n_groups
        mu = block_sums(mu, inverse, n_blocks)
        if var is not None:
            var = block_sums(var, inverse, n_blocks)
    scaled = np.asarray(mu, dtype=np.float64) / divisor

    if source.conformal is None and source.base == "raw":
        # No p-value, no correction, no BH: this row exists to quantify what
        # thresholding a prediction with no uncertainty at all actually costs.
        prediction = scaled[0]
        pvals = np.where(prediction > target.null, 0.0, 1.0)
        return UncertaintyResult(
            prediction=prediction,
            pvals=pvals,
            pvals_adj=pvals,
            confidence=1.0 - pvals,
        )

    # Both aleatoric terms (variance head, residual estimate) arrive on the count scale,
    # so both are divided by the squared divisor; the count rows take neither.
    empirical = source.conformal is None and source.base in COUNT_PVALUE_METHODS
    if empirical:
        aleatoric = None
    elif var is not None:
        aleatoric = np.asarray(var, dtype=np.float64) / (divisor * divisor)
    else:
        aleatoric = _per_unit_residual(residual, scaled.shape[1], clustering) / (
            divisor * divisor
        )
    mu_bar, var_bar = mixture(scaled, aleatoric)
    sd = np.sqrt(np.maximum(var_bar, 0.0))

    if source.conformal is not None:
        return _conformal_result(config, target, values, grid, source, mu_bar, clustering)

    groups = _bh_groups(config, grid, clustering)
    # The count p-value: the fraction of members whose prediction does not exceed the null.
    count = 1.0 - (scaled > target.null).mean(axis=0)
    if empirical:
        # The interval is the middle `coverage` of the same draws with no
        # observation-noise term, so it under-covers by construction.
        lower, upper = draw_interval(scaled, float(config.sampling.coverage))
        return UncertaintyResult(
            prediction=mu_bar,
            pvals=count,
            pvals_adj=count,
            confidence=pvals_to_confidence(
                count, p_min=config.inference.p_min, p_max=config.inference.p_max
            ),
            std=sd,
            lower=lower,
            upper=upper,
        )

    if source.counts:
        # The tail is taken on the count scale, where the predictive is discrete
        # (hence `* divisor`); only the reported arrays are divided back.
        pvals = nb_pvals(mu_bar * divisor, var_bar * divisor * divisor, target.null * divisor)
        low, high = nb_interval(
            mu_bar * divisor, var_bar * divisor * divisor, float(config.sampling.coverage)
        )
        lower, upper = low / divisor, high / divisor
    else:
        z = NormalDist().inv_cdf(0.5 * (1.0 + float(config.sampling.coverage)))
        pvals = gaussian_pvals(mu_bar, sd, target.null)
        lower, upper = mu_bar - z * sd, mu_bar + z * sd

    pvals_adj = bh_adjust_grouped(pvals, groups)
    # The count is kept beside the tail as the assumption-free comparator.
    return UncertaintyResult(
        prediction=mu_bar,
        pvals=pvals,
        pvals_adj=pvals_adj,
        confidence=pvals_to_confidence(
            pvals_adj, p_min=config.inference.p_min, p_max=config.inference.p_max
        ),
        std=sd,
        lower=lower,
        upper=upper,
        extra={
            "pvals_count": count,
            "pvals_count_adj": bh_adjust_grouped(count, groups),
        },
    )


def _classification_result(
    config: Config,
    target: Target,
    plan: TargetPlan,
    grid: ScaleGrid,
    source: MethodSource,
    members: np.ndarray,
    clustering: Clustering | None,
    layout: object,
) -> UncertaintyResult:
    """One cell-type row at one scale, from the members' scale-1 probabilities.

    Apart from the raw and conformal rows, it states a local fdr (each block's Poisson-binomial
    tail, averaged over members) selected by the Sun & Cai step-up; ``layout`` is the scale's
    ``stats.BlockLayout``.
    """
    probability = np.clip(members, 0.0, 1.0)
    inverse, n_groups = grid.test.inverse, grid.test.n_groups
    sums = block_sums(probability, inverse, n_groups)
    mu_bar, spread = mixture(sums, None)

    if source.conformal is None and source.base == "raw":
        # The Bayes rule for 0-1 loss, generalised to a block as the plug-in count of
        # units whose own argmax is the target; it makes no attempt at FDR control.
        hits = block_sums((probability[0] > 0.5).astype(np.float64), inverse, n_groups)[0]
        pvals = np.where(hits > target.null, 0.0, 1.0)
        return UncertaintyResult(
            prediction=mu_bar, pvals=pvals, pvals_adj=pvals, confidence=1.0 - pvals
        )

    if source.conformal is not None:
        return _conformal_result(
            config, target, plan.scales[grid.scale], grid, source, mu_bar, clustering
        )

    # Accumulated rather than stacked: a hundred members over a whole slide's
    # scale-1 units would be most of a gigabyte held only to be averaged.
    lfdr = np.zeros(n_groups, dtype=np.float64)
    for member in probability:
        lfdr += poisson_binomial_tail(pad_blocks(member, layout), target.null)
    lfdr /= len(probability)
    adjusted = sun_cai_adjust(lfdr, _bh_groups(config, grid, clustering))
    return UncertaintyResult(
        prediction=mu_bar,
        pvals=lfdr,
        pvals_adj=adjusted,
        confidence=pvals_to_confidence(
            adjusted, p_min=config.inference.p_min, p_max=config.inference.p_max
        ),
        std=np.sqrt(np.maximum(spread, 0.0)),
    )


def _evaluation_records(
    pvals_adj: np.ndarray,
    truth: np.ndarray,
    null: float,
    evaluable: np.ndarray,
    clusters: np.ndarray | None,
    fdr_grid: Sequence[float],
    scale: int,
    label: str | None,
) -> list[dict[str, object]]:
    """Realised FDR and power at each nominal level, marginally and per cluster."""
    rows = np.flatnonzero(evaluable)
    adjusted, positives = pvals_adj[rows], truth[rows] > null
    groups: list[tuple[int | None, np.ndarray]] = [(None, np.ones(len(rows), dtype=bool))]
    if clusters is not None:
        labels = np.asarray(clusters)[rows]
        groups += [(int(value), labels == value) for value in np.unique(labels)]

    records: list[dict[str, object]] = []
    for cluster, mask in groups:
        selected_truth, selected_p = positives[mask], adjusted[mask]
        n_true = int(selected_truth.sum())
        for q in fdr_grid:
            called = selected_p <= q
            discoveries = int(called.sum())
            true_calls = int((called & selected_truth).sum())
            records.append(
                {
                    "scale": int(scale),
                    "clustering": label,
                    "cluster": cluster,
                    "q": float(q),
                    "n_units": int(mask.sum()),
                    "n_true": n_true,
                    "n_discoveries": discoveries,
                    "fdp": 0.0 if discoveries == 0 else 1.0 - true_calls / discoveries,
                    "power": 0.0 if n_true == 0 else true_calls / n_true,
                }
            )
    return records


def _coverage_records(
    source: MethodSource,
    divisor: float,
    result: UncertaintyResult,
    truth: np.ndarray,
    evaluable: np.ndarray,
    clusters: np.ndarray | None,
    levels: Sequence[float],
    scale: int,
    label: str | None,
    draws: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Interval coverage and width at each nominal level, marginally and per cluster.

    Each row is scored against the kind of interval it reports: Gaussian, negative binomial on
    the count scale, or the empirical quantiles of ``draws``.
    """
    rows = np.flatnonzero(evaluable)
    groups = None if clusters is None else np.asarray(clusters)[rows]
    mu_bar = result.prediction[rows]
    sd_bar = result.std[rows]
    ordered = None
    if draws is not None:
        # Sorted once for every level, on the reported scale: the draws arrive
        # on the count scale, as the members did.
        ordered = sort_draws(np.asarray(draws, dtype=np.float64)[:, rows] / divisor)
    records: list[dict[str, object]] = []
    for level in levels:
        bounds = None
        if ordered is not None:
            bounds = draw_interval(ordered, float(level), presorted=True)
        elif source.counts:
            low, high = nb_interval(
                mu_bar * divisor, (sd_bar * divisor) ** 2, float(level)
            )
            bounds = (low / divisor, high / divisor)
        for entry in coverage_and_width(
            mu_bar, sd_bar, truth[rows], level, groups, bounds
        ):
            records.append({"scale": int(scale), "clustering": label, **entry})
    return records


def _write_units(
    workspace: BaselineWorkspace,
    config: Config,
    modality: str,
    grid: ScaleGrid,
    views: Sequence[object],
) -> None:
    """Write the shared per-unit arrays without dropping any the tree already has.

    The pipeline's writer is called, then arrays it did not produce (other clusterings) are merged
    back in.
    """
    prior: dict[str, dict[str, np.ndarray]] = {}
    for view in views:
        path = workspace.units_file(config.model, view.name, modality, grid.scale)
        if path.exists():
            with np.load(path) as handle:
                prior[view.name] = {key: handle[key] for key in handle.files}
    _pipeline_units(workspace, config, modality, grid, views)
    for view in views:
        if view.name not in prior:
            continue
        path = workspace.units_file(config.model, view.name, modality, grid.scale)
        with np.load(path) as handle:
            written = {key: handle[key] for key in handle.files}
        merged = {**prior[view.name], **written}
        if set(merged) != set(written):
            np.savez_compressed(path, **merged)


def _write_target_arrays(
    workspace: BaselineWorkspace,
    config: Config,
    plan: TargetPlan,
    grid: ScaleGrid,
    views: Sequence[object],
) -> None:
    """Write the target's shared arrays only where the tree does not have them."""
    missing = [
        view
        for view in views
        if not (
            workspace.target_dir(config.model, view.name, plan.target.name)
            / f"prediction_scale{grid.scale}.npy"
        ).exists()
    ]
    if missing:
        _pipeline_target_arrays(workspace, config, plan, grid, missing)


def _write_row(
    config: Config,
    baselines: BaselineConfig,
    workspace: BaselineWorkspace,
    plan: TargetPlan,
    grids: dict[int, ScaleGrid],
    views: dict[int, Sequence[object]],
    source: MethodSource,
    results: dict[tuple[int, str | None], UncertaintyResult],
    draws: dict[int, np.ndarray] | None = None,
) -> None:
    """Write one row's arrays, maps and evaluation, sample by sample.

    ``results`` is keyed by ``(scale, cluster label or None)``; ``draws`` maps a scale to the
    per-member values of a row whose interval is a quantile of them.
    """
    target = plan.target
    visualize = baselines.visualizes(config)
    names = {view.name for scale in views for view in views[scale]}
    collected: dict[str, dict[str, np.ndarray]] = {name: {} for name in names}
    records: dict[str, dict[str, list[dict[str, object]]]] = {
        name: {"fdr": [], "coverage": []} for name in names
    }

    for (scale, label), result in results.items():
        grid, values = grids[scale], plan.scales[scale]
        clustering = None if label is None else grid.clusters[label]
        suffix = f"scale{scale}" if label is None else f"{label}_scale{scale}"
        for view in views[scale]:
            method_dir = workspace.method_dir(config.model, view.name, target.name, source.label)
            method_dir.mkdir(parents=True, exist_ok=True)
            # The only prediction that is a property of the row rather than of
            # the target: a mixture over members, not the pipeline's output.
            np.save(
                method_dir / f"prediction_scale{scale}.npy",
                result.prediction[view.rows].astype(np.float32),
            )
            if visualize and view.canvas is not None:
                _render(
                    view.canvas,
                    view.positions,
                    result.prediction[view.rows] / values.display,
                    method_dir / f"prediction_scale{scale}.png",
                    colour="magnitude",
                    overlay=_overlay(config, view, values),
                )
            null_dir = workspace.null_dir(
                config.model, view.name, target.name, source.label, target.null_name
            )
            _write_view_result(
                config, target, view, grid, values, result, scale, suffix, null_dir,
                collected[view.name], visualize,
            )

            if values.test_y is not None and result.pvals_adj is not None:
                evaluable = grid.evaluable[view.rows]
                clusters = None if clustering is None else clustering.test[view.rows]
                records[view.name]["fdr"] += _evaluation_records(
                    result.pvals_adj[view.rows], values.test_y[view.rows], target.null,
                    evaluable, clusters, baselines.fdr_grid, scale, label,
                )
                # Coverage and width are checked for genes only: an interval on a count
                # of Bernoullis would cover a parameter rather than a label.
                if target.modality == "gene" and result.std is not None:
                    member_draws = None
                    if draws is not None and scale in draws:
                        member_draws = draws[scale][:, view.rows]
                    sliced = UncertaintyResult(
                        prediction=result.prediction[view.rows], std=result.std[view.rows]
                    )
                    records[view.name]["coverage"] += _coverage_records(
                        source, values.divisor, sliced, values.test_y[view.rows],
                        evaluable, clusters, baselines.coverage, scale, label,
                        member_draws,
                    )

        # Written where the pipeline writes its own pooled curve, so the two are read together.
        if (
            visualize
            and len(views[scale]) > 1
            and values.test_y is not None
            and result.pvals_adj is not None
        ):
            pooled = workspace.pooled_null_dir(
                config.model, target.name, source.label, target.null_name
            )
            _write_pooled_curve(target, grid, values, result, pooled / f"fdp_curve_{suffix}.png")

    for name, arrays in collected.items():
        null_dir = workspace.null_dir(
            config.model, name, target.name, source.label, target.null_name
        )
        null_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(null_dir / "result.npz", **arrays)
        write_json(
            {
                "method": source.label,
                "row": source.name,
                "baseline": source.base,
                "conformal": source.conformal,
                "target": target.name,
                "null": {"name": target.null_name, "value": target.null},
                "bh_scope": config.inference.bh_scope,
                "caveat": CONFORMAL_CAVEAT if source.conformal else None,
                "fdr": records[name]["fdr"],
                "coverage": records[name]["coverage"],
            },
            null_dir / "evaluation.json",
        )


def _write_run_meta(
    config: Config,
    baselines: BaselineConfig,
    workspace: BaselineWorkspace,
    rows: Sequence[str],
    targets: dict[str, dict[str, object]],
    display: dict[str, dict[str, float]],
    nulls: dict[str, list[dict[str, object]]],
    cluster_labels: Sequence[str],
    samples: Sequence[str],
    stores: dict[str, dict[str, object]],
) -> None:
    """Add this run to the tree's header without dropping what the pipeline put there."""
    path = workspace.run_meta(config.model)
    existing: dict[str, object] = read_json(path) if path.exists() else {}

    methods = list(existing.get("methods") or [])
    methods += [row for row in rows if row not in methods]
    labels = list(existing.get("cluster_labels") or [])
    labels += [label for label in cluster_labels if label not in labels]
    merged_targets = {**(existing.get("targets") or {})}
    for name, entry in targets.items():
        merged_targets[name] = {**merged_targets.get(name, {}), **entry, "nulls": nulls[name]}

    # Rows and draw stores are merged because a run with ``--methods`` re-runs part of a
    # tree; the scalar options are replaced because they describe the rows just written.
    previous_baselines: dict[str, object] = dict(existing.get("baselines") or {})
    merged_rows = list(previous_baselines.get("rows") or [])
    merged_rows += [row for row in rows if row not in merged_rows]
    merged_stores = {**(previous_baselines.get("draw_stores") or {}), **stores}

    write_json(
        {
            **existing,
            "model": config.model,
            "mode": existing.get("mode", config.mode),
            "methods": methods,
            "bh_scope": config.inference.bh_scope,
            "cluster_labels": sorted(labels),
            "scales": list(config.scales),
            "samples": list(samples),
            "confidence_scale": {
                "p_min": config.inference.p_min,
                "p_max": config.inference.p_max,
            },
            "display_divisor": {**(existing.get("display_divisor") or {}), **display},
            "targets": merged_targets,
            "baselines": {
                **previous_baselines,
                "rows": merged_rows,
                "prefix": baselines.prefix,
                "n_members": baselines.n_members,
                "n_resamples": baselines.n_resamples,
                "n_passes": baselines.n_passes,
                "n_clusters": baselines.clusters(config),
                "residual_variance": baselines.residual_variance,
                "nll_beta": baselines.nll_beta,
                "warmup_epochs": baselines.warmup_epochs,
                "seed": baselines.seed,
                "draw_stores": merged_stores,
                # Every conformalised row, the marginal one included, borrows the
                # pipeline's held-out fold predictions.
                "caveat": (
                    CONFORMAL_CAVEAT
                    if baselines.conformalize or baselines.conformal_marginal
                    else None
                ),
            },
        },
        path,
    )


def _grid_config(config: Config, baselines: BaselineConfig) -> Config:
    """A copy of the run's config that makes ``_build_grid`` compute the baselines' clusters."""
    grid_config = copy.deepcopy(config)
    grid_config.n_clusters = baselines.clusters(config)
    grid_config.inference.methods = ["UTOPIA"]
    return grid_config


def _sources(
    config: Config,
    baselines: BaselineConfig,
    workspace: BaselineWorkspace,
    geometries: Sequence[object],
    modality: str,
    scales: Sequence[int],
    span: int,
    targets: Sequence[Target],
    calib_pred: TargetStore,
    test: Store,
    cost: dict[str, float],
) -> list[MethodSource]:
    """Train every bundle, generate every draw store, and list the rows to write.

    Gene-only methods are dropped for the cell-type modality, and ``cost`` gains the seconds spent.
    """
    chosen = [
        method
        for method in METHODS
        if method in baselines.methods
        and not (modality != "gene" and method in GENE_ONLY_METHODS)
    ]
    # Bundles are built at the union of these scales, not the reported ones, so a run
    # without scale 1 still builds the scale-1 bundle the aggregated rows are summed from.
    wanted = {
        method: [1] if method in AGGREGATED_METHODS else list(scales) for method in chosen
    }
    build = sorted({scale for method_scales in wanted.values() for scale in method_scales})

    # One target appearing under two nulls is one row: the nulls differ, the
    # predictions do not.
    seen: dict[str, Target] = {}
    for target in targets:
        seen.setdefault(target.name, target)

    stores: dict[str, dict[int, TargetStore]] = {method: {} for method in chosen}
    heteroscedastic: dict[str, bool] = {}
    distribution: dict[str, str | None] = {}
    for scale in build:
        here = [method for method in chosen if scale in wanted[method]]
        _train(
            config, baselines, workspace, geometries, modality, scale, span,
            here, calib_pred, cost,
        )
        grouping = coarse.group_units(test.sample_index, test.positions, scale, span)
        for method in here:
            meta = train_stage.bundle_meta(
                workspace.baseline_models(config.model, modality, method, scale)
            )
            specs = [
                DrawTarget(
                    name=target.name,
                    channels=list(target.channels),
                    weights=draw_stage.scale_weights(meta, target.channels, modality),
                )
                for target in seen.values()
            ]
            directory, elapsed = draw_stage.generate(
                config, baselines, workspace, geometries, modality, method,
                scale, grouping, test.sample_index, test.positions, specs,
            )
            cost[f"draws/{modality}/{method}/scale{scale}"] = elapsed
            store = TargetStore.open(directory, "pred")
            stores[method][scale] = store
            heteroscedastic[method] = bool(store.meta.get("heteroscedastic"))
            distribution[method] = store.meta.get("distribution")

    sources: list[MethodSource] = []
    for method in chosen:
        plain = MethodSource(
            method, baselines.label(method), method, None,
            stores[method], heteroscedastic[method], distribution[method],
        )
        sources.append(plain)
        # The conformal rows read the same stores as the plain row and differ only in name.
        if method == "raw" and baselines.conformal_marginal:
            row = CONFORMAL_MARGINAL
            sources.append(
                replace(plain, name=row, label=baselines.label(row), conformal="marginal")
            )
        if baselines.conformalize and method in UNCERTAIN_METHODS:
            row = method + CONFORMAL_SUFFIX
            sources.append(
                replace(plain, name=row, label=baselines.label(row), conformal="cluster")
            )
    return sources


def _train(
    config: Config,
    baselines: BaselineConfig,
    workspace: BaselineWorkspace,
    geometries: Sequence[object],
    modality: str,
    scale: int,
    span: int,
    chosen: Sequence[str],
    calib_pred: TargetStore,
    cost: dict[str, float],
) -> None:
    """Fit whatever bundles one scale needs and the workspace does not have.

    One training set is built per target scale (rescaled or raw counts) and shared by its bundles.
    """
    channels = calib_pred.channels
    current, _ = _predicted_channels(config, modality)
    if list(current) != list(channels):
        raise ValueError(
            f"{modality}: the calibration store holds {len(channels)} channel(s) but this config "
            f"asks for {len(current)}; 'predict_genes' changed since `calibrate` ran, and a "
            f"baseline must be trained on exactly what was calibrated. Re-run `calibrate` first."
        )
    missing = [
        method
        for method in chosen
        if train_stage.needs_training(
            workspace.baseline_models(config.model, modality, method, scale),
            channels,
            len(train_stage.member_seeds(config, baselines, method)),
            scale,
            coarse.coarse_batch_size(config, scale),
            train_stage._distribution(modality, method),
        )
    ]
    if not missing:
        print(f"[baselines] {modality} scale {scale}: every bundle already present")
        return

    # Scale 1 reuses the rescaling the calibration stage recorded, so a member's output and
    # the stored point prediction mean the same number; coarser scales compute their own.
    thresholds = None
    if modality == "gene" and scale == 1:
        stored = calib_pred.meta["thresholds"]
        thresholds = np.array([stored[name] for name in channels], dtype=np.float32)

    for rescale in (True, False):
        group = [
            method
            for method in missing
            if (method not in COUNT_SCALE_METHODS) == rescale
        ]
        if not group:
            continue
        kind = "rescaled" if rescale else "counts"
        scratch = workspace.baseline_scratch(config.model, modality, scale, kind)
        inputs = coarse.coarse_training_inputs(
            config, workspace, geometries, modality, scale, span, scratch,
            thresholds if rescale else None, rescale,
        )
        print(
            f"[baselines] {modality} scale {scale} ({kind} targets): {len(inputs.table)} "
            f"training unit(s), batch {inputs.hyper['batch_size']} x "
            f"{config.default.epochs} epoch(s) = "
            f"{-(-len(inputs.table) // int(inputs.hyper['batch_size'])) * config.default.epochs} "
            f"gradient step(s); bundles {group}"
        )
        for method in group:
            cost[f"train/{modality}/{method}/scale{scale}"] = train_stage.train_bundle(
                config,
                baselines,
                modality,
                method,
                scale,
                inputs,
                workspace.baseline_models(config.model, modality, method, scale),
            )
        del inputs
        shutil.rmtree(scratch, ignore_errors=True)


def run(config: Config, baselines: BaselineConfig) -> None:
    """Run every configured baseline over every target, one target at a time.

    Bundles are trained and draws generated once per modality; each target is then read, written
    and freed, so peak memory is one target's arrays.
    """
    workspace = BaselineWorkspace(config)
    geometries = workspace.load_geometries()
    names = [geometry.name for geometry in geometries]
    span = max((max(geometry.lattice_shape) for geometry in geometries), default=1)
    targets = load_targets(config.targets_path)
    grid_config = _grid_config(config, baselines)
    rows = baselines.labels()
    cost: dict[str, float] = {}
    started = time.time()

    print(f"[baselines] model '{config.model}'; row(s): {', '.join(rows)}")
    print(f"[baselines] results -> {workspace.results_root(config.model)}")
    print(f"[baselines] models and draws -> {workspace.baselines_root(config.model)}")
    print(f"[baselines] M = {baselines.n_members} (ensembles) / {baselines.n_resamples} "
          f"(resampling), K = {baselines.n_passes}, clusters {baselines.clusters(config)}, "
          f"BH scope {config.inference.bh_scope}")
    print(f"[baselines] genes: {list(COUNT_PVALUE_METHODS)} report the count of draws past the "
          "null, unadjusted and not conditioned on histology")
    print("[baselines] genes: one bundle per scale, trained on that scale's ROI blocks "
          "(mean features, summed counts), except "
          f"{list(AGGREGATED_METHODS)}, which keep one scale-1 bundle and are summed "
          "over the block; cell types: one scale-1 classifier")
    if baselines.conformalize or baselines.conformal_marginal:
        print(f"[baselines] note: {CONFORMAL_CAVEAT}")

    meta_by_target: dict[str, dict[str, object]] = {}
    display_by_target: dict[str, dict[str, float]] = {}
    nulls_by_target: dict[str, list[dict[str, object]]] = {}
    cluster_labels: list[str] = []
    drawn: dict[str, dict[str, object]] = {}
    sample_names: list[str] = []

    for modality in config.predict:
        selected = [target for target in targets if target.modality == modality]
        if not selected:
            continue
        stores = open_stores(workspace, config.model, modality)
        calib, test = stores["calib"], stores["test"]
        truth_name = "truth_labels" if modality == "cell_type" else "truth"
        if not test.has(truth_name):
            raise ValueError(
                f"{modality}: {test.directory}/{truth_name}.npy is missing, and the baselines "
                "have no purpose except to be scored; re-run `calibrate` with 'benchmark: true'"
            )
        sample_names = names

        grids = {
            scale: _build_grid(grid_config, workspace, names, calib, test, scale, span)
            for scale in config.scales
        }
        if modality == "gene":
            # Scored against the truth UTOPIA is scored against: every transcript
            # in a block.  The per-scale models still train on tissue sums.
            add_off_tissue(stores, config, geometries, grids, span)
        canvases = {
            scale: _canvases(workspace, geometries, test, names, scale) for scale in config.scales
        }
        views = {
            scale: _sample_views(grids[scale], canvases[scale], names) for scale in config.scales
        }
        layouts = {}
        if modality == "cell_type":
            layouts = {
                scale: block_layout(
                    grids[scale].test.inverse, grids[scale].test.n_groups, grids[scale].test.counts
                )
                for scale in config.scales
            }

        model_scales = list(config.scales) if modality == "gene" else [1]
        sources = _sources(
            config, baselines, workspace, geometries, modality, model_scales, span,
            selected, stores["calib_pred"], test, cost,
        )
        # A store built for another grouping would misattribute every unit, so each
        # store's unit count is checked against the grid it is read on.
        for source in sources:
            for scale, store in source.stores.items():
                if scale in grids and store.n_units != grids[scale].test.n_groups:
                    raise ValueError(
                        f"{modality} {source.name}: the scale-{scale} draw store holds "
                        f"{store.n_units} unit(s) but this scale's grid reports "
                        f"{grids[scale].test.n_groups}; the store was generated against a "
                        "different grouping and must be regenerated"
                    )
        for source in sources:
            if source.conformal is None:
                for scale, store in source.stores.items():
                    drawn[f"{modality}/{source.base}/scale{scale}"] = {
                        "path": str(store.path),
                        "bytes": store.path.stat().st_size,
                        "members": store.meta.get("members"),
                        "units": int(store.n_units),
                    }

        for scale in config.scales:
            _write_units(workspace, config, modality, grids[scale], views[scale])
            cluster_labels = sorted(set(cluster_labels) | set(grids[scale].clusters))

        print(f"[baselines] {modality}: {len(selected)} target(s) x {len(sources)} row(s)")
        for target in selected:
            plan = _build_plan(config, target, stores, grids)
            meta_by_target[target.name] = {
                "modality": target.modality,
                "channels": list(target.channels),
            }
            display_by_target[target.name] = {
                str(scale): float(plan.scales[scale].display) for scale in config.scales
            }
            entry = {"name": target.null_name, "value": target.null}
            recorded = nulls_by_target.setdefault(target.name, [])
            if entry not in recorded:
                recorded.append(entry)
            for scale in config.scales:
                _write_target_arrays(workspace, config, plan, grids[scale], views[scale])

            # The residual-variance term depends on the target, scale and cluster label,
            # never on the row, so it is estimated once and shared by the rows.
            residuals: dict[tuple[int, str | None], float | np.ndarray] = {}
            for source in sources:
                results: dict[tuple[int, str | None], UncertaintyResult] = {}
                # One store's members at a time: holding every scale's would be K draws
                # over all their units at once for no gain.
                held: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
                # A row whose interval is a quantile of its draws needs them again for
                # the coverage check, so its per-scale means outlive `held`.
                reported_draws: dict[int, np.ndarray] = {}
                for scale in config.scales:
                    grid = grids[scale]
                    store_scale = source.store_scale(scale)
                    if store_scale not in held:
                        held = {
                            store_scale: draw_stage.read_members(
                                source.stores[store_scale], target.name, source.heteroscedastic
                            )
                        }
                    members = held[store_scale]
                    if modality == "gene" and source.conformal is None and (
                        source.base in COUNT_PVALUE_METHODS
                    ):
                        reported_draws[scale] = members[0]
                    for label in _labels(config, source, grid):
                        clustering = None if label is None else grid.clusters[label]
                        if modality == "gene":
                            if (scale, label) not in residuals:
                                residuals[(scale, label)] = _residual_term(
                                    baselines, plan, grids, scale, label
                                )
                            result = _regression_result(
                                config, target, plan, grid, source,
                                members, residuals[(scale, label)], clustering,
                            )
                        else:
                            result = _classification_result(
                                config, target, plan, grid, source,
                                members[0], clustering, layouts[scale],
                            )
                        results[(scale, label)] = result
                _write_row(
                    config, baselines, workspace, plan, grids, views, source, results,
                    reported_draws or None,
                )
                # Freed here so the arrays do not outlive the row into the next target's plan.
                del held, results, reported_draws
            del plan
        print(f"[baselines] {modality}: done")

    cost["total"] = time.time() - started
    _write_run_meta(
        config, baselines, workspace, rows, meta_by_target, display_by_target,
        nulls_by_target, cluster_labels, sample_names, drawn,
    )
    workspace.cost(config.model).parent.mkdir(parents=True, exist_ok=True)
    write_json(cost, workspace.cost(config.model))
    print(
        f"[baselines] wrote {len(rows)} row(s) into {workspace.results_root(config.model)} "
        f"in {cost['total']:.1f}s"
    )
