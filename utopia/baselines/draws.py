"""Running the trained baselines over the test area and storing what they said.

One pass over the test features gives one draw for every target, written straight into a per-method,
per-scale :class:`~utopia.io.TargetStore` of raw count-scale predictions with no null folded in.
The store is ``(targets, units)`` for ``raw``, ``(targets, draws, units)`` for most methods, and
``(targets, 2M, units)`` float32 -- M means, then M variances -- for the gene deep ensembles.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from ..calibration.draws import (
    DrawTarget,
    TestFeatures,
    check_fits_float16,
    draw_signature,
    target_columns,
)
from ..config import Config
from ..geometry import Geometry
from ..io import TargetStore, read_store_meta, write_json
from ..models.default import default_device, enable_dropout, rebuild
from .coarse import CoarseFeatures, CoarseGrouping, lattice_block_features
from .config import BaselineConfig, BaselineWorkspace
from .models import (
    HeteroscedasticRegressor,
    NegativeBinomialRegressor,
    predict_heteroscedastic,
    predict_negative_binomial,
    rebuild_member,
)

#: Maps a bundle's recorded distribution to its members' network class and the function
#: that runs one; both predictors return ``(mean, variance)`` per kept channel.
_PREDICTORS = {
    "gaussian": (HeteroscedasticRegressor, predict_heteroscedastic),
    "negative_binomial": (NegativeBinomialRegressor, predict_negative_binomial),
}


def scale_weights(
    meta: dict[str, object], channels: Sequence[str], modality: str
) -> np.ndarray:
    """The weight each member channel enters a target's sum with, at one scale.

    For a gene bundle trained on rescaled targets it is the gene's upper threshold in
    ``meta["thresholds"]``, which puts a ``[0, 1]`` prediction back on the count scale; a
    count-scale bundle and a cell-type one weigh every channel by 1.
    """
    thresholds = meta.get("thresholds")
    if modality != "gene" or meta.get("count_scale") or thresholds is None:
        return np.ones(len(channels), dtype=np.float32)
    missing = [name for name in channels if name not in thresholds]
    if missing:
        raise KeyError(
            f"the bundle for scale {meta.get('scale')} was not trained for gene(s) {missing}; "
            f"it holds {len(thresholds)}. Re-run the baselines after changing 'predict_genes'."
        )
    return np.array([thresholds[name][1] for name in channels], dtype=np.float32)


class Members:
    """One baseline's trained members, ready to run over the test features.

    ``distribution`` is ``"gaussian"`` (rescaled target), ``"negative_binomial"`` (raw counts) or
    None for a point prediction; ``n_draws`` is M members, or for ``mc_dropout`` K stochastic
    passes through its single model.
    """

    def __init__(
        self, baselines: BaselineConfig, modality: str, method: str, path: Path
    ) -> None:
        """Load the bundle at ``path`` and rebuild its members on the default device."""
        bundle = torch.load(path, map_location="cpu")
        self.channels: list[str] = bundle["channels"]
        self.heteroscedastic = bool(bundle["heteroscedastic"])
        self.distribution: str | None = bundle.get("distribution")
        self.stochastic = method == "mc_dropout"
        self.seed = int(baselines.seed)
        self.device = default_device()
        dropout = bundle["dropout"] if baselines.dropout is None else baselines.dropout
        input_dim, output_dim = bundle["input_dim"], bundle["output_dim"]
        network, self.predict = _PREDICTORS.get(self.distribution, (None, None))
        self.models = []
        for state in bundle["state_dicts"]:
            if network is not None:
                model = rebuild_member(network, state, input_dim, output_dim, dropout)
            else:
                model = rebuild(modality, state, input_dim, output_dim, dropout)
            self.models.append(model)
        self.n_draws = baselines.members(method)
        for model in self.models:
            model.to(self.device).eval()
            if self.stochastic:
                # Only dropout goes back into training mode: batch normalisation keeps its
                # running statistics, so successive passes differ by the mask alone.
                enable_dropout(model)

    def channel_index(self, names: Sequence[str]) -> np.ndarray:
        """Output columns of the named channels."""
        lookup = {name: column for column, name in enumerate(self.channels)}
        return np.array([lookup[name] for name in names], dtype=np.int64)

    def draw(
        self, index: int, features: TestFeatures, batch_size: int, columns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Run draw ``index`` over the whole test area, keeping the output ``columns``.

        Returns ``(mean, variance)``, each ``(n_units, len(columns))`` float32 on the scale the
        bundle was fitted against; ``variance`` is None for members without a variance head.
        """
        model = self.models[0] if self.stochastic else self.models[index]
        if self.stochastic:
            torch.manual_seed(self.seed + index)
        selected = torch.as_tensor(columns, device=self.device)
        mean = np.empty((features.n_units, len(columns)), dtype=np.float32)
        variance = np.empty_like(mean) if self.heteroscedastic else None
        with torch.inference_mode():
            for start, stop, block in features.batches(batch_size):
                x = torch.as_tensor(np.asarray(block, dtype=np.float32), device=self.device)
                if self.heteroscedastic:
                    mean[start:stop], variance[start:stop] = self.predict(model, x, selected)
                else:
                    mean[start:stop] = model(x).index_select(1, selected).float().cpu().numpy()
        return mean, variance


def generate(
    config: Config,
    baselines: BaselineConfig,
    workspace: BaselineWorkspace,
    geometries: Sequence[Geometry],
    modality: str,
    method: str,
    scale: int,
    grouping: CoarseGrouping,
    sample_index: np.ndarray,
    positions: np.ndarray,
    targets: Sequence[DrawTarget],
) -> tuple[Path, float]:
    """Run every draw of one baseline over the test area and store the result.

    ``grouping`` maps the scale-1 test units onto this scale's units; returns
    ``(store directory, wall-clock seconds)``, the seconds being zero when a stored set with the
    same signature was reused.
    """
    directory = workspace.baseline_draws(config.model, modality, method, scale)
    bundle = workspace.baseline_models(config.model, modality, method, scale)
    predictions = workspace.prediction_dir(config.model, modality) / "pred.npy"
    n_units = int(grouping.n_groups)
    # The scale fixes which units a draw is taken at and which network made it.
    settings = {
        "scale": int(scale),
        "n_draws": int(baselines.members(method)),
        "dropout": baselines.dropout,
        "seed": int(baselines.seed),
        "batch_size": int(baselines.batch_size),
    }
    signature = draw_signature(
        method, settings, config.extractor, bundle, predictions, targets, n_units
    )
    if read_store_meta(directory).get("draw_signature") == signature:
        print(f"[baselines] {modality} {method} scale {scale}: reusing stored draws")
        return directory, 0.0

    from tqdm import tqdm

    started = time.time()
    members = Members(baselines, modality, method, bundle)
    features = TestFeatures(workspace, config, geometries, sample_index, positions)
    if not grouping.is_identity:
        # A coarse unit's input is its block's mean feature vector, which is what
        # the bundle was trained on; built once here and sliced per batch.
        lattice_values = None
        if config.features_cover_lattice:
            lattice_values = lattice_block_features(workspace, config, geometries, grouping)
        features = CoarseFeatures(features, grouping, lattice_values=lattice_values)

    columns, offsets = target_columns(members.channel_index, targets)

    # Rewritten whole: a target left over from an earlier list would otherwise
    # still be readable under a signature it does not belong to.
    shutil.rmtree(directory, ignore_errors=True)
    stack = None if method == "raw" else members.n_draws
    if members.heteroscedastic:
        # M means followed by M variances, in float32 because a variance near the
        # floor would underflow to zero in float16 and then decide a p-value.
        stack, dtype = 2 * members.n_draws, np.float32
    else:
        dtype = np.float16
    store = TargetStore.create(
        directory,
        "pred",
        [target.name for target in targets],
        n_units,
        dtype,
        stack=stack,
    )
    writer = store.writer()

    for index in tqdm(range(members.n_draws), desc=f"[baselines] {method}", unit="draw"):
        mean, variance = members.draw(index, features, baselines.batch_size, columns)
        for row, (target, offset) in enumerate(zip(targets, offsets)):
            values = (mean[:, offset] * target.weights).sum(axis=1)
            if dtype == np.float16:
                check_fits_float16(target.name, values)
            if method == "raw":
                writer[row] = values.astype(dtype)
            else:
                writer[row, index] = values.astype(dtype)
            if variance is not None:
                # Variances scale by the squared weight; channels are treated as independent
                # given the member, so a multi-gene target's cross-gene covariance is left out.
                writer[row, members.n_draws + index] = (
                    (variance[:, offset] * target.weights**2).sum(axis=1).astype(dtype)
                )
    writer.flush()

    meta = read_store_meta(directory)
    meta["draw_signature"] = signature
    meta["members"] = int(members.n_draws)
    meta["heteroscedastic"] = bool(members.heteroscedastic)
    meta["distribution"] = members.distribution
    meta["scale"] = int(scale)
    write_json(meta, directory / "meta.json")
    elapsed = time.time() - started
    print(
        f"[baselines] {modality} {method} scale {scale}: {members.n_draws} draw(s) over "
        f"{len(targets)} target(s) x {n_units} unit(s) in {elapsed:.1f}s"
    )
    return directory, elapsed


def read_members(
    store: TargetStore, target_name: str, heteroscedastic: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read one target's members out of a baseline draw store.

    Returns ``(mu, var)``, each ``(M, N)`` float32; ``var`` is None unless ``heteroscedastic``, and
    a ``raw`` store, which has no member axis, comes back as a single member.
    """
    values = store.read([target_name])[0]
    if values.ndim == 1:
        return values[None, :], None
    if not heteroscedastic:
        return values, None
    members = values.shape[0] // 2
    return values[:members], values[members:]
