"""Stage 2: turn each sample's H&E into one feature vector per superpixel.

``features.method`` picks ``fast`` (CATCH: one pass per 224-px window at stride 112, tokens fused
onto the lattice), ``slow`` (one pass per tissue superpixel) or ``very_slow`` (one pass per
superpixel of the whole lattice). Rows are float16, global stream then local stream, for the
tissue superpixels in tissue-mask order; ``very_slow`` writes every superpixel in row-major
lattice order. The PCA is fitted once across all samples, on tissue superpixels only.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import Config
from ..geometry import Geometry
from ..io import load_image, write_json
from ..workspace import Workspace
from . import encoder, fast, harmony, slow, very_slow
from .encoder import Projection

#: Extractor module per ``features.method``.
MODULES = {"fast": fast, "slow": slow, "very_slow": very_slow}

__all__ = ["run"]


def _fit_projections(
    config: Config, workspace: Workspace, model: torch.nn.Module, device: torch.device
) -> tuple[Projection, Projection]:
    """Fit the global and local PCA jointly over the fit windows of every sample."""
    if config.features.method == "fast":
        tokens = config.features.fit_tokens_per_patch
    else:
        # The per-superpixel extractors keep only the centre token.
        tokens = None
    fit = encoder.PcaFit(device, tokens, config.features.random_state)
    module = MODULES[config.features.method]
    for sample in config.samples:
        geometry = Geometry.load(workspace.geometry(sample.name))
        image = load_image(workspace.he(sample.name))
        before = fit.n_windows
        module.fit_rows(model, image, geometry.tissue_mask, device, config, fit)
        print(f"[features] {sample.name}: {fit.n_windows - before} fit windows")
        del image
    return fit.projections(config.features.pca)


def _harmonise(
    config: Config, workspace: Workspace, window_globals: dict[str, np.ndarray], global_dim: int
) -> None:
    """Harmony-correct the window embeddings across samples and re-fuse the global stream.

    Fusion is linear in the window embeddings, so the global half of each feature file is
    overwritten without a second GPU pass.
    """
    names = list(window_globals)
    geometries = {name: Geometry.load(workspace.geometry(name)) for name in names}
    selections = {}
    for name in names:
        grid = window_globals[name].shape[:2]
        selections[name] = fast.select_windows(geometries[name].tissue_mask, grid, config)
    stacked = np.concatenate([window_globals[name][selections[name]] for name in names])
    batches = [name for name in names for _ in range(int(selections[name].sum()))]
    print(f"[features] harmony over {len(stacked)} windows from {len(names)} samples")
    corrected = harmony.correct(
        stacked, batches, config.features.harmony_iterations, config.features.random_state
    )

    offset = 0
    for name in names:
        present = selections[name]
        count = int(present.sum())
        block = np.zeros_like(window_globals[name])
        block[present] = corrected[offset : offset + count]
        offset += count
        geometry = geometries[name]
        fused = fast.fuse_globals(block, geometry.lattice_shape, present)
        path = workspace.features(name, config.extractor)
        features = np.load(path)
        features[:, :global_dim] = fused[geometry.tissue_mask].astype(np.float16)
        np.save(path, features)
        print(f"[features]   {name}: corrected {global_dim} global channels")


def run(config: Config) -> None:
    """Run the feature-extraction stage for every sample in ``config``."""
    workspace = Workspace(config)
    settings = config.features
    module = MODULES[settings.method]
    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    model = encoder.load_encoder(settings.model_dir, device)
    print(
        f"[features] {settings.method} extractor -> '{config.extractor}' on {device} "
        f"({config.regime.replace('_', '-')}, harmony {'on' if config.harmony_enabled else 'off'})"
    )

    projections: tuple[Projection, Projection] | None = None
    if settings.pca:
        projections = _fit_projections(config, workspace, model, device)
    global_dim = projections[0].dim if projections else encoder.RAW_DIM

    if config.features_cover_lattice:
        order = "lattice order (row-major over every superpixel)"
    else:
        order = "tissue mask order (numpy.argwhere of tissue_mask.png)"

    window_globals: dict[str, np.ndarray] = {}
    for sample in config.samples:
        print(f"[features] {sample.name}")
        geometry = Geometry.load(workspace.geometry(sample.name))
        image = load_image(workspace.he(sample.name))
        features, provenance, windows = module.extract_sample(
            model, image, geometry.tissue_mask, device, config, projections
        )
        del image
        np.save(workspace.features(sample.name, config.extractor), features)
        write_json(
            {
                "extractor": config.extractor,
                "method": settings.method,
                "encoder": "UNI vit_large_patch16_224",
                "harmony": config.harmony_enabled,
                "pca": settings.pca,
                "dim": int(features.shape[1]),
                "n_units": int(features.shape[0]),
                "order": order,
                **provenance,
            },
            workspace.features_meta(sample.name, config.extractor),
        )
        print(f"[features]   wrote {features.shape} float16")
        if config.harmony_enabled:
            window_globals[sample.name] = windows

    if config.harmony_enabled:
        _harmonise(config, workspace, window_globals, global_dim)
