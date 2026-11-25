"""Histology clustering: PCA plus KMeans over the extracted features.

Cluster labels make the conformal p-values cluster-conditional. Each scale gets one PCA and one
KMeans per cluster count, fitted jointly on every unit of every sample at that scale.
Labels are stored per sample as dense lattice images; only missing ``(k, scale)`` pairs are fitted.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .aggregate import (
    block_mean_features,
    coarse_shape,
    coarsen_mask,
    group_index,
    group_mean,
    lattice_keys,
)
from .config import Config
from .geometry import Geometry
from .io import save_image
from .workspace import Workspace

#: Label written where a lattice cell holds no tissue.
BACKGROUND = 255


def reduce_with_pca(
    values: np.ndarray, pca_dim: int, random_state: int
) -> tuple[np.ndarray, object | None]:
    """Fit one PCA on all units and return the reduced float32 values with the fitted PCA.

    Features that are already narrow enough are returned unchanged, with None for the PCA.
    """
    from sklearn.decomposition import PCA

    if values.shape[1] <= pca_dim:
        return values, None
    pca = PCA(n_components=min(pca_dim, *values.shape), random_state=random_state).fit(values)
    return pca.transform(values).astype(np.float32), pca


def fit_and_save_kmeans(
    values: np.ndarray, n_clusters: int, random_state: int, pca: object | None, model_path: Path
) -> object:
    """Fit KMeans (``n_init=10``, k-means++, lloyd) on every unit and pickle it with the PCA."""
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(values)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as handle:
        pickle.dump({"pca": pca, "kmeans": kmeans}, handle)
    return kmeans


def coarsen_features(
    features: np.ndarray, positions: np.ndarray, scale: int, span: int
) -> tuple[np.ndarray, np.ndarray]:
    """Average features within each ``scale x scale`` block of the lattice.

    Returns ``(coarse_features, coarse_positions)``, positions in coarse-lattice coordinates.
    """
    zeros = np.zeros(len(positions), dtype=np.int64)
    keys = lattice_keys(zeros, positions, scale, span)
    _, inverse, counts = group_index(keys)
    coarse = group_mean(np.asarray(features, dtype=np.float32), inverse, len(counts), counts)
    parents = np.unique(positions // scale, axis=0)
    return coarse, parents


def scale_units(
    features: np.ndarray, geometry: Geometry, scale: int, span: int, cover_lattice: bool
) -> tuple[np.ndarray, np.ndarray]:
    """One sample's units at ``scale`` as ``(features, coarse positions)``.

    A unit needs at least one tissue superpixel; its feature is the mean over its tissue
    superpixels, or over all its superpixels when features cover the lattice (``very_slow``).
    """
    if cover_lattice:
        kept = coarsen_mask(geometry.tissue_mask, scale)
        parents = np.argwhere(kept).astype(np.int32)
        return block_mean_features(features, geometry.lattice_shape, scale, parents), parents

    tissue = np.argwhere(geometry.tissue_mask).astype(np.int32)
    if scale == 1:
        return np.asarray(features, dtype=np.float32), tissue
    return coarsen_features(features, tissue, scale, span)


def run(config: Config) -> None:
    """Run the clustering stage, fitting only the ``(k, scale)`` pairs not yet present."""
    workspace = Workspace(config)
    source = config.cluster_source
    settings = config.clustering
    geometries = workspace.load_geometries()
    span = max(max(geometry.lattice_shape) for geometry in geometries)

    existing: dict[str, dict[str, np.ndarray]] = {}
    for geometry in geometries:
        path = workspace.clusters(geometry.name, source)
        if path.exists():
            with np.load(path) as data:
                existing[geometry.name] = {key: data[key] for key in data.files}
        else:
            existing[geometry.name] = {}

    # Scale 1 is always clustered, even if not reported at: fold generation checks cluster
    # coverage on the scale-1 lattice.
    for scale in sorted(set(config.scales) | {1}):
        missing = []
        for n_clusters in config.n_clusters:
            key = f"k{n_clusters}_scale{scale}"
            present_everywhere = all(key in existing[geometry.name] for geometry in geometries)
            if not present_everywhere:
                missing.append(n_clusters)
        for n_clusters in sorted(set(config.n_clusters) - set(missing)):
            print(f"[cluster] k{n_clusters}_scale{scale}: already present, skipping")
        if not missing:
            continue

        blocks: list[np.ndarray] = []
        positions: list[np.ndarray] = []
        for geometry in geometries:
            features = np.load(workspace.features(geometry.name, config.extractor), mmap_mode="r")
            values, units = scale_units(
                features, geometry, scale, span, config.features_cover_lattice
            )
            blocks.append(values)
            positions.append(units)
        values = np.concatenate(blocks)
        del blocks
        input_dim = values.shape[1]
        values, pca = reduce_with_pca(values, settings.pca_dim, settings.random_state)
        print(
            f"[cluster] scale {scale}: PCA {input_dim} -> "
            f"{values.shape[1]} fitted on all {len(values)} units"
        )

        for n_clusters in missing:
            key = f"k{n_clusters}_scale{scale}"
            model_path = workspace.clustering_model(source, scale, n_clusters)
            kmeans = fit_and_save_kmeans(values, n_clusters, settings.random_state, pca, model_path)

            offset = 0
            for geometry, units in zip(geometries, positions):
                labels = kmeans.labels_[offset : offset + len(units)].astype(np.uint8)
                offset += len(units)
                shape = coarse_shape(geometry.lattice_shape, scale)
                image = np.full(shape, BACKGROUND, dtype=np.uint8)
                image[units[:, 0], units[:, 1]] = labels
                existing[geometry.name][key] = image
            print(f"[cluster] {key}: KMeans fitted on all {len(values)} units")
        del values

    from . import plot

    for geometry in geometries:
        np.savez_compressed(workspace.clusters(geometry.name, source), **existing[geometry.name])
        for key, image in existing[geometry.name].items():
            map_path = workspace.cluster_map(geometry.name, source, key)
            save_image(plot.cluster_map(image), map_path)
        print(f"[cluster] {geometry.name}: {sorted(existing[geometry.name])}")
