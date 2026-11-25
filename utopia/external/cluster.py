"""Stage ``cluster`` of an external run: histology clusters of the spots.

The spot features of every sample, calibration and test together, are reduced with one PCA
(100 dimensions by default), optionally corrected for slide-to-slide batch effects with Harmony
(``he_batch_effect_correction``), and clustered with KMeans once per requested cluster count. The
labels are written into both stores as ``clusters_k{n}.npy``, which is where the ``infer`` stage
looks for them. This stage does nothing when the user supplies the cluster labels.
"""

from __future__ import annotations

import numpy as np

from .. import plot
from ..clustering import BACKGROUND, fit_and_save_kmeans, reduce_with_pca
from ..config import EXTERNAL_EXTRACTOR, EXTERNAL_ROLES, Config
from ..features import harmony
from ..io import load_mask, read_store_meta, read_store_units, save_image
from ..workspace import Workspace
from .inputs import (
    CLUSTER_PREFIX,
    CLUSTER_SUFFIX,
    provided_cluster_labels,
    store_directory,
)


def reduce_features(
    config: Config, features: np.ndarray, batches: list[str]
) -> tuple[np.ndarray, object]:
    """Reduce ``(N, 1024)`` float32 spot features to what KMeans is fitted on, with the PCA.

    With ``he_batch_effect_correction`` the reduced features are then Harmony-corrected, with one
    batch per sample; ``batches`` names the sample of every row.
    """
    settings = config.clustering
    reduced, pca = reduce_with_pca(features, settings.pca_dim, settings.random_state)
    print(
        f"[cluster] PCA {features.shape[1]} -> {reduced.shape[1]} fitted on all "
        f"{len(reduced)} spots"
    )

    if config.he_batch_effect_correction:
        n_samples = len(set(batches))
        print(f"[cluster] Harmony over {len(reduced)} spots from {n_samples} samples")
        reduced = harmony.correct(
            reduced, batches, config.features.harmony_iterations, settings.random_state
        )
    return reduced, pca


def render_cluster_maps(config: Config, workspace: Workspace, label: str) -> None:
    """Draw one cluster map per sample for the clustering ``label``, on its tissue mask if any.

    A clustering whose labels do not fit the colour table (user labels can be any integers) is not
    drawn.
    """
    names = config.sample_names
    for role in EXTERNAL_ROLES:
        directory = store_directory(config, workspace, role)
        labels = np.load(directory / f"{CLUSTER_PREFIX}{label}{CLUSTER_SUFFIX}")
        if labels.min() < 0 or labels.max() >= len(plot.CLUSTER_COLOURS):
            print(
                f"[cluster] {label}: labels run from {int(labels.min())} to {int(labels.max())}, "
                f"outside the {len(plot.CLUSTER_COLOURS)} map colours; no cluster map is drawn"
            )
            return

        units = read_store_units(directory)
        meta = read_store_meta(directory)
        square = meta.get("spot_shape") == "square"
        for sample in config.samples_with_role(role):
            index = names.index(sample.name)
            rows = units["sample_index"] == index
            shape = (int(meta["canvas_shape"][index][0]), int(meta["canvas_shape"][index][1]))
            painted, drawn = plot.paint(
                labels[rows], units["positions"][rows], shape, units["radius"][rows], square
            )
            label_image = np.full(shape, BACKGROUND, dtype=np.uint8)
            label_image[drawn] = painted[drawn].astype(np.uint8)
            image = plot.cluster_map(label_image)

            mask_path = workspace.tissue_mask(sample.name)
            if mask_path.exists():
                tissue = load_mask(mask_path)
                if tissue.shape == shape:
                    image[tissue & ~drawn] = plot.MAGNITUDE_TISSUE
            save_image(image, workspace.cluster_map(sample.name, config.external_extractor, label))


def load_features(
    config: Config, workspace: Workspace
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Stack the spot features of every sample as float32, in configuration order.

    Returns the ``(N, 1024)`` features, the sample name of every row, and each sample's spot count.
    """
    blocks: list[np.ndarray] = []
    batches: list[str] = []
    n_spots: dict[str, int] = {}
    for sample in config.samples:
        path = workspace.features(sample.name, EXTERNAL_EXTRACTOR)
        if not path.exists():
            raise FileNotFoundError(
                f"sample '{sample.name}' has no spot features at {path}; run the features stage "
                "first"
            )
        features = np.load(path).astype(np.float32)
        blocks.append(features)
        batches.extend([sample.name] * len(features))
        n_spots[sample.name] = len(features)
    return np.concatenate(blocks), batches, n_spots


def _settings(config: Config) -> np.ndarray:
    """The clustering settings a saved label file must have been made with to be reused."""
    return np.array([config.clustering.pca_dim, config.clustering.random_state], dtype=np.int64)


def saved_labels(config: Config, workspace: Workspace, label: str) -> dict[str, np.ndarray] | None:
    """Cluster labels an earlier run of this stage saved for every sample, or None.

    They are reused only when every sample has them, made with the current clustering settings,
    and newer than the sample's spot features and as long.
    """
    found: dict[str, np.ndarray] = {}
    for sample in config.samples:
        features_path = workspace.features(sample.name, EXTERNAL_EXTRACTOR)
        labels_path = workspace.clusters(sample.name, config.external_extractor)
        if not features_path.exists() or not labels_path.exists():
            return None
        if labels_path.stat().st_mtime < features_path.stat().st_mtime:
            return None
        with np.load(labels_path) as data:
            if label not in data.files or "settings" not in data.files:
                return None
            if not np.array_equal(data["settings"], _settings(config)):
                return None
            labels = data[label]
        n_features = len(np.load(features_path, mmap_mode="r"))
        if len(labels) != n_features:
            return None
        found[sample.name] = labels
    return found


def save_labels(
    config: Config, workspace: Workspace, label: str, labels_by_sample: dict[str, np.ndarray]
) -> None:
    """Keep each sample's cluster labels in the workspace, beside its spot features."""
    for sample in config.samples:
        path = workspace.clusters(sample.name, config.external_extractor)
        kept: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path) as data:
                # Labels made with other settings are dropped rather than mixed with the new ones.
                if "settings" in data.files and np.array_equal(data["settings"], _settings(config)):
                    kept = {key: data[key] for key in data.files}
        kept["settings"] = _settings(config)
        kept[label] = labels_by_sample[sample.name]
        np.savez_compressed(path, **kept)


def write_labels_to_stores(
    config: Config, workspace: Workspace, label: str, labels_by_sample: dict[str, np.ndarray]
) -> None:
    """Write one clustering into the calibration and prediction stores, in their row order."""
    for role in EXTERNAL_ROLES:
        stacked = np.concatenate(
            [labels_by_sample[sample.name] for sample in config.samples_with_role(role)]
        )
        directory = store_directory(config, workspace, role)
        n_units = len(read_store_units(directory)["sample_index"])
        if len(stacked) != n_units:
            raise ValueError(
                f"the {role} store holds {n_units} spots but the spot features hold "
                f"{len(stacked)}; run the import and features stages again"
            )
        np.save(directory / f"{CLUSTER_PREFIX}{label}{CLUSTER_SUFFIX}", stacked)


def run(config: Config) -> None:
    """Run the clustering stage of an external run.

    A cluster count that was already fitted on the current spot features is not fitted again: its
    saved labels are written into the stores, so running the stage again after an ``import``
    gives exactly the same clusters.
    """
    supplied = provided_cluster_labels(config)
    if supplied:
        print(f"[cluster] cluster labels {supplied} are supplied by the user; nothing to fit")
        return

    workspace = Workspace(config)
    source = config.external_extractor

    reusable: dict[str, dict[str, np.ndarray]] = {}
    for n_clusters in config.n_clusters:
        label = f"k{n_clusters}"
        found = saved_labels(config, workspace, label)
        if found is not None:
            reusable[label] = found

    # The features are only read, and the PCA only fitted, when some cluster count is missing.
    reduced, pca, n_spots = None, None, {}
    if len(reusable) < len(config.n_clusters):
        features, batches, n_spots = load_features(config, workspace)
        reduced, pca = reduce_features(config, features, batches)
        del features

    for n_clusters in config.n_clusters:
        label = f"k{n_clusters}"
        if label in reusable:
            labels_by_sample = reusable[label]
            print(f"[cluster] {label}: already fitted on these spot features, reusing its labels")
        else:
            model_path = workspace.clustering_model(source, 1, n_clusters)
            kmeans = fit_and_save_kmeans(
                reduced, n_clusters, config.clustering.random_state, pca, model_path
            )

            labels_by_sample = {}
            offset = 0
            for sample in config.samples:
                stop = offset + n_spots[sample.name]
                labels_by_sample[sample.name] = kmeans.labels_[offset:stop].astype(np.int16)
                offset = stop
            sizes = np.bincount(kmeans.labels_, minlength=n_clusters)
            print(
                f"[cluster] {label}: KMeans fitted on all {len(reduced)} spots; cluster sizes "
                f"{sizes.min()} to {sizes.max()}"
            )

        # The stores are checked before the labels are kept, so labels of spots that no longer
        # match the stores are never saved for reuse.
        write_labels_to_stores(config, workspace, label, labels_by_sample)
        if label not in reusable:
            save_labels(config, workspace, label, labels_by_sample)
        render_cluster_maps(config, workspace, label)
