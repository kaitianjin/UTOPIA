"""Clustering utilities for high-dimensional data."""

import argparse
import os
import numpy as np
from typing import Optional, Union, List

from .plot import plot_rgb_image, convert_histology_to_rgb
from ..utils.io import load_mask
from ..utils.clustering import cluster_data
import sys

def find_roi_indices(img_path: str, scale: int) -> List[int]:
    """
    Find all ROI indices i in embeddings_image_roi_i_scale_j.npy files for a specific scale.

    Args:
        img_path: Path containing ROI embedding files
        scale: Scale factor

    Returns:
        List of unique ROI indices sorted in ascending order
    """
    roi_indices = set()
    for filename in os.listdir(img_path):
        if filename.startswith(f"embeddings_roi_") and filename.endswith(f"_scale_{scale}.npy"):
            # Extract the ROI index from filename
            # Format: embeddings_image_roi_i_scale_j.npy
            try:
                parts = filename.replace("embeddings_roi_", "").replace(".npy", "").split("_scale_")
                roi_idx = int(parts[0])
                roi_indices.add(roi_idx)
            except (ValueError, IndexError):
                continue

    if not roi_indices:
        raise FileNotFoundError(f"No ROI embedding files found for scale {scale} in {img_path}")

    return sorted(list(roi_indices))


def histology_clustering(
    embedding_paths: Union[str, List[str]],
    n_clusters: int = 15,
    pca_threshold: int = 80,
    random_state: Optional[int] = 42,
    pca_save_path: Optional[Union[str, List[str]]] = None,
    kmeans_save_path: Optional[Union[str, List[str]]] = None,
    pca_load_path: Optional[str] = None,
    kmeans_load_path: Optional[str] = None,
    label_save_path: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> np.ndarray:
    """
    Load and concatenate histology embeddings, then cluster them.

    This function loads 2D numpy arrays from one or more .npy files,
    concatenates them by rows, and applies clustering using cluster_data.
    Optionally saves the labels split back to their original sizes.

    Parameters
    ----------
    embedding_paths : str or list of str
        Path or list of paths to .npy files containing embedding data.
        Each file should contain a 2D array of shape (N, d) where N is the
        number of samples and d is the embedding dimension. All arrays
        must have the same number of columns (d).
    n_clusters : int, default=15
        Number of clusters to form.
    pca_threshold : int, default=80
        Dimension threshold for applying PCA before clustering.
    random_state : int or None, default=42
        Random state for reproducibility.
    pca_save_path : str, list of str, or None, default=None
        Path or list of paths to save the fitted PCA instance.
    kmeans_save_path : str, list of str, or None, default=None
        Path or list of paths to save the fitted KMeans instance.
    pca_load_path : str or None, default=None
        Path to load a pre-fitted PCA instance.
    kmeans_load_path : str or None, default=None
        Path to load a pre-fitted KMeans instance.
    label_save_path : str, list of str, or None, default=None
        Path or list of paths to save the cluster labels. If provided, labels
        will be split back to their original sizes and saved separately.
        Must match the length of embedding_paths if both are lists.
    **kwargs : dict
        Additional parameters to pass to cluster_data/KMeans.

    Returns
    -------
    labels : np.ndarray
        Cluster labels of shape (N_total,) with dtype np.uint8,
        where N_total is the sum of all samples from all embedding files.
    """
    # Convert single path to list for uniform processing
    emb_paths = embedding_paths if isinstance(embedding_paths, list) else [embedding_paths]

    # Load and concatenate all embeddings
    embeddings_list = []
    embedding_shapes = []
    for emb_path in emb_paths:
        embeddings = np.load(emb_path)
        print(f"Loaded embeddings from {emb_path}, shape: {embeddings.shape}")
        embeddings_list.append(embeddings)
        embedding_shapes.append(embeddings.shape[0])  # Store the number of rows

    # Concatenate all embeddings by rows
    concatenated_embeddings = np.concatenate(embeddings_list, axis=0)
    print(f"Concatenated embeddings shape: {concatenated_embeddings.shape}")

    # Apply clustering using cluster_data
    labels = cluster_data(
        data=concatenated_embeddings,
        n_clusters=n_clusters,
        pca_threshold=pca_threshold,
        random_state=random_state,
        pca_save_path=pca_save_path,
        kmeans_save_path=kmeans_save_path,
        pca_load_path=pca_load_path,
        kmeans_load_path=kmeans_load_path,
        **kwargs
    )

    labels = labels.astype(np.uint8)

    # Save labels if save path is provided
    if label_save_path is not None:
        save_paths = label_save_path if isinstance(label_save_path, list) else [label_save_path]

        # Validate that save paths match embedding paths
        if len(save_paths) != len(emb_paths):
            raise ValueError(
                f"Number of label save paths ({len(save_paths)}) must match "
                f"number of embedding paths ({len(emb_paths)})"
            )

        # Split labels back to original sizes and save
        start_idx = 0
        for save_path, size in zip(save_paths, embedding_shapes):
            end_idx = start_idx + size
            split_labels = labels[start_idx:end_idx]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, split_labels)
            print(f"Saved labels to {save_path}, shape: {split_labels.shape}")
            start_idx = end_idx

    return labels


def create_he_clusters_image(he_clusters: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create a 2D cluster image from 1D cluster labels and a boolean mask.

    This function takes 1D cluster labels and maps them back to their 2D spatial
    positions according to a boolean mask, filling non-masked positions with -1.

    Parameters
    ----------
    he_clusters : np.ndarray
        1D array of cluster labels with shape (N,), where N is the number of
        True values in the mask.
    mask : np.ndarray
        2D boolean array of shape (W, L) indicating valid positions. True values
        indicate positions where cluster labels should be placed.

    Returns
    -------
    he_clusters_image : np.ndarray
        2D array of shape (W, L) containing cluster labels at masked positions
        and -1 at non-masked positions.

    Examples
    --------
    >>> he_clusters = np.array([0, 1, 2, 0, 1])
    >>> mask = np.array([[True, False, True],
    ...                  [True, True, False],
    ...                  [False, True, False]])
    >>> img = create_he_clusters_image(he_clusters, mask)
    >>> print(img)
    [[ 0. -1.  1.]
     [ 2.  0. -1.]
     [-1.  1. -1.]]
    """
    he_clusters_image = np.ones(mask.shape) * -1
    he_clusters_image[mask] = he_clusters
    return he_clusters_image


def run_clustering(data_paths,
                   scale,
                   wsi,
                   roi,
                   pca_threshold,
                   n_clusters,
                   delete_raw_features=False):
    """
    Run clustering based on command-line arguments.

    Parameters
    ----------
    data_paths : list of str
        Path(s) to directories containing embedding files
    scale : int
        Scale factor
    roi : bool
        Process ROI embeddings
    pca_threshold : int
        PCA dimension threshold
    n_clusters : int
        Number of clusters
    wsi : bool
        Process WSI (whole slide image) embeddings
    delete_raw_features : bool, default=False
        If True and scale != 1, delete the raw embedding files after clustering
    """
    if wsi:
        # Process full image embeddings
        print(f"Processing full image embeddings at scale {scale}")

        # Build embedding paths
        embedding_paths = []
        pca_save_paths = []
        kmeans_save_paths = []
        label_save_paths = []

        for path in data_paths:

            emb_file = os.path.join(path, "embeddings", f"embeddings_scale_{scale}.npy")
            embedding_paths.append(emb_file)
            os.makedirs(os.path.join(path, "histology_clustering"), exist_ok=True)
            pca_save_paths.append(os.path.join(path, "histology_clustering", f"pca_dim_{pca_threshold}_scale_{scale}.pkl"))
            kmeans_save_paths.append(os.path.join(path, "histology_clustering", f"kmeans_dim_{pca_threshold}_scale_{scale}_num_cls_{n_clusters}.pkl"))
            label_save_paths.append(os.path.join(path, "histology_clustering", f"he_clusters_scale_{scale}_num_{n_clusters}.npy"))

        # Run clustering
        labels = histology_clustering(
            embedding_paths=embedding_paths,
            n_clusters=n_clusters,
            pca_threshold=pca_threshold,
            pca_save_path=pca_save_paths,
            kmeans_save_path=kmeans_save_paths,
            label_save_path=label_save_paths
        )

        print(f"Clustering complete! Generated {len(labels)} labels")
        print(f"Saved PCA models to: {pca_save_paths}")
        print(f"Saved KMeans models to: {kmeans_save_paths}")
        print(f"Saved cluster labels to: {label_save_paths}")

        # Generate and save he_clusters_image for each image path
        for path, label_path in zip(data_paths, label_save_paths):
            # Load the mask
            mask_path = os.path.join(path, "mask", f"mask-small_scale_{scale}.png")
            mask = load_mask(mask_path)
            he_clusters = np.load(label_path)
            he_clusters_image = create_he_clusters_image(he_clusters, mask)
            image_save_path = os.path.join(path, "histology_clustering", f"he_clusters_image_scale_{scale}_num_{n_clusters}.npy")
            np.save(image_save_path, he_clusters_image)
            he_clusters_image_rgb = convert_histology_to_rgb(he_clusters_image, n_clusters)
            plot_rgb_image(he_clusters_image_rgb, 
                           save_path=os.path.join(path, "histology_clustering", f"he_clusters_image_scale_{scale}_num_{n_clusters}.png"))
            print(f"Saved cluster image to: {image_save_path}")


        # Delete raw embedding files if requested and scale != 1
        if delete_raw_features and scale != 1:
            for emb_path in embedding_paths:
                if os.path.exists(emb_path):
                    os.remove(emb_path)
                    print(f"Deleted raw embedding file: {emb_path}")

    if roi:
        # Process ROI embeddings
        print(f"Processing ROI embeddings at scale {scale}")

        for path in data_paths:
            # Find ROI indices
            embedding_path = os.path.join(path, "embeddings")
            roi_indices = find_roi_indices(embedding_path, scale)
            
            # Load PCA and KMeans models
            pca_load_path = os.path.join(path, "histology_clustering", f"pca_dim_{pca_threshold}_scale_{scale}.pkl")
            kmeans_load_path = os.path.join(path, "histology_clustering", f"kmeans_dim_{pca_threshold}_scale_{scale}_num_cls_{n_clusters}.pkl")

            # Process each ROI separately
            for roi_idx in roi_indices:
                emb_file = os.path.join(embedding_path, f"embeddings_roi_{roi_idx}_scale_{scale}.npy")
                mask_path = os.path.join(path, "mask", f"mask-small_roi_{roi_idx}_scale_{scale}.png")
                embeddings_2d = np.load(emb_file)
                mask = load_mask(mask_path)
                # Apply clustering
                labels = cluster_data(
                    data=embeddings_2d,
                    n_clusters=n_clusters,
                    pca_threshold=pca_threshold,
                    pca_load_path=pca_load_path,
                    kmeans_load_path=kmeans_load_path
                )

                # Save labels
                label_save_path = os.path.join(path, "histology_clustering", f"he_clusters_roi_{roi_idx}_scale_{scale}_num_{n_clusters}.npy")
                np.save(label_save_path, labels)
                print(f"Saved ROI {roi_idx} cluster labels to {label_save_path}, shape: {labels.shape}")

                # Create and save the cluster image
                he_clusters_image = create_he_clusters_image(labels, mask)
                image_save_path = os.path.join(path, "histology_clustering", f"he_clusters_image_roi_{roi_idx}_scale_{scale}_num_{n_clusters}.npy")
                np.save(image_save_path, he_clusters_image)
                he_clusters_image_rgb = convert_histology_to_rgb(he_clusters_image, n_clusters)
                plot_rgb_image(he_clusters_image_rgb, 
                               save_path=os.path.join(path, "histology_clustering", f"he_clusters_image_roi_{roi_idx}_scale_{scale}_num_{n_clusters}.png"))
                print(f"Saved ROI cluster image to: {image_save_path}")

                # Delete raw ROI embedding file if requested and scale != 1
                if delete_raw_features and scale != 1:
                    if os.path.exists(emb_file):
                        os.remove(emb_file)
                        print(f"Deleted raw embedding file: {emb_file}")

            print(f"ROI clustering complete for {path}!")


def run_clustering_multi_scale(data_paths,
                               scales,
                               wsi,
                               roi,
                               pca_threshold,
                               n_clusters,
                               delete_raw_features=False):
    """
    Run clustering for multiple scales.

    Parameters
    ----------
    data_paths : list of str
        Path(s) to directories containing embedding files
    scales : list of int
        List of scale factors to process
    roi : bool
        Process ROI embeddings
    pca_threshold : int
        PCA dimension threshold
    n_clusters : int
        Number of clusters
    wsi : bool
        Process WSI (whole slide image) embeddings
    delete_raw_features : bool, default=False
        If True and scale != 1, delete the raw embedding files after clustering
    """
    for scale in scales:
        print(f"\n{'='*50}")
        print(f"Processing scale {scale}")
        print(f"{'='*50}")
        run_clustering(
            data_paths=data_paths,
            scale=scale,
            wsi=wsi,
            roi=roi,
            pca_threshold=pca_threshold,
            n_clusters=n_clusters,
            delete_raw_features=delete_raw_features
        )


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Cluster histology embeddings using PCA and KMeans'
    )
    parser.add_argument(
        '--data_paths',
        nargs='+',
        help='Path(s) to directories containing embedding files'
    )
    parser.add_argument(
        '--scale',
        type=int,
        nargs='+',
        default=[1],
        help='Scale factor(s) (default: 1). Can specify multiple scales.'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        default=False,
        help='Delete raw embedding files after clustering (only for scale != 1)'
    )
    parser.add_argument(
        '--roi',
        action='store_true',
        help='Process ROI embeddings instead of full image embeddings'
    )
    parser.add_argument(
        '--wsi',
        action='store_true',
        help='Process WSI (whole slide image) embeddings'
    )
    parser.add_argument(
        '--pca_threshold',
        type=int,
        default=80,
        help='PCA dimension threshold (default: 80)'
    )
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=15,
        help='Number of clusters (default: 15)'
    )

    args = parser.parse_args()

    # Validate arguments
    for scale in args.scale:
        if scale <= 0:
            parser.error("--scale values must be positive integers")
    if args.pca_threshold <= 0:
        parser.error("--pca_threshold must be a positive integer")
    if args.n_clusters <= 0:
        parser.error("--n_clusters must be a positive integer")

    run_clustering_multi_scale(
        data_paths=args.data_paths,
        scales=args.scale,
        roi=args.roi,
        wsi=args.wsi,
        pca_threshold=args.pca_threshold,
        n_clusters=args.n_clusters,
        delete_raw_features=args.clean
    )


if __name__ == "__main__":
    sys.exit(main())