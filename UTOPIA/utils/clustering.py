"""Clustering utilities for high-dimensional data."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Optional, Union, List
from .io import save_pickle, load_pickle
import os 
def cluster_data(
    data: np.ndarray,
    n_clusters: int = 15,
    pca_threshold: int = 80,
    random_state: Optional[int] = 42,
    pca_save_path: Optional[Union[str, List[str]]] = None,
    kmeans_save_path: Optional[Union[str, List[str]]] = None,
    pca_load_path: Optional[str] = None,
    kmeans_load_path: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """
    Cluster N units with d-dimensional features using k-means.

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (N, d) where N is the number of units and d is the
        dimension of the features.
    n_clusters : int, default=15
        Number of clusters to form.
    pca_threshold : int, default=80
        Dimension threshold for applying PCA before clustering.
        If d > pca_threshold, PCA will be applied to reduce to pca_threshold dimensions.
    random_state : int or None, default=42
        Random state for reproducibility. Used for k-means and PCA.
    pca_save_path : str, list of str, or None, default=None
        Path or list of paths to save the fitted PCA instance (as pickle file).
    kmeans_save_path : str, list of str, or None, default=None
        Path or list of paths to save the fitted KMeans instance (as pickle file).
    pca_load_path : str or None, default=None
        Path to load a pre-fitted PCA instance. If provided, PCA will not be fit.
    kmeans_load_path : str or None, default=None
        Path to load a pre-fitted KMeans instance. If provided, KMeans will not be fit.
    **kwargs : dict
        Additional parameters to pass to KMeans (e.g., n_init, max_iter, tol).

    Returns
    -------
    labels : np.ndarray
        Cluster labels of shape (N,) with dtype np.uint8.

    Examples
    --------
    >>> data = np.random.rand(1000, 100)
    >>> labels = cluster_data(data, n_clusters=10, pca_threshold=80)
    >>> labels.shape
    (1000,)
    >>> labels.dtype
    dtype('uint8')

    >>> # Save fitted models
    >>> labels = cluster_data(data, n_clusters=10,
    ...                       pca_save_path='pca_model.pkl',
    ...                       kmeans_save_path='kmeans_model.pkl')

    >>> # Save to multiple paths
    >>> labels = cluster_data(data, n_clusters=10,
    ...                       pca_save_path=['pca1.pkl', 'pca2.pkl'],
    ...                       kmeans_save_path=['kmeans1.pkl', 'kmeans2.pkl'])

    >>> # Use pre-fitted models
    >>> labels = cluster_data(new_data,
    ...                       pca_load_path='pca_model.pkl',
    ...                       kmeans_load_path='kmeans_model.pkl')
    """
    
    d = data.shape[1]

    # PCA
    if pca_load_path is not None:
        pca = load_pickle(pca_load_path)
        data = pca.transform(data)
    elif d > pca_threshold:
        n_components = min(d, pca_threshold)
        pca = PCA(n_components=n_components, random_state=random_state)
        data = pca.fit_transform(data)

        if pca_save_path is not None:
            # Handle both single path and list of paths
            paths = pca_save_path if isinstance(pca_save_path, list) else [pca_save_path]
            for path in paths:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                save_pickle(pca, path)

    # KMeans
    if kmeans_load_path is not None:
        kmeans = load_pickle(kmeans_load_path)
        labels = kmeans.predict(data)
    else:
        kmeans_params = {
            'n_clusters': n_clusters,
            'random_state': random_state
        }
        # Override with user-provided kwargs
        kmeans_params.update({k: v for k, v in kwargs.items()
                             if k in ['n_init', 'max_iter', 'tol', 'algorithm']})

        kmeans = KMeans(**kmeans_params)
        labels = kmeans.fit_predict(data)

        if kmeans_save_path is not None:
            # Handle both single path and list of paths
            paths = kmeans_save_path if isinstance(kmeans_save_path, list) else [kmeans_save_path]
            for path in paths:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                save_pickle(kmeans, path)

    # Ensure output is uint8
    if labels.max() > 255:
        raise ValueError(f"Number of unique labels ({labels.max() + 1}) exceeds "
                        f"uint8 range. Consider using fewer clusters.")

    return labels.astype(np.uint8)

