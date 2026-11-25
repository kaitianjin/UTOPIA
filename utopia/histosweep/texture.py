"""GLCM texture refinement of the density filter.

Gives the same mask as the reference HistoSweep implementation, superpixel for superpixel, but
computes the co-occurrence statistics of a whole block of superpixels at once. One quirk is kept on
purpose to that end: the grey image wraps modulo 256 before it is quantised (see
:func:`grey_levels`).
"""

from __future__ import annotations

import numpy as np


def grey_levels(image: np.ndarray, levels: int = 64) -> np.ndarray:
    """Quantise an ``(H, W, 3)`` float32 H&E image with values in ``[0, 255]`` to grey levels.

    Returns ``(H, W)`` uint8 values in ``[0, levels)``.
    """
    from skimage.color import rgb2gray

    grey = rgb2gray(image)
    # rgb2gray does not rescale a float image, so this product wraps modulo 256 in uint8; kept
    # to match the reference HistoSweep implementation.
    grey = (grey * 255).astype(np.uint8)
    return (grey / 255 * (levels - 1)).astype(np.uint8)


def glcm_features(patches: np.ndarray, levels: int = 64, block: int = 2048) -> np.ndarray:
    """Energy, homogeneity and entropy of each patch's grey-level co-occurrence matrix.

    Takes ``(N, h, w)`` quantised uint8 patches and returns ``(N, 3)`` float64; the matrix is the
    symmetric, normalised, distance-1 horizontal one, as ``skimage``'s ``graycomatrix`` builds it.
    """
    n_patches = len(patches)
    features = np.empty((n_patches, 3), dtype=np.float64)

    indices = np.arange(levels)
    homogeneity_weights = 1.0 / (1.0 + (indices[:, None] - indices[None, :]) ** 2.0)

    for start in range(0, n_patches, block):
        chunk = patches[start : start + block].astype(np.int64)
        size = len(chunk)
        pairs = (chunk[:, :, :-1] * levels + chunk[:, :, 1:]).reshape(size, -1)
        offsets = (np.arange(size, dtype=np.int64)[:, None] * levels * levels + pairs).ravel()
        counts = np.bincount(offsets, minlength=size * levels * levels)
        counts = counts.reshape(size, levels, levels)

        symmetric = (counts + counts.transpose(0, 2, 1)).astype(np.float64)
        probability = symmetric / symmetric.sum(axis=(1, 2), keepdims=True)

        energy = np.sqrt((probability**2).sum(axis=(1, 2)))
        homogeneity = (probability * homogeneity_weights).sum(axis=(1, 2))
        log_probability = np.log2(
            probability, out=np.zeros_like(probability), where=probability > 0
        )
        entropy = -(probability * log_probability).sum(axis=(1, 2))

        features[start : start + size] = np.stack([energy, homogeneity, entropy], axis=1)
    return features


def colour_reject(patches: np.ndarray) -> np.ndarray:
    """Flag patches whose average colour cannot be stained tissue: green, grey or near-white."""
    means = patches.reshape(len(patches), -1, 3).mean(axis=1)
    red, green, blue = means[:, 0], means[:, 1], means[:, 2]
    is_green = (green > red + 20) & (green > blue + 20)
    is_grey = means.std(axis=1) < 10
    is_bright = (red > 230) & (green > 230) & (blue > 230)
    return is_green | is_grey | is_bright


def refine_by_texture(
    image: np.ndarray,
    background_candidates: np.ndarray,
    patch_size: int = 16,
    levels: int = 64,
    random_state: int = 45,
) -> np.ndarray:
    """Return the ``(R, C)`` background-candidate mask with well-textured superpixels cleared.

    Candidates whose colour could be tissue are clustered into three groups by GLCM statistics, and
    the two groups with the lowest ``energy + homogeneity - entropy`` are reclaimed as tissue.
    """
    from sklearn.mixture import GaussianMixture

    from .metrics import patchify

    refined = np.asarray(background_candidates, dtype=bool).copy()
    candidate_rows = np.flatnonzero(refined.ravel())
    if len(candidate_rows) < 3:
        return refined

    rgb_patches, _ = patchify(image, patch_size)
    scored_rows = candidate_rows[~colour_reject(rgb_patches)[candidate_rows]]
    if len(scored_rows) < 3:
        return refined

    grey_patches, _ = patchify(grey_levels(image, levels)[..., None], patch_size)
    features = glcm_features(grey_patches[scored_rows, :, :, 0], levels=levels)
    span = features.max(axis=0) - features.min(axis=0)
    normalised = (features - features.min(axis=0)) / np.where(span > 0, span, 1.0)

    # Columns are clustered in the order homogeneity, energy, entropy, kept to match the reference
    # HistoSweep implementation: the order changes the last bit of every summed distance.
    labels = GaussianMixture(n_components=3, random_state=random_state).fit_predict(
        normalised[:, [1, 0, 2]]
    )
    scores = np.array(
        [
            normalised[labels == group][:, 0].mean()
            + normalised[labels == group][:, 1].mean()
            - normalised[labels == group][:, 2].mean()
            for group in range(3)
        ]
    )
    textured = np.isin(labels, np.argsort(scores, kind="stable")[:2])

    refined.ravel()[scored_rows[textured]] = False
    return refined
