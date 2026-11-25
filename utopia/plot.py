"""Rendering per-unit values back onto the slide.

A confidence map colours the test units by confidence on a heat scale and the calibration
units by their measured truth, so one picture shows both the inference and its evidence. Units
are lattice cells (built-in superpixels), or disks or squares of a given radius (external spots).
An external map can also show the tissue mask behind the spots, in grey.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .io import PathLike

#: Painted where a confidence map has no tissue: ``hot`` runs black -> red -> white, so a mid grey
#: stays apart from both ends. Magnitude maps keep white, which ``turbo`` never reaches.
CONFIDENCE_BACKGROUND = (172, 172, 172)  # #ACACAC
#: Painted where a confidence map has tissue but no unit: darker than the background, and
#: still apart from both ends of ``hot``.
CONFIDENCE_TISSUE = (100, 100, 100)
#: Painted where a magnitude map has tissue but no unit: a light grey on the white background.
MAGNITUDE_TISSUE = (225, 225, 225)

#: Distinct colours for histology cluster maps.
CLUSTER_COLOURS = np.array(
    [
        [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189],
        [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34],
        [23, 190, 207], [174, 199, 232], [255, 187, 120], [152, 223, 138],
        [255, 152, 150], [197, 176, 213], [196, 156, 148], [247, 182, 210],
        [199, 199, 199], [219, 219, 141], [158, 218, 229], [16, 60, 90],
        [128, 64, 7], [22, 80, 22], [107, 20, 20], [74, 52, 94], [70, 43, 38],
        [114, 60, 97], [64, 64, 64], [94, 94, 17], [12, 95, 104], [0, 0, 0],
    ],
    dtype=np.uint8,
)


def _offsets(radius: float, square: bool) -> np.ndarray:
    """``(P, 2)`` integer offsets covering a disk of ``radius`` around a centre, in row-major order.

    With ``square`` they cover the square of half-width ``radius`` instead.
    """
    span = int(np.floor(radius))
    grid = np.arange(-span, span + 1)
    rows, columns = np.meshgrid(grid, grid, indexing="ij")
    if square:
        return np.stack([rows.ravel(), columns.ravel()], axis=1)
    inside = rows**2 + columns**2 <= radius**2
    return np.stack([rows[inside], columns[inside]], axis=1)


def paint(
    values: np.ndarray,
    positions: np.ndarray,
    shape: tuple[int, int],
    radius: np.ndarray | float,
    square: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw per-unit values onto a canvas, returning ``(image, mask of the cells covered)``.

    Each unit is one cell, or a filled disk when ``radius`` (scalar or per unit) is 1 or more;
    with ``square`` it is a filled square of half-width ``radius`` instead of a disk.
    """
    image = np.zeros(shape, dtype=float)
    drawn = np.zeros(shape, dtype=bool)
    if len(positions) == 0:
        return image, drawn

    radii = np.broadcast_to(np.asarray(radius, dtype=float).ravel(), (len(positions),))
    for value in np.unique(radii):
        rows = radii == value
        centres = positions[rows]
        if value < 1.0:
            offsets = np.zeros((1, 2), dtype=int)
        else:
            offsets = _offsets(value, square)
        cells = centres[:, None, :] + offsets[None, :, :]
        cells = cells.reshape(-1, 2)
        repeated = np.repeat(np.asarray(values, dtype=float)[rows], len(offsets))
        inside = (
            (cells[:, 0] >= 0) & (cells[:, 0] < shape[0])
            & (cells[:, 1] >= 0) & (cells[:, 1] < shape[1])
        )
        cells, repeated = cells[inside], repeated[inside]
        image[cells[:, 0], cells[:, 1]] = repeated
        drawn[cells[:, 0], cells[:, 1]] = True
    return image, drawn


def _colourise(values: np.ndarray, colormap: str) -> np.ndarray:
    """Map values in ``[0, 1]`` through a matplotlib colormap to uint8 RGB."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgba = plt.get_cmap(colormap)(np.clip(values, 0.0, 1.0))
    return (rgba[..., :3] * 255).astype(np.uint8)


def confidence_map(
    confidence: np.ndarray,
    drawn: np.ndarray,
    truth: np.ndarray | None = None,
    truth_drawn: np.ndarray | None = None,
    colormap: str = "hot",
    truth_colormap: str = "turbo",
    background: Sequence[int] = CONFIDENCE_BACKGROUND,
    tissue: np.ndarray | None = None,
    tissue_colour: Sequence[int] = CONFIDENCE_TISSUE,
) -> np.ndarray:
    """Colour the test units by confidence and the calibration units by truth, as uint8 RGB.

    Inputs are ``(R, C)`` images in ``[0, 1]`` with their drawn masks, truth painted last; an
    optional boolean ``tissue`` mask is painted in ``tissue_colour`` underneath the units.
    """
    image = np.empty((*confidence.shape, 3), dtype=np.uint8)
    image[:] = np.asarray(background, dtype=np.uint8)
    if tissue is not None:
        image[tissue] = np.asarray(tissue_colour, dtype=np.uint8)
    image[drawn] = _colourise(confidence, colormap)[drawn]
    if truth is not None and truth_drawn is not None:
        image[truth_drawn] = _colourise(truth, truth_colormap)[truth_drawn]
    return image


def magnitude_map(
    values: np.ndarray,
    drawn: np.ndarray,
    truth: np.ndarray | None = None,
    truth_drawn: np.ndarray | None = None,
    colormap: str = "turbo",
    tissue: np.ndarray | None = None,
) -> np.ndarray:
    """Colour test units by predicted magnitude and calibration units by truth, on white.

    ``tissue`` is an optional boolean tissue mask, painted light grey underneath the units.
    """
    return confidence_map(
        values, drawn, truth, truth_drawn, colormap=colormap, truth_colormap=colormap,
        background=(255, 255, 255), tissue=tissue, tissue_colour=MAGNITUDE_TISSUE,
    )


def cluster_map(labels: np.ndarray, background: int = 255) -> np.ndarray:
    """Colour a histology cluster label image; cells labelled ``background`` are drawn white."""
    is_cluster = labels != background
    if is_cluster.any():
        n_clusters = int(labels[is_cluster].max()) + 1
    else:
        n_clusters = 0
    if n_clusters > len(CLUSTER_COLOURS):
        raise ValueError(
            f"{n_clusters} clusters exceed the {len(CLUSTER_COLOURS)} available colours; "
            "extend CLUSTER_COLOURS in utopia/plot.py"
        )
    image = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    for cluster in range(n_clusters):
        image[labels == cluster] = CLUSTER_COLOURS[cluster]
    return image


def save_fdp_curve(
    fdr: Sequence[float],
    fdp: Sequence[float],
    n_discoveries: Sequence[int],
    path: PathLike,
) -> None:
    """Plot realised false-discovery proportion against the nominal level.

    Levels with fewer than 1% of the largest discovery count are dropped as too noisy.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.asarray(n_discoveries)
    fewest_shown = max(1, int(0.01 * counts.max()))
    keep = counts >= fewest_shown
    figure, axis = plt.subplots(figsize=(4.5, 4.5))
    axis.plot(np.asarray(fdr)[keep], np.asarray(fdp)[keep], linewidth=2, label="realised FDP")
    axis.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="nominal")
    axis.set_xlabel("target FDR")
    axis.set_ylabel("realised FDP")
    axis.grid(alpha=0.3)
    axis.legend(loc="upper left")
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
