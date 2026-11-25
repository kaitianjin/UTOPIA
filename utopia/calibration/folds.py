"""Spatially blocked cross-validation folds.

Folds are contiguous rectangles of superpixels; only a fold's *centre*, inset from the fold by a
safe boundary, is used for calibration, so calibration units are not neighbours of training data.
A block is subdivided while removing it would leave some histology cluster under-represented.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: A rectangle on the superpixel lattice, ``(row0, col0, row1, col1)``, half-open.
Box = tuple[int, int, int, int]
#: Which sides of a box lie on a true outer boundary, ``(left, right, top, bottom)``.
Sides = tuple[bool, bool, bool, bool]

#: Fraction of every cluster that must remain in the training set for a fold to
#: be accepted without further subdivision.
MIN_CLUSTER_RETENTION = 0.8


def tile_box(
    box: Box, width: int, sides: Sides, safe_boundary: int
) -> list[tuple[Box, Box, Sides]] | None:
    """Cover a box with overlapping folds whose centres tile it exactly.

    Returns ``(fold_box, centre_box, centre_sides)`` triples, or None when the box is too small to
    tile at this width; sides on a true outer boundary are not inset.
    """
    row0, col0, row1, col1 = box
    check_left, check_right, check_top, check_bottom = sides
    step = width - 2 * safe_boundary
    if step <= 0:
        return None
    if (col1 - col0) - safe_boundary * ((not check_left) + (not check_right)) <= 0:
        return None
    if (row1 - row0) - safe_boundary * ((not check_top) + (not check_bottom)) <= 0:
        return None

    tiles: list[tuple[Box, Box, Sides]] = []
    for top in range(row0, row1, step):
        bottom = min(top + width, row1)
        centre_bottom = bottom
        for left in range(col0, col1, step):
            right = min(left + width, col1)
            touches = (
                bool(left == col0 and check_left),
                bool(right == col1 and check_right),
                bool(top == row0 and check_top),
                bool(bottom == row1 and check_bottom),
            )
            centre_left = left + (0 if touches[0] else safe_boundary)
            centre_right = right - (0 if touches[1] else safe_boundary)
            centre_top = top + (0 if touches[2] else safe_boundary)
            centre_bottom = bottom - (0 if touches[3] else safe_boundary)
            if centre_right > centre_left and centre_bottom > centre_top:
                fold = (top, left, bottom, right)
                centre = (centre_top, centre_left, centre_bottom, centre_right)
                tiles.append((fold, centre, touches))
            if centre_right == col1:
                break
        if centre_bottom == row1:
            break
    return tiles or None


def _cluster_counts(labels: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Count occurrences of each label in ``wanted`` within ``labels``."""
    present, counts = np.unique(labels, return_counts=True)
    lookup = dict(zip(present.tolist(), counts.tolist()))
    return np.array([lookup.get(int(label), 0) for label in wanted], dtype=np.int64)


def generate_folds(
    cluster_image: np.ndarray,
    observed_mask: np.ndarray,
    roi_box: Box,
    fold_widths: Sequence[int] = (200, 100, 50),
    safe_boundary: int = 16,
) -> tuple[list[Box], list[Box]]:
    """Generate cross-validation folds for one region of interest.

    Returns ``(fold_boxes, centre_boxes)`` of equal length; ``fold_widths`` are tried from coarsest
    to finest, and centres of folds from different parents may overlap.
    """
    row0, col0, row1, col1 = roi_box
    observed = np.zeros(observed_mask.shape, dtype=bool)
    observed[row0:row1, col0:col1] = observed_mask[row0:row1, col0:col1]

    labels = np.unique(cluster_image[observed])
    totals = _cluster_counts(cluster_image[observed], labels)

    fold_boxes: list[Box] = []
    centre_boxes: list[Box] = []

    def descend(box: Box, centre: Box, sides: Sides, depth: int) -> None:
        """Accept ``box`` as a fold, or split it further and recurse."""
        if depth > 0:
            top, left, bottom, right = box
            box_labels = cluster_image[top:bottom, left:right]
            box_observed = observed[top:bottom, left:right]
            inside = _cluster_counts(box_labels[box_observed], labels)
            retained = (totals - inside) / np.maximum(totals, 1)
            if depth >= len(fold_widths) or np.all(retained >= MIN_CLUSTER_RETENTION):
                fold_boxes.append(box)
                centre_boxes.append(centre)
                return

        tiles = tile_box(box, fold_widths[depth], sides, safe_boundary)
        if tiles is None:
            fold_boxes.append(box)
            centre_boxes.append(centre)
            return
        for sub_box, sub_centre, sub_sides in tiles:
            descend(sub_box, sub_centre, sub_sides, depth + 1)

    if not fold_widths:
        return [roi_box], [roi_box]
    descend(roi_box, roi_box, (True, True, True, True), 0)
    return fold_boxes, centre_boxes


def paint_fold_ids(
    shape: tuple[int, int], centre_boxes: Sequence[Box], first_id: int = 0
) -> np.ndarray:
    """Render centre boxes into an int32 lattice image of fold ids, ``-1`` where none covers.

    Later boxes overwrite earlier ones, so a superpixel in several centres gets exactly one fold.
    """
    image = np.full(shape, -1, dtype=np.int32)
    for offset, (row0, col0, row1, col1) in enumerate(centre_boxes):
        image[row0:row1, col0:col1] = first_id + offset
    return image


def boxes_to_array(boxes: Sequence[Box]) -> np.ndarray:
    """Pack boxes into an ``(n, 4)`` int32 array."""
    return np.array(list(boxes), dtype=np.int32).reshape(-1, 4)


def inside_box(positions: np.ndarray, box: Box) -> np.ndarray:
    """Return a boolean mask of the positions falling inside ``box``."""
    row0, col0, row1, col1 = box
    return (
        (positions[:, 0] >= row0)
        & (positions[:, 0] < row1)
        & (positions[:, 1] >= col0)
        & (positions[:, 1] < col1)
    )
