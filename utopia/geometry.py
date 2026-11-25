"""Per-sample spatial bookkeeping.

Per-unit data are flat ``(N, ...)`` arrays in mask order, the order of ``numpy.argwhere(mask)``.
A sample carries only lattice-sized state: the tissue mask, the ROIs and one mask per modality of
the superpixels its ``positions.npy`` lists. A role's usable units are its region within the
tissue mask, further limited for cell types (see :func:`limited_to_positions`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .io import PathLike


def limited_to_positions(modality: str) -> bool:
    """Whether a modality has ground truth only at the positions its ``positions.npy`` lists.

    A cell type is known only where it was annotated, whereas a tissue superpixel that
    ``genes/positions.npy`` does not list was measured and holds zero counts.
    """
    return modality == "cell_type"


@dataclass(frozen=True)
class Roi:
    """A rectangular region of interest, always in scale-1 superpixel coordinates.

    An ROI configured in raw H&E pixels is converted before it becomes a :class:`Roi`.
    """

    top_left: tuple[int, int]  # (row, col) of the top-left superpixel
    height: int  # in superpixels
    width: int  # in superpixels
    roles: tuple[str, ...]  # subset of {"train", "calib"}

    @property
    def box(self) -> tuple[int, int, int, int]:
        """The ROI as ``(row_start, col_start, row_stop, col_stop)``."""
        row, column = self.top_left
        return row, column, row + self.height, column + self.width


@dataclass
class Geometry:
    """Everything the pipeline needs to know about one sample's spatial layout."""

    name: str
    lattice_shape: tuple[int, int]  # (R, C) of the scale-1 superpixel lattice
    tissue_mask: np.ndarray  # (R, C) bool
    rois: list[Roi] = field(default_factory=list)
    # Modality name -> (R, C) bool mask of the superpixels listed in its ``positions.npy``.
    measured: dict[str, np.ndarray] = field(default_factory=dict)

    def role_region(self, role: str) -> np.ndarray:
        """Region of the sample playing ``role``; ``test`` is everything outside all ROIs."""
        outside_all_rois = role == "test"
        region = np.zeros(self.lattice_shape, dtype=bool)
        for roi in self.rois:
            if outside_all_rois or role in roi.roles:
                row0, col0, row1, col1 = roi.box
                region[row0:row1, col0:col1] = True
        return ~region if outside_all_rois else region

    def unit_mask(self, role: str, modality: str | None = None) -> np.ndarray:
        """Usable unit mask for ``role``: its region within the tissue mask.

        A modality limited to its listed positions is intersected with them as well; with
        ``modality`` None no modality is considered.
        """
        mask = self.role_region(role) & self.tissue_mask
        if modality is not None and limited_to_positions(modality):
            mask = mask & self.measured[modality]
        return mask

    def feature_rows(self, positions: np.ndarray, cover_lattice: bool = False) -> np.ndarray:
        """Row numbers of ``positions`` in this sample's feature array.

        Features hold one row per tissue superpixel in mask order, or with ``cover_lattice``
        (``very_slow``) one row per lattice superpixel in row-major order.
        """
        if cover_lattice:
            n_columns = self.lattice_shape[1]
            return positions[:, 0].astype(np.int64) * n_columns + positions[:, 1]
        # Lookup from lattice position to row of the mask-ordered features, -1 outside the tissue.
        index = np.full(self.lattice_shape, -1, dtype=np.int32)
        index[self.tissue_mask] = np.arange(int(self.tissue_mask.sum()), dtype=np.int32)
        return index[positions[:, 0], positions[:, 1]]

    def save(self, path: PathLike) -> None:
        """Write the geometry to ``path`` as a compressed ``.npz`` archive."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "tissue_mask": self.tissue_mask,
            "roi_boxes": np.array([roi.box for roi in self.rois], dtype=np.int32).reshape(-1, 4),
            "roi_roles": np.array([";".join(roi.roles) for roi in self.rois], dtype="U32"),
            "lattice_shape": np.array(self.lattice_shape, dtype=np.int32),
            "name": np.array(self.name),
        }
        for modality, mask in self.measured.items():
            arrays[f"measured_{modality}"] = mask
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: PathLike) -> "Geometry":
        """Read a geometry written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            boxes = data["roi_boxes"]
            roles = data["roi_roles"]
            rois = [
                Roi(
                    top_left=(int(box[0]), int(box[1])),
                    height=int(box[2] - box[0]),
                    width=int(box[3] - box[1]),
                    roles=tuple(str(role).split(";")),
                )
                for box, role in zip(boxes, roles)
            ]
            measured = {
                key[len("measured_") :]: data[key]
                for key in data.files
                if key.startswith("measured_")
            }
            return cls(
                name=str(data["name"]),
                lattice_shape=(int(data["lattice_shape"][0]), int(data["lattice_shape"][1])),
                tissue_mask=data["tissue_mask"],
                rois=rois,
                measured=measured,
            )


def positions_to_mask(positions: np.ndarray, lattice_shape: tuple[int, int]) -> np.ndarray:
    """Build a boolean lattice mask marking the given ``(N, 2)`` positions."""
    mask = np.zeros(lattice_shape, dtype=bool)
    mask[positions[:, 0], positions[:, 1]] = True
    return mask


def positions_to_rows(positions: np.ndarray, lattice_shape: tuple[int, int]) -> np.ndarray:
    """``(R, C)`` lookup holding ``i`` at ``positions[i]`` and ``-1`` everywhere else."""
    index = np.full(lattice_shape, -1, dtype=np.int32)
    index[positions[:, 0], positions[:, 1]] = np.arange(len(positions), dtype=np.int32)
    return index


def gather_units(
    geometries: Sequence[Geometry], role: str, modality: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the usable units of several samples into ``(sample_index, positions)``.

    Rows are ordered by sample (its index in ``geometries``) and then by mask order.
    """
    sample_index: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    for index, geometry in enumerate(geometries):
        found = np.argwhere(geometry.unit_mask(role, modality)).astype(np.int32)
        positions.append(found)
        sample_index.append(np.full(len(found), index, dtype=np.int16))
    if not positions:
        return np.zeros(0, dtype=np.int16), np.zeros((0, 2), dtype=np.int32)
    return np.concatenate(sample_index), np.concatenate(positions)
