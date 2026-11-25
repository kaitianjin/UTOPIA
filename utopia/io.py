"""File primitives: images, JSON, YAML, and the gene-major on-demand prediction store.

The non-obvious piece is :class:`TargetStore`: predictions for all genes live in one ``.npy``
file laid out **gene-major** ``(T, N)``, one row per target channel (gene or cell type) and one
column per spatial unit, so reading one gene touches one contiguous row.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PathLike = Union[str, os.PathLike]


def read_json(path: PathLike) -> Any:
    """Return the object stored in the JSON file at ``path``."""
    with open(path) as handle:
        return json.load(handle)


def write_json(obj: Any, path: PathLike) -> None:
    """Write ``obj`` to ``path`` as JSON, creating parent directories."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2)


def read_yaml_as_text(path: PathLike) -> Any:
    """Return the object stored in the YAML file at ``path``, with every value kept as text.

    A plain YAML reader would turn an unquoted ``ON`` or ``NO`` into a boolean and ``1`` into a
    number, which is wrong for gene and cell-type names; the caller converts the known numbers.
    """
    with open(path) as handle:
        return yaml.load(handle, Loader=yaml.BaseLoader)


def load_image(path: PathLike) -> np.ndarray:
    """Load an image into an ``(H, W, 3)`` uint8 array, dropping any alpha channel.

    Handles Pillow formats plus ``.svs`` (openslide) and ``.btf`` / ``.ome.tif`` (tifffile).
    """
    suffix = "".join(Path(path).suffixes[-2:]).lower()
    if suffix.endswith(".svs"):
        import openslide

        slide = openslide.OpenSlide(str(path))
        img = np.array(slide.read_region((0, 0), 0, slide.level_dimensions[0]).convert("RGB"))
    elif suffix.endswith(".btf") or suffix.endswith(".ome.tif"):
        import tifffile

        img = np.asarray(tifffile.imread(str(path)))
    else:
        img = np.asarray(Image.open(path))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    return img


def save_image(img: np.ndarray, path: PathLike) -> None:
    """Save an ``(H, W)`` or ``(H, W, 3)`` uint8 array as an image file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(path)


def load_mask(path: PathLike) -> np.ndarray:
    """Load a boolean mask image (any non-zero pixel is ``True``)."""
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask.any(axis=2)
    return mask > 0


def save_mask(mask: np.ndarray, path: PathLike) -> None:
    """Save a boolean mask as a 0/255 PNG."""
    save_image(np.asarray(mask, dtype=bool).astype(np.uint8) * 255, path)


def create_memmap(path: PathLike, shape: Sequence[int], dtype: Any) -> np.memmap:
    """Create a zero-filled ``.npy`` file and return it opened for writing."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        str(path), mode="w+", dtype=np.dtype(dtype), shape=tuple(shape)
    )


def open_memmap(path: PathLike, mode: str = "r") -> np.memmap:
    """Open an existing ``.npy`` file as a memory map."""
    return np.lib.format.open_memmap(str(path), mode=mode)


class TargetStore:
    """A memory-mapped ``(T, N)`` array of per-unit values, one row per channel, plus its metadata.

    The array is ``<directory>/<name>.npy`` (``(T, K, N)`` when it stacks K values per unit) and
    the directory's ``meta.json`` names the channels.
    """

    def __init__(self, directory: PathLike, name: str, meta: dict[str, Any]) -> None:
        """Refer to the array ``<directory>/<name>.npy`` and the directory's metadata."""
        self.directory = Path(directory)
        self.name = name
        self.meta = meta

    @classmethod
    def create(
        cls,
        directory: PathLike,
        name: str,
        channels: Sequence[str],
        n_units: int,
        dtype: Any,
        meta: dict[str, Any] | None = None,
        stack: int | None = None,
    ) -> "TargetStore":
        """Allocate the zero-filled backing array and update the directory's ``meta.json``.

        ``stack`` gives a ``(T, stack, N)`` array; ``meta`` is merged into ``meta.json``.
        """
        directory = Path(directory)
        shape = (len(channels), n_units) if stack is None else (len(channels), stack, n_units)
        create_memmap(directory / f"{name}.npy", shape, dtype)
        merged = read_store_meta(directory)
        merged["channels"] = list(channels)
        merged.update(meta or {})
        write_json(merged, directory / "meta.json")
        return cls(directory, name, merged)

    @classmethod
    def open(cls, directory: PathLike, name: str) -> "TargetStore":
        """Open an existing store written by :meth:`create`."""
        directory = Path(directory)
        return cls(directory, name, read_store_meta(directory))

    @property
    def path(self) -> Path:
        """Path of the backing ``.npy`` file."""
        return self.directory / f"{self.name}.npy"

    @property
    def channels(self) -> list[str]:
        """Channel names in row order."""
        return self.meta["channels"]

    @property
    def n_units(self) -> int:
        """Number of spatial units, read from the array's own header."""
        return int(open_memmap(self.path, mode="r").shape[-1])

    def channel_index(self, names: Sequence[str]) -> np.ndarray:
        """Return the row indices of ``names`` (raises ``KeyError`` if absent)."""
        lookup = {channel: row for row, channel in enumerate(self.channels)}
        return np.array([lookup[name] for name in names], dtype=np.int64)

    def read(self, names: Sequence[str]) -> np.ndarray:
        """Read the named channels as float32, ``(len(names), N)`` or ``(len(names), K, N)``."""
        rows = self.channel_index(names)
        data = open_memmap(self.path, mode="r")
        return np.asarray(data[rows], dtype=np.float32)

    def writer(self) -> np.memmap:
        """Open the backing array for in-place writing."""
        return open_memmap(self.path, mode="r+")


def read_store_meta(directory: PathLike) -> dict[str, Any]:
    """Read a store directory's ``meta.json``, or an empty dict if it has none."""
    path = Path(directory) / "meta.json"
    return read_json(path) if path.exists() else {}


#: Per-unit arrays a store may carry beside its values, all optional, and the dtype of each.
UNIT_ARRAYS = {
    "positions": np.int32,
    "sample_index": np.int16,
    "radius": np.float32,
    "evaluable": bool,
}


def write_store_units(
    directory: PathLike,
    positions: np.ndarray | None = None,
    sample_index: np.ndarray | None = None,
    evaluable: np.ndarray | None = None,
    radius: np.ndarray | None = None,
    sample_names: Sequence[str] | None = None,
) -> None:
    """Record what each column of a :class:`TargetStore` refers to.

    Each array given is saved as a plain ``.npy`` beside the values; ``sample_names`` go into
    ``meta.json``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    given = {
        "positions": positions,
        "sample_index": sample_index,
        "radius": radius,
        "evaluable": evaluable,
    }
    for name, dtype in UNIT_ARRAYS.items():
        array = given[name]
        if array is not None:
            np.save(directory / f"{name}.npy", np.asarray(array, dtype=dtype))
    if sample_names is not None:
        meta = read_store_meta(directory)
        meta["samples"] = list(sample_names)
        write_json(meta, directory / "meta.json")


def read_store_units(directory: PathLike) -> dict[str, np.ndarray]:
    """Read whichever of :data:`UNIT_ARRAYS` the store directory holds."""
    directory = Path(directory)
    found = {}
    for name in UNIT_ARRAYS:
        path = directory / f"{name}.npy"
        if path.exists():
            found[name] = np.load(path)
    return found


def store_clusterings(directory: PathLike) -> dict[str, Path]:
    """Map each clustering label to its ``clusters_<label>.npy`` file in a store directory."""
    return {
        path.stem[len("clusters_"):]: path
        for path in sorted(Path(directory).glob("clusters_*.npy"))
    }
