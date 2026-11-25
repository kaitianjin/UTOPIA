"""Collect a results tree into the one file ``utopia view`` writes.

A page holds every null its targets were tested against and draws one confidence map per method
and null. It must open from ``file://``, where a sibling ``.npy`` cannot be fetched, so every array
is quantised to 16 bits over its own range, concatenated into one buffer, gzipped and base64'd into
a single ``<script>`` that the browser inflates.
"""

from __future__ import annotations

import base64
import gzip
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import __version__
from .index import ModelIndex, NullIndex, SampleIndex, TargetIndex, sorted_nulls

#: Axes a page can be fixed on, in output-path order. A fixed axis leaves the page's selectors
#: and enters its file name, which keeps a per-target page small enough to inline.
FIXABLE = ("model", "sample", "target")
#: Written above this many bytes, the page is reported as impractical to open.
SIZE_WARNING = 128 * 1024 * 1024
#: Placeholders the page template carries for the blobs injected into it.
MANIFEST_TOKEN = "/*__MANIFEST__*/null"
PAYLOAD_TOKEN = "__PAYLOAD__"
LOGO_TOKEN = "__LOGO__"
VERSION_TOKEN = "__VERSION__"
#: The wordmark shown in the page header, inlined so the file stays self-contained.
LOGO_PATH = Path(__file__).parent / "logo.png"
#: Raised when the filters of a selection leave nothing to show.
NO_RESULTS_MESSAGE = (
    "the selection matched no results; check the filters against 'utopia view --list'"
)


@dataclass
class Selection:
    """Which slice of a results tree to build a page for.

    Every field is a list of names to keep, or None to keep the axis whole.
    """

    models: Sequence[str] | None = None
    samples: Sequence[str] | None = None
    targets: Sequence[str] | None = None
    methods: Sequence[str] | None = None
    nulls: Sequence[str] | None = None
    clusters: Sequence[str] | None = None
    scales: Sequence[int] | None = None

    def keeps(self, axis: str, value: str | int) -> bool:
        """Whether one value of one axis survives this selection."""
        wanted = getattr(self, axis)
        return wanted is None or str(value) in {str(item) for item in wanted}


@dataclass
class Payload:
    """The arrays a page carries, in the order they are concatenated."""

    # Array key -> its header (offset, byte length, dtype, value range).
    entries: dict[str, dict[str, object]] = field(default_factory=dict)
    chunks: list[bytes] = field(default_factory=list)  # quantised buffers, in offset order
    offset: int = 0  # total bytes written so far

    def add(self, key: str, loader: Callable[[], np.ndarray], kind: str = "value") -> str:
        """Quantise one array into the buffer, unless ``key`` is already there, and return ``key``.

        ``kind`` is ``"value"`` (16 bits over the array's own range), ``"position"`` (exact integer
        coordinates, ``(N, 2)``) or ``"label"`` (cluster labels as uint8).
        """
        if key in self.entries:
            return key
        values = loader()
        if kind == "position":
            width = "u32" if int(np.max(values, initial=0)) > 0xFFFF else "u16"
            raw = np.ascontiguousarray(values, dtype=np.uint32 if width == "u32" else np.uint16)
            header = {"dtype": width, "columns": 2, "n": int(len(values))}
        elif kind == "label":
            raw = np.ascontiguousarray(values, dtype=np.uint8)
            header = {"dtype": "u8", "columns": 1, "n": int(len(values))}
        else:
            finite = np.asarray(values, dtype=np.float64).ravel()
            low = float(np.min(finite)) if len(finite) else 0.0
            high = float(np.max(finite)) if len(finite) else 1.0
            span = high - low
            scaled = (finite - low) / span * 0xFFFF if span > 0 else np.zeros(len(finite))
            raw = np.ascontiguousarray(np.rint(scaled), dtype=np.uint16)
            header = {"dtype": "u16", "columns": 1, "n": int(len(finite)), "min": low, "max": high}

        buffer = raw.tobytes()
        header.update(offset=self.offset, bytes=len(buffer))
        self.entries[key] = header
        self.chunks.append(buffer)
        self.offset += len(buffer)
        return key

    def encode(self) -> str:
        """Gzip the whole buffer once and base64 it for the ``<script>`` tag."""
        return base64.b64encode(gzip.compress(b"".join(self.chunks), 6)).decode("ascii")


class Locator:
    """The lazy counterpart of :class:`Payload`, used by ``--serve``.

    It records how to load each array instead of loading it, so arrays are read one at a time.
    """

    def __init__(self) -> None:
        """Start with no arrays recorded."""
        self.loaders: dict[str, Callable[[], np.ndarray]] = {}

    def add(self, key: str, loader: Callable[[], np.ndarray], kind: str = "value") -> str:
        """Record how to load one array, without loading it; ``kind`` is accepted and ignored."""
        self.loaders.setdefault(key, loader)
        return key

    def read(self, key: str) -> np.ndarray:
        """Load one array as the float32 the page expects; an unknown key raises KeyError."""
        return np.ascontiguousarray(self.loaders[key](), dtype=np.float32).ravel()


def _units_entry(
    payload: Payload | Locator,
    model: str,
    sample: SampleIndex,
    modality: str,
    scale: int,
    selection: Selection,
) -> dict[str, object] | None:
    """Read one ``units_<modality>_scale{s}.npz`` into the manifest and payload.

    Returns None when the sample has no such file.
    """
    path = sample.units.get((modality, scale))
    if path is None:
        return None
    prefix = f"u/{model}/{sample.name}/{modality}/{scale}"
    entry: dict[str, object] = {}
    with np.load(path) as data:
        names = list(data.files)
        if "canvas_shape" in names:
            entry["canvas"] = [int(value) for value in data["canvas_shape"]]
        entry["radius"] = float(data["radius"]) if "radius" in names else 0.0
        # External spots may be squares of half-width ``radius`` instead of disks.
        entry["square"] = bool(data["square"]) if "square" in names else False
        if "tissue_mask" in names:
            # An external sample's tissue mask on the canvas, row-major, painted behind its spots.
            tissue = data["tissue_mask"]
            payload.add(f"{prefix}/tissue", lambda values=tissue: values.ravel(), "label")
            entry["tissue"] = f"{prefix}/tissue"
        if "positions" in names:
            positions = data["positions"]
            entry["n"] = int(len(positions))
            if "canvas" not in entry:
                entry["canvas"] = [int(positions[:, 0].max()) + 1, int(positions[:, 1].max()) + 1]
            payload.add(f"{prefix}/positions", lambda values=positions: values, "position")
            entry["positions"] = f"{prefix}/positions"
        if "calib_positions" in names and len(data["calib_positions"]):
            calib = data["calib_positions"]
            entry["n_calib"] = int(len(calib))
            payload.add(f"{prefix}/calib_positions", lambda values=calib: values, "position")
            entry["calib_positions"] = f"{prefix}/calib_positions"
        labels = {}
        for name in names:
            if not name.startswith("clusters_"):
                continue
            label = name[len("clusters_"):]
            if not selection.keeps("clusters", label):
                continue
            key = f"{prefix}/clusters/{label}"
            payload.add(key, lambda values=data[name]: values, "label")
            labels[label] = key
        if labels:
            entry["clusters"] = labels
    return entry


def _target_entry(
    payload: Payload | Locator,
    model: str,
    sample: SampleIndex,
    target: TargetIndex,
    selection: Selection,
    scales: Sequence[int],
    display: dict[str, float],
    ranges: dict[str, dict[str, list[float]]],
) -> dict[str, object]:
    """Read one target's own arrays and index the methods applied to it.

    ``ranges`` holds the per-slide colour ranges of an external run, keyed scale then kind
    (``prediction``, ``truth``, ``calib_truth``); it is empty for the built-in model.
    """
    entry: dict[str, object] = {
        "modality": target.modality or "gene",
        "display": display,
        "range": ranges,
        "arrays": {},
        "methods": {},
    }
    for scale in scales:
        per_scale: dict[str, str] = {}
        for name, available in (
            ("prediction", target.scales),
            ("truth", target.truth_scales),
            ("calib_truth", target.calib_scales),
        ):
            if scale not in available:
                continue
            path = target.directory / f"{name}_scale{scale}.npy"
            key = f"t/{model}/{sample.name}/{target.name}/{scale}/{name}"
            payload.add(key, lambda source=path: np.load(source))
            per_scale[name] = key
        if per_scale:
            entry["arrays"][str(scale)] = per_scale

    for method_name, method in target.methods.items():
        if not selection.keeps("methods", method_name):
            continue
        nulls: dict[str, object] = {}
        for value, null in method.nulls.items():
            if not selection.keeps("nulls", value):
                continue
            found = _null_entry(
                payload, model, sample, target, method_name, null, selection, scales
            )
            if found["arrays"]:
                nulls[value] = found
        if nulls:
            entry["methods"][method_name] = {"nulls": nulls}
    return entry


def _null_entry(
    payload: Payload | Locator,
    model: str,
    sample: SampleIndex,
    target: TargetIndex,
    method: str,
    null: NullIndex,
    selection: Selection,
    scales: Sequence[int],
) -> dict[str, object]:
    """Read the adjusted p-values one (method, null) produced, per label and scale.

    Only ``pvals_adj`` is carried, because the page computes the confidence score from it.
    """
    entry: dict[str, dict[str, str]] = {}
    pairs = null.keys.get("pvals_adj", [])
    if not pairs:
        return {"arrays": entry}
    source = null.directory / "result.npz"
    for label, scale in pairs:
        if scale not in scales:
            continue
        if label is not None and not selection.keeps("clusters", label):
            continue
        suffix = f"scale{scale}" if label is None else f"{label}_scale{scale}"
        key = f"p/{model}/{sample.name}/{target.name}/{method}/{null.value}/{label or '-'}/{scale}"
        payload.add(key, lambda path=source, name=f"pvals_adj_{suffix}": _npz_member(path, name))
        entry.setdefault(label or "-", {})[str(scale)] = key
    return {"arrays": entry}


def _npz_member(path: Path, name: str) -> np.ndarray:
    """Read one member of a ``.npz`` without keeping the archive open."""
    with np.load(path) as data:
        return data[name]


def build_manifest(
    models: dict[str, ModelIndex],
    selection: Selection,
    payload: Payload | Locator,
    source: Path,
    fixed: dict[str, str] | None = None,
) -> dict[str, object]:
    """Index a tree into the manifest the page navigates, filling ``payload``.

    ``payload`` is a :class:`Payload` to inline the arrays, or a :class:`Locator` to load them on
    request.
    """
    manifest: dict[str, object] = {
        "source": str(source),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "confidence": {"p_min": 0.01, "p_max": 0.5},
        "fixed": dict(fixed or {}),
        "label": " · ".join((fixed or {}).values()) or "all results",
        "models": {},
    }
    for model_name, model in models.items():
        if not selection.keeps("models", model_name):
            continue
        header = model.header
        divisors = header.get("display_divisor", {})
        ranges = header.get("display_range", {})
        scale_header = header.get("confidence_scale")
        if scale_header:
            manifest["confidence"] = scale_header
        samples: dict[str, object] = {}
        for sample_name, sample in model.samples.items():
            if not selection.keeps("samples", sample_name):
                continue
            targets: dict[str, object] = {}
            units: dict[str, object] = {}
            for target_name, target in sample.targets.items():
                if not selection.keeps("targets", target_name):
                    continue
                scales = [
                    scale
                    for scale in sorted(set(target.scales))
                    if selection.keeps("scales", scale)
                ]
                if not scales:
                    continue
                entry = _target_entry(
                    payload, model_name, sample, target, selection, scales,
                    divisors.get(target_name, {}),
                    ranges.get(target_name, {}).get(sample_name, {}),
                )
                if not entry["methods"]:
                    continue
                targets[target_name] = entry
                modality = entry["modality"]
                for scale in scales:
                    key = f"{modality}/{scale}"
                    if key in units:
                        continue
                    found = _units_entry(payload, model_name, sample, modality, scale, selection)
                    if found is not None:
                        units[key] = found
            if targets:
                samples[sample_name] = {"units": units, "targets": targets}
        if samples:
            manifest["models"][model_name] = {"header": header, "samples": samples}
    if isinstance(payload, Payload):
        manifest["arrays"] = payload.entries
        manifest["inline"] = True
    else:
        manifest["inline"] = False
    return manifest


def manifest_nulls(manifest: dict[str, object]) -> list[str]:
    """Every null a page's manifest holds results for, over all its targets, smallest first.

    A null is named after its ``null_<name>`` directory; names that are not numbers sort last.
    """
    found: set[str] = set()
    for model in manifest["models"].values():
        for sample in model["samples"].values():
            for target in sample["targets"].values():
                for method in target["methods"].values():
                    found.update(method["nulls"])
    return sorted_nulls(found)


def render_page(manifest: dict[str, object], payload: Payload | None) -> str:
    """Substitute the manifest, the wordmark and the array blob into the template."""
    template = (Path(__file__).parent / "page.html").read_text()
    page = template.replace(MANIFEST_TOKEN, json.dumps(manifest, separators=(",", ":")))
    page = page.replace(LOGO_TOKEN, base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii"))
    page = page.replace(VERSION_TOKEN, __version__)
    return page.replace(PAYLOAD_TOKEN, "" if payload is None else payload.encode())


def build_page(
    models: dict[str, ModelIndex],
    selection: Selection,
    source: Path,
    out: Path,
    fixed: dict[str, str] | None = None,
) -> Path:
    """Write one self-contained HTML page for the selected slice of a tree."""
    payload = Payload()
    manifest = build_manifest(models, selection, payload, source, fixed)
    if not manifest["models"]:
        raise ValueError(NO_RESULTS_MESSAGE)
    page = render_page(manifest, payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    size = out.stat().st_size
    nulls = manifest_nulls(manifest)
    print(f"[view] {out}: null(s) {', '.join(nulls)}; {len(payload.entries)} array(s), "
          f"{payload.offset / 1e6:.1f} MB raw -> {size / 1e6:.1f} MB")
    if size > SIZE_WARNING:
        print(
            f"[view] warning: {size / 1e6:.0f} MB is past what a browser opens comfortably; "
            "fix more axes with --fix, narrow the selection, or use --serve"
        )
    return out


def _safe(name: str) -> str:
    """A name usable as one path component, whatever the tree called it."""
    return name.replace("/", "_").replace("\\", "_").strip() or "unnamed"


def page_plan(
    models: dict[str, ModelIndex], selection: Selection, fix: Sequence[str]
) -> list[tuple[Path, dict[str, str], Selection]]:
    """One ``(relative path, fixed axes, selection)`` per page to write, in path order.

    There is one page per combination of the axes in ``fix``, at ``<model>/<sample>/<target>.html``
    when all three are fixed and a single ``viewer.html`` when none is.
    """
    unknown = [axis for axis in fix if axis not in FIXABLE]
    if unknown:
        raise ValueError(f"cannot fix {unknown}; available axes: {list(FIXABLE)}")
    ordered = [axis for axis in FIXABLE if axis in fix]

    combinations: list[dict[str, str]] = []
    for model_name, model in models.items():
        if not selection.keeps("models", model_name):
            continue
        for sample_name, sample in model.samples.items():
            if not selection.keeps("samples", sample_name):
                continue
            for target_name in sample.targets:
                if not selection.keeps("targets", target_name):
                    continue
                combinations.append(
                    {"model": model_name, "sample": sample_name, "target": target_name}
                )

    plans: dict[tuple, dict[str, str]] = {}
    for combination in combinations:
        plans.setdefault(tuple(combination[axis] for axis in ordered),
                         {axis: combination[axis] for axis in ordered})

    pages = []
    for key, fixed in sorted(plans.items()):
        parts = [_safe(value) for value in key]
        path = Path(*parts[:-1], f"{parts[-1]}.html") if parts else Path("viewer.html")
        narrowed = Selection(
            models=[fixed["model"]] if "model" in fixed else selection.models,
            samples=[fixed["sample"]] if "sample" in fixed else selection.samples,
            targets=[fixed["target"]] if "target" in fixed else selection.targets,
            methods=selection.methods,
            nulls=selection.nulls,
            clusters=selection.clusters,
            scales=selection.scales,
        )
        pages.append((path, fixed, narrowed))
    return pages


def build_pages(
    models: dict[str, ModelIndex],
    selection: Selection,
    source: Path,
    out_dir: Path,
    fix: Sequence[str],
) -> list[Path]:
    """Write the whole tree of pages :func:`page_plan` describes."""
    pages = page_plan(models, selection, fix)
    if not pages:
        raise ValueError(NO_RESULTS_MESSAGE)
    written = [
        build_page(models, narrowed, source, out_dir / path, fixed)
        for path, fixed, narrowed in pages
    ]
    print(f"[view] {len(written)} page(s) under {out_dir}")
    return written
