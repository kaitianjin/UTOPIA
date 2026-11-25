"""Command-line entry point.

Stages call each other's functions directly, so a run is one process.

    utopia check   config.yaml
    utopia run     config.yaml [--stage preprocess features cluster calibrate sample infer]
    utopia run     config.yaml [--stage import features cluster infer]      (mode: external)
    utopia view    <workspace|results-dir> [filters] [--out page.html | --serve]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable

from .config import Config, load_config
from .external import STAGES as EXTERNAL_STAGES

#: Stage order of the built-in model; ``utopia run`` without ``--stage`` executes all of them.
STAGES: list[str] = ["preprocess", "features", "cluster", "calibrate", "sample", "infer"]
#: Every stage name ``--stage`` accepts, built-in stages first, then the external-only ones.
ALL_STAGES: list[str] = STAGES + [stage for stage in EXTERNAL_STAGES if stage not in STAGES]


def _stage_functions() -> dict[str, Callable[[Config], None]]:
    """Map stage names to their entry points, imported lazily so ``check`` never loads torch."""
    from . import clustering, features, inference, preprocess
    from .calibration import draws as calibration_draws
    from .calibration import run as calibration_run

    return {
        "preprocess": preprocess.run,
        "features": features.run,
        "cluster": clustering.run,
        "calibrate": calibration_run.run,
        "sample": calibration_draws.run,
        "infer": inference.run,
    }


def _external_stage_functions() -> dict[str, Callable[[Config], None]]:
    """Map the stages of an external run that lists sample folders to their entry points."""
    from . import inference
    from .external import cluster as external_cluster
    from .external import features as external_features
    from .external import inputs as external_inputs

    return {
        "import": external_inputs.run,
        "features": external_features.run,
        "cluster": external_cluster.run,
        "infer": inference.run,
    }


def _stages_of(config: Config) -> tuple[list[str], dict[str, Callable[[Config], None]]]:
    """Return the stage order of a run and the function behind each stage.

    The built-in model has six stages, an external run with sample folders has four, and an
    external run without samples only has ``infer``.
    """
    if config.imports_samples:
        return EXTERNAL_STAGES, _external_stage_functions()
    if config.is_external:
        from . import inference

        return ["infer"], {"infer": inference.run}
    return STAGES, _stage_functions()


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the requested command."""
    from .view.bundle import FIXABLE

    parser = argparse.ArgumentParser(prog="utopia", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="run the pipeline")
    run_parser.add_argument("config", help="path to config.yaml")
    run_parser.add_argument(
        "--stage",
        nargs="+",
        choices=ALL_STAGES,
        help=f"stages to run, in order (default: all of {' '.join(STAGES)}; with mode "
        f"'external' and sample folders: all of {' '.join(EXTERNAL_STAGES)})",
    )

    check_parser = commands.add_parser("check", help="validate the input tree and configuration")
    check_parser.add_argument("config", help="path to config.yaml")

    view_parser = commands.add_parser(
        "view", help="build an interactive page for a results tree"
    )
    view_parser.add_argument("path", help="a workspace, a results directory or one model's tree")
    view_parser.add_argument(
        "--out-dir",
        help="directory to write the pages into (default: <workspace>/visualization)",
    )
    view_parser.add_argument(
        "--fix",
        nargs="*",
        choices=list(FIXABLE),
        help="axes to fix per page, naming its file (default: model sample target); "
        "pass --fix with no axes for one page holding everything",
    )
    view_parser.add_argument(
        "--out", help="write a single page to exactly this path, fixing no axis"
    )
    view_parser.add_argument("--list", action="store_true", help="print the axes and exit")
    view_parser.add_argument("--serve", action="store_true", help="serve the tree instead of inlining it")
    view_parser.add_argument("--host", default="127.0.0.1", help="interface --serve binds to")
    view_parser.add_argument("--port", type=int, default=8765, help="port --serve binds to")
    for axis, help_text in (
        ("model", "models to include"),
        ("sample", "test samples to include"),
        ("target", "targets to include"),
        ("method", "uncertainty methods to include"),
        ("null", "null values to include"),
        ("k", "cluster labels to include, e.g. k5 k10 (or 5 10)"),
        ("scale", "superpixel scales to include"),
    ):
        view_parser.add_argument(f"--{axis}", nargs="+", help=help_text)

    args = parser.parse_args(argv)

    if args.command == "check":
        config = load_config(args.config)
        if config.imports_samples:
            from .external import inputs as external_inputs

            external_inputs.check(config)
        elif config.is_external:
            from .inference import check_stores

            check_stores(config)
        else:
            from . import preprocess

            preprocess.check(config)
        return 0

    if args.command == "view":
        return _view(args)

    config = load_config(args.config)
    available, functions = _stages_of(config)
    stages = args.stage or available
    unavailable = [stage for stage in stages if stage not in available]
    if unavailable:
        raise SystemExit(
            f"stage(s) {unavailable} do not exist for this run (mode '{config.mode}'); "
            f"available: {available}"
        )
    config.workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, config.workspace / "config.yaml")

    for stage in available:
        if stage in stages:
            print(f"\n=== {stage} ===")
            functions[stage](config)
    return 0


def _view(args: argparse.Namespace) -> int:
    """Build or serve the interactive viewer for a results tree."""
    from pathlib import Path

    from .view import (
        FIXABLE,
        Selection,
        axes,
        build_page,
        build_pages,
        index_tree,
        output_root,
        results_root,
    )

    root = results_root(Path(args.path))
    models = index_tree(root)
    if not models:
        raise SystemExit(f"no results tree under {root}")
    if args.list:
        for axis, values in axes(models).items():
            print(f"{axis:>9}: {' '.join(str(value) for value in values)}")
        return 0

    clusters = args.k
    if clusters is not None:
        # ``--k 5`` and ``--k k5`` name the same clustering; results are keyed by the label.
        available = axes(models)["clusters"]
        clusters = [item if item in available else f"k{item}" for item in clusters]
    selection = Selection(
        models=args.model,
        samples=args.sample,
        targets=args.target,
        methods=args.method,
        nulls=args.null,
        clusters=clusters,
        scales=None if args.scale is None else [int(value) for value in args.scale],
    )
    if args.serve:
        from .view.server import serve

        serve(models, selection, root, host=args.host, port=args.port)
        return 0

    if args.out:
        if args.fix:
            raise SystemExit("--out writes a single page, so it cannot be combined with --fix")
        build_page(models, selection, root, Path(args.out))
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else output_root(Path(args.path))
    fix = list(FIXABLE) if args.fix is None else args.fix
    build_pages(models, selection, root, out_dir, fix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
