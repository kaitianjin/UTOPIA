"""Command-line entry point: ``python -m utopia.baselines config.yaml``.

Runs the UQ baselines against a workspace the pipeline has already filled.  The config is the same
file the pipeline run used; the options come from its ``baselines:`` block (inline, or in a
``<config>.baselines.yaml`` beside it), which the flags override for one invocation.

    python -m utopia.baselines config.yaml
    python -m utopia.baselines config.yaml --methods raw deep_ensemble
    python -m utopia.baselines config.yaml --list
"""

from __future__ import annotations

import argparse

from .config import METHODS, BaselineWorkspace, load


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the baselines against an already-filled workspace."""
    parser = argparse.ArgumentParser(
        prog="python -m utopia.baselines", description=__doc__.splitlines()[0]
    )
    parser.add_argument("config", help="the same config.yaml the pipeline run used")
    parser.add_argument(
        "--baselines",
        help="YAML holding the 'baselines:' block (default: <config>.baselines.yaml, "
        "or a 'baselines:' key inside the config itself)",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=list(METHODS), help="override 'baselines.methods'"
    )
    parser.add_argument(
        "--members", type=int,
        help="override 'baselines.n_members' (M, the deep-ensemble members)",
    )
    parser.add_argument(
        "--resamples", type=int,
        help="override 'baselines.n_resamples' (M, the bootstrap members of 'resampling'; "
        "separate from --members because it sets the resolution of that row's count p-value)",
    )
    parser.add_argument("--passes", type=int, help="override 'baselines.n_passes' (K)")
    parser.add_argument(
        "--clusters", nargs="+", type=int, help="override 'baselines.n_clusters'"
    )
    parser.add_argument(
        "--no-conformal-marginal",
        dest="conformal_marginal",
        action="store_false",
        default=None,
        help="drop the marginal split-conformal row, which is otherwise reported beside "
        "the four baselines as UTOPIA's direct ablation",
    )
    parser.add_argument(
        "--conformal",
        action="store_true",
        help="also report each baseline's mean through cluster-conditional conformal; "
        "not a baseline, and off unless asked for",
    )
    parser.add_argument(
        "--prefix",
        help="override 'baselines.prefix', the string every row's directory name starts with "
        "(default 'baseline_', which is what keeps a baseline row out of the pipeline's own "
        "mc_dropout/ and resampling/)",
    )
    images = parser.add_mutually_exclusive_group()
    images.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        default=None,
        help="write the prediction, confidence and FDP-curve images (default: follow the run's "
        "own 'inference.visualize')",
    )
    images.add_argument(
        "--no-visualize", dest="visualize", action="store_false", help="write arrays alone"
    )
    parser.add_argument(
        "--list", action="store_true", help="print what would run, then exit"
    )
    args = parser.parse_args(argv)

    config, baselines = load(args.config, args.baselines)
    if args.methods:
        baselines.methods = list(args.methods)
    if args.members:
        baselines.n_members = args.members
    if args.resamples:
        baselines.n_resamples = args.resamples
    if args.passes:
        baselines.n_passes = args.passes
    if args.clusters:
        baselines.n_clusters = list(args.clusters)
    if args.conformal:
        baselines.conformalize = True
    if args.conformal_marginal is not None:
        baselines.conformal_marginal = args.conformal_marginal
    if args.prefix is not None:
        baselines.prefix = args.prefix
    if args.visualize is not None:
        baselines.visualize = args.visualize

    if args.list:
        print(f"workspace : {config.workspace}")
        print(f"model     : {config.model}")
        print(f"modalities: {', '.join(config.predict)}")
        print(f"scales    : {config.scales}")
        print(f"clusters  : {baselines.clusters(config)}")
        print(f"rows      : {', '.join(baselines.labels())}")
        print(f"M / K     : {baselines.n_members} (ensembles), {baselines.n_resamples} "
              f"(resampling) / {baselines.n_passes}")
        print(f"images    : {baselines.visualizes(config)}")
        workspace = BaselineWorkspace(config)
        print(f"results   : {workspace.results_root(config.model)}")
        print(f"apparatus : {workspace.baselines_root(config.model)}")
        return 0

    from .run import run

    run(config, baselines)
    return 0
