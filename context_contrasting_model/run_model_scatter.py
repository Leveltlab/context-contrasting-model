from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import model_scatter, transition_templates


_DIRECTION_ARG_ALIASES = {
    "-NO": "__DIRECTION_NEG_NO__",
    "-O": "__DIRECTION_NEG_O__",
}


def _normalize_direction_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    for arg in argv:
        if arg.startswith("-NO"):
            normalized.append(f"__DIRECTION_NEG_NO__{arg.removeprefix('-NO')}")
        elif arg.startswith("-O"):
            normalized.append(f"__DIRECTION_NEG_O__{arg.removeprefix('-O')}")
        else:
            normalized.append(_DIRECTION_ARG_ALIASES.get(arg, arg))
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the publication model-scatter simulation.")
    parser.add_argument("--output-dir", type=Path, default=transition_templates.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--n-steps-per-phase", type=int, default=400)
    parser.add_argument("--test-trials", type=int, default=5)
    parser.add_argument("--training-trials", type=int, default=7)
    parser.add_argument("--training-stimulus-order", choices=("randomized", "fixed"), default="randomized")
    parser.add_argument("--seed", type=int, default=7151)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--scalar-noise-multiplier", type=float, default=1.75)
    parser.add_argument("--canonical-only", action="store_true", help="Not supported; template tuning is sampled by cell.")
    parser.add_argument("--transition-sampling", choices=("data-like", "equal"), default="data-like")
    parser.add_argument("--zscore-std-floor", type=float, default=0.04)
    parser.add_argument("--response-tail-fraction", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--image-format", choices=("png", "svg", "eps"), default="svg")
    parser.add_argument("--axis-clip-percentile", type=float, default=99.0)
    parser.add_argument("--plot-center-panels", dest="plot_center_panels", action="store_true", default=True)
    parser.add_argument("--no-plot-center-panels", dest="plot_center_panels", action="store_false")
    parser.add_argument("--plot-by-transition", action="store_true")
    parser.add_argument("--export-panels", dest="export_panels", action="store_true", default=True)
    parser.add_argument("--no-export-panels", dest="export_panels", action="store_false")
    
    """
    1  silent_broad_FFonly
    2  silent_broad_FB_mid
    3  silent_broad_FB_strong
    4  weak_broad_FFonly
    5  mid_broad_FFonly
    6  strong_broad_FFonly
    7  weak_broad_FB_all
    8  mid_broad_FB_all
    9  strong_broad_FB_all
    10 mixed_broad_FB_all
    11 narrow_diag_FB_all
    12 narrow_weak_FB_all
    13 narrow_mid_FFonly
    """
    
    parser.add_argument(
        "--fam-examples",
        nargs="*",
        default=["2", "3", "-NO/+O", "+NO", "13", "6"],
        metavar="TEMPLATE_NUM_OR_DIRECTION",
        help=(
            "Template numbers, directions (+NO, +O, -NO), or diagonal examples (+NO/+O, +O/+NO, -NO/+O, +O/-NO); "
            "combine as +NO13 to constrain a specific template, and append s/m/h to prefer small/mid/high displacement."
        ),
    )
    parser.add_argument(
        "--nov-examples",
        nargs="*",
        default=["+NO/+Om", "11", "7", "2", "6"],
        metavar="TEMPLATE_NUM_OR_DIRECTION",
        help=(
            "Template numbers, directions (+NO, +O, -NO), or diagonal examples (+NO/+O, +O/+NO, -NO/+O, +O/-NO); "
            "combine as +NO13 to constrain a specific template, and append s/m/h to prefer small/mid/high displacement."
        ),
    )
    parser.add_argument(
        "--use-center-examples",
        action="store_true",
        help="Deprecated: highlighted examples are selected from sampled cells above --threshold.",
    )
    return parser.parse_args(_normalize_direction_args(sys.argv[1:]))


def main() -> None:
    args = parse_args()
    if args.canonical_only:
        raise ValueError("--canonical-only is not supported by the publication sampler.")
    args.skip_center_panels = not args.plot_center_panels
    transition_templates.configure_model_scatter(args.n_steps_per_phase)
    model_scatter.run_model_scatter(args)
    transition_templates.write_metadata(args)


if __name__ == "__main__":
    main()
