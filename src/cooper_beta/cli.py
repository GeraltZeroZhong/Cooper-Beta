from __future__ import annotations

import argparse

from .bootstrap import configure_thread_environment
from .config import build_config
from .pipeline import apply_runtime_overrides, run_pipeline
from .runtime import runtime_summary

configure_thread_environment()


def _looks_like_hydra_override(token: str) -> bool:
    return token.startswith(("+", "~")) or ("=" in token)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cooper-beta",
        description=(
            "Detect beta-barrel-like protein chains from PDB/mmCIF inputs and write a CSV "
            "summary. Extra unknown arguments are forwarded as Hydra overrides."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help=(
            "Input path: a single structure file, or a directory containing structure files. "
            "When omitted, Hydra config `input.path` is used."
        ),
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Legacy shortcut for `runtime.workers`.",
    )
    parser.add_argument(
        "--prepare-workers",
        "--prep",
        type=int,
        default=None,
        help="Legacy shortcut for `runtime.prepare_workers`.",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Legacy shortcut for `output.csv_path`.",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check whether Python and DSSP are available, then exit.",
    )
    args, hydra_overrides = parser.parse_known_args(argv)
    if args.path and _looks_like_hydra_override(args.path):
        hydra_overrides = [args.path, *hydra_overrides]
        args.path = None

    cfg = build_config(hydra_overrides)
    cfg = apply_runtime_overrides(
        cfg,
        input_path=args.path,
        workers=args.workers,
        prepare_workers=args.prepare_workers,
        out_csv=args.out,
    )

    if args.check_env or cfg.runtime.check_env:
        summary = runtime_summary(cfg.runtime.dssp_bin_path)
        print(f"Python: {summary['python']}")
        print(f"DSSP: {summary['dssp']}")
        return

    run_pipeline(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
