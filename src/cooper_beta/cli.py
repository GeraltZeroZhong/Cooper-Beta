from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

from hydra.errors import HydraException
from omegaconf.errors import OmegaConfBaseException

from .bootstrap import configure_thread_environment
from .config import build_config
from .exceptions import CooperBetaError
from .pipeline import apply_runtime_overrides, run_pipeline_result
from .runtime import runtime_summary


def _looks_like_hydra_override(token: str) -> bool:
    return token.startswith(("+", "~")) or ("=" in token)


def _package_version() -> str:
    try:
        return version("cooper-beta")
    except PackageNotFoundError:
        return "0.0.0"


def _has_override(overrides: list[str], key: str) -> bool:
    prefixes = (f"{key}=", f"+{key}=", f"++{key}=")
    return any(override.startswith(prefixes) for override in overrides)


def _has_true_override(overrides: list[str], key: str) -> bool:
    for override in overrides:
        if not override.startswith((f"{key}=", f"+{key}=", f"++{key}=")):
            continue
        value = override.split("=", 1)[1].strip().lower()
        return value in {"1", "true", "yes", "on"}
    return False


def _reject_unknown_options(parser: argparse.ArgumentParser, tokens: list[str]) -> None:
    for token in tokens:
        if token.startswith("-"):
            parser.error(f"unrecognized argument: {token}")


def _recover_positional_path(
    parser: argparse.ArgumentParser,
    path: str | None,
    tokens: list[str],
) -> tuple[str | None, list[str]]:
    remaining: list[str] = []
    recovered_path = path
    for token in tokens:
        if not _looks_like_hydra_override(token) and not token.startswith("-"):
            if recovered_path is None:
                recovered_path = token
                continue
            parser.error(f"unexpected extra input path: {token}")
        remaining.append(token)
    return recovered_path, remaining


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cooper-beta",
        description=(
            "Detect beta-barrel-like protein chains from PDB/mmCIF inputs and write a CSV "
            "summary. Advanced KEY=VALUE arguments are treated as Hydra overrides."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help=(
            "Input structure file or directory. Required unless input.path=... is provided."
        ),
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of analysis worker processes.",
    )
    parser.add_argument(
        "--prepare-workers",
        "--prep",
        type=int,
        default=None,
        help="Number of preparation worker processes (default: follows --workers).",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Write results CSV to this path.",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check whether Python and DSSP are available, then exit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )
    args, hydra_overrides = parser.parse_known_args(argv)
    if args.path and _looks_like_hydra_override(args.path):
        hydra_overrides = [args.path, *hydra_overrides]
        args.path = None
    args.path, hydra_overrides = _recover_positional_path(
        parser,
        args.path,
        hydra_overrides,
    )
    _reject_unknown_options(parser, hydra_overrides)

    if (
        args.path is None
        and not args.check_env
        and not _has_override(hydra_overrides, "input.path")
        and not _has_true_override(hydra_overrides, "runtime.check_env")
    ):
        parser.error("the input path is required (or pass input.path=...)")

    try:
        cfg = build_config(hydra_overrides)
        cfg = apply_runtime_overrides(
            cfg,
            input_path=args.path,
            workers=args.workers,
            prepare_workers=args.prepare_workers,
            out_csv=args.out,
        )

        configure_thread_environment()

        if args.check_env or cfg.runtime.check_env:
            summary = runtime_summary(cfg.runtime.dssp_bin_path, require_dssp=False)
            print(f"Python: {summary['python']} ({summary['python_executable']})")
            print(f"DSSP: {summary['dssp']}")
            if summary["dssp"] == "not found":
                raise SystemExit(2)
            return

        run_pipeline_result(cfg, write_csv=True, print_summary=True, strict_input=True)
    except (
        CooperBetaError,
        FileNotFoundError,
        HydraException,
        OmegaConfBaseException,
        OSError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
