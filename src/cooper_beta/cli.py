from __future__ import annotations

import os as _os

# Avoid over-subscription when using multiprocessing + BLAS
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse

from .pipeline import main as _run
from .runtime import runtime_summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="cooper-beta",
        description="Detect beta-barrel-like protein chains from PDB/mmCIF inputs and write a CSV summary.",
    )
    ap.add_argument(
        "path",
        nargs="?",
        default="data/",
        help="Input path: a single structure file, or a directory containing .pdb/.cif/.mmcif files (default: data/).",
    )
    ap.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of analysis workers (default: CPU count minus one, with a minimum of one).",
    )
    ap.add_argument(
        "--prepare-workers",
        "--prep",
        type=int,
        default=None,
        help="Number of preparation workers for DSSP/parsing (default: same as --workers).",
    )
    ap.add_argument(
        "--out",
        "-o",
        default="cooper_beta_results.csv",
        help="Output CSV path (default: cooper_beta_results.csv).",
    )
    ap.add_argument(
        "--check-env",
        action="store_true",
        help="Check whether Python and DSSP are available, then exit.",
    )
    args = ap.parse_args(argv)

    if args.check_env:
        summary = runtime_summary()
        print(f"Python: {summary['python']}")
        print(f"DSSP: {summary['dssp']}")
        return

    prep = args.prepare_workers if args.prepare_workers is not None else args.workers
    _run(args.path, workers=args.workers, prepare_workers=prep, out_csv=args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
