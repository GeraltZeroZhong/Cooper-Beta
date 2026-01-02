# -*- coding: utf-8 -*-
from __future__ import annotations

import os as _os

# Avoid over-subscription when using multiprocessing + BLAS
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from pathlib import Path

from .pipeline import main as _run


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="cooper-beta",
        description="Detect beta-barrel-like chains from PDB/mmCIF files and write a CSV summary.",
    )
    ap.add_argument(
        "path",
        nargs="?",
        default="data/",
        help="输入路径：单个文件，或包含 .pdb/.cif/.mmcif 的目录（默认：data/）",
    )
    ap.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="分析阶段进程数（默认：CPU-1 或 1）",
    )
    ap.add_argument(
        "--prepare-workers",
        "--prep",
        type=int,
        default=None,
        help="准备阶段进程数（DSSP/解析，默认：与 --workers 相同）",
    )
    ap.add_argument(
        "--out",
        "-o",
        default="cooper_beta_results.csv",
        help="输出 CSV 路径（默认：cooper_beta_results.csv）",
    )
    args = ap.parse_args(argv)

    prep = args.prepare_workers if args.prepare_workers is not None else args.workers
    _run(args.path, workers=args.workers, prepare_workers=prep, out_csv=args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
