# -*- coding: utf-8 -*-
"""
Legacy entry point kept for backward compatibility.

Examples:
  python main.py data/ 8 4

Preferred:
  cooper-beta data/ --workers 8 --prepare-workers 4
  python -m cooper_beta data/ --workers 8 --prepare-workers 4
"""
from __future__ import annotations

import sys
from pathlib import Path

import os as _os
# Avoid over-subscription when using multiprocessing + BLAS
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# Ensure src/ is importable when running from the repository root
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cooper_beta.pipeline import main  # noqa: E402


def _legacy_cli(argv: list[str]) -> None:
    target_path = "data/"
    workers = None
    prepare_workers = None

    # 用法：
    #   python main.py <path> [analyze_workers] [prepare_workers]
    if len(argv) > 1:
        target_path = argv[1]
    if len(argv) > 2:
        try:
            workers = int(argv[2])
        except Exception:
            workers = None
    if len(argv) > 3:
        try:
            prepare_workers = int(argv[3])
        except Exception:
            prepare_workers = None
    else:
        prepare_workers = workers

    main(target_path, workers=workers, prepare_workers=prepare_workers)


if __name__ == "__main__":  # pragma: no cover
    _legacy_cli(sys.argv)
