"""
Repository convenience entry point.

Preferred installed command:
  cooper-beta path/to/structures --workers 8 --prepare-workers 4
"""
from __future__ import annotations

# ruff: noqa: E402, I001

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cooper_beta.cli import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    main()
