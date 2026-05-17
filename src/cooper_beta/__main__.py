from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - path execution convenience
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cooper_beta.cli import main
else:
    from .cli import main

if __name__ == "__main__":  # pragma: no cover
    main()
