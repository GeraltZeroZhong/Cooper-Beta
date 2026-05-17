from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - path execution convenience
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from cooper_beta.evaluation.app import main
else:
    from .app import main

if __name__ == "__main__":  # pragma: no cover
    main()
