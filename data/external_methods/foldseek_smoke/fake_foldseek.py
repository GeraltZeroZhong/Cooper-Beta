#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def _touch_db(prefix: str) -> None:
    path = Path(prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fake foldseek db\n", encoding="utf-8")


def _write_hits(query_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    query_files = [query_path] if query_path.is_file() else sorted(query_path.glob("*.pdb"))
    for path in query_files:
        query_id = f"{path.name}_A"
        if "toy_barrel" in path.stem and "nonbarrel" not in path.stem:
            rows.append(
                "\t".join(
                    [
                        query_id,
                        "ref_barrel_A",
                        "16",
                        "16",
                        "16",
                        "1.0",
                        "1.0",
                        "0.72",
                        "0.74",
                        "0.73",
                        "0.73",
                        "120",
                    ]
                )
            )
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def main(argv: list[str]) -> int:
    if not argv:
        return 2
    command = argv[0]
    if command == "createdb":
        _touch_db(argv[2])
        return 0
    if command == "createindex":
        Path(argv[2]).mkdir(parents=True, exist_ok=True)
        return 0
    if command == "easy-search":
        _write_hits(Path(argv[1]), Path(argv[3]))
        return 0
    print(f"Unsupported fake Foldseek command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
