"""Materialize compressed structures and BinaryCIF as parser-ready files."""

from __future__ import annotations

import gzip
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SUPPORTED_STRUCTURE_EXTENSIONS = frozenset({".pdb", ".ent", ".cif", ".mmcif", ".bcif"})


def structure_extension(path: Path) -> str:
    """Return the coordinate extension beneath an optional gzip suffix."""
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return Path(name).suffix


@contextmanager
def materialized_structure_path(path: Path) -> Iterator[Path]:
    """Yield PDB or text mmCIF, retaining BinaryCIF categories and identities."""
    with path.open("rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    extension = structure_extension(path)
    if not compressed and extension != ".bcif":
        yield path
        return

    with tempfile.TemporaryDirectory(prefix="cooper-beta-format-") as directory:
        materialized = path
        if compressed:
            materialized = Path(directory) / f"input{extension}"
            with gzip.open(path, "rb") as source, materialized.open("wb") as target:
                shutil.copyfileobj(source, target)
        if extension == ".bcif":
            from mmcif.io.BinaryCifReader import BinaryCifReader
            from mmcif.io.PdbxWriter import PdbxWriter

            containers = BinaryCifReader().deserialize(str(materialized))
            materialized = Path(directory) / "input.cif"
            with materialized.open("w", encoding="utf-8") as handle:
                PdbxWriter(handle).write(containers)
        yield materialized
