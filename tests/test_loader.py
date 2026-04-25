from __future__ import annotations

import gzip
import os
from pathlib import Path

from cooper_beta.loader import ProteinLoader

MINIMAL_MODEL_PDB = """\
MODEL        1
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C
ATOM      3  C   ALA A   1       1.000   1.000   0.000  1.00 80.00           C
ATOM      4  O   ALA A   1       1.000   1.500   0.800  1.00 80.00           O
TER
ENDMDL
END
"""


def test_loader_accepts_gzipped_pdb_with_plain_pdb_suffix(tmp_path: Path):
    pdb_path = tmp_path / "alphafold-style.pdb"
    pdb_path.write_bytes(gzip.compress(MINIMAL_MODEL_PDB.encode("utf-8")))

    loader = ProteinLoader(str(pdb_path))

    assert loader.available_chains() == ["A"]


def test_loader_exports_cryst1_for_dssp_compatibility(tmp_path: Path):
    pdb_path = tmp_path / "model.pdb"
    pdb_path.write_text(MINIMAL_MODEL_PDB, encoding="utf-8")
    loader = ProteinLoader(str(pdb_path))

    exported_path = loader._export_protein_only_pdb()
    try:
        exported_text = Path(exported_path).read_text(encoding="utf-8")
    finally:
        os.remove(exported_path)

    assert "CRYST1" in exported_text.splitlines()[1]
