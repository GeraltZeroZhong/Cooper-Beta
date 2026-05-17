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


def test_loader_maps_dssp_blank_residue_key_to_nonstandard_amino_acid(tmp_path: Path):
    pdb_path = tmp_path / "mse.pdb"
    pdb_path.write_text(
        """\
MODEL        1
HETATM    1  N   MSE A   1       0.000   0.000   0.000  1.00 80.00           N
HETATM    2  CA  MSE A   1       1.000   0.000   0.000  1.00 80.00           C
HETATM    3  C   MSE A   1       1.000   1.000   0.000  1.00 80.00           C
HETATM    4  O   MSE A   1       1.000   1.500   0.800  1.00 80.00           O
TER
ENDMDL
END
""",
        encoding="utf-8",
    )
    loader = ProteinLoader(str(pdb_path), fail_on_dssp_error=False)
    loader.secondary_structure = {("A", (" ", 1, " ")): "E"}

    residues = loader.get_chain_data("A")

    assert len(residues) == 1
    assert residues[0]["hetfield"] == "H_MSE"
    assert residues[0]["is_sheet"] is True


def test_loader_runs_dssp_on_mmcif_without_pdb_export(tmp_path: Path, monkeypatch):
    mmcif_path = tmp_path / "model.cif"
    mmcif_path.write_text("data_model\n", encoding="utf-8")
    loader = ProteinLoader.__new__(ProteinLoader)
    loader.file_path = str(mmcif_path)
    loader.dssp_bin = "/usr/bin/mkdssp"
    loader.fail_on_dssp_error = True
    loader.secondary_structure = None
    loader.secondary_structure_error = None
    loader.model = object()
    loader._structure_file_type = "MMCIF"

    def fail_export():
        raise AssertionError("mmCIF DSSP path must not export through PDB")

    class FakeDSSP:
        def __init__(self, model, in_file, *, dssp, file_type=""):
            assert model is loader.model
            assert in_file == str(mmcif_path)
            assert dssp == "/usr/bin/mkdssp"
            assert file_type == "MMCIF"

        def keys(self):
            return [("AA", (" ", 1, " "))]

        def __getitem__(self, key):
            assert key == ("AA", (" ", 1, " "))
            return (None, None, "E")

    monkeypatch.setattr(loader, "_export_protein_only_pdb", fail_export)
    monkeypatch.setattr("cooper_beta.loader.require_dssp_binary", lambda value: value)
    monkeypatch.setattr("cooper_beta.loader.DSSP", FakeDSSP)

    loader._run_secondary_structure()

    assert loader.secondary_structure == {("AA", (" ", 1, " ")): "E"}
