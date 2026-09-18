from __future__ import annotations

import gzip
from pathlib import Path

import pytest
from Bio.PDB import MMCIFIO, PDBIO, MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from mmcif.io.IoAdapterPy import IoAdapterPy

from cooper_beta import detect
from cooper_beta.config import build_config
from cooper_beta.exceptions import DsspError, DsspNotFoundError
from cooper_beta.loader import ProteinLoader
from cooper_beta.polymer_sequence import declared_polymer_sequences
from cooper_beta.runtime import require_dssp_binary
from cooper_beta.structure_io import materialized_structure_path

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "M4QT10.cif"


def _binary_cif(source: Path, destination: Path) -> None:
    adapter = IoAdapterPy(raiseExceptions=True)
    containers = adapter.readFile(str(source))
    adapter.writeFile(
        str(destination), containers, fmt="bcif", useStringTypes=True, applyTypes=False
    )


def _dssp_available() -> bool:
    try:
        require_dssp_binary()
    except (DsspError, DsspNotFoundError):
        return False
    return True


@pytest.mark.parametrize("compressed", [False, True])
def test_binary_cif_preserves_polymer_and_atom_identity(tmp_path: Path, compressed: bool):
    source = tmp_path / ("example.bcif.gz" if compressed else "example.bcif")
    _binary_cif(EXAMPLE, source)
    if compressed:
        source.write_bytes(gzip.compress(source.read_bytes()))
    with materialized_structure_path(source) as converted:
        observed = MMCIF2Dict(converted)
        expected = MMCIF2Dict(EXAMPLE)
        for key in expected:
            if key != "data_":
                assert observed[key] == expected[key], key
    assert declared_polymer_sequences(source) == declared_polymer_sequences(EXAMPLE)


def test_binary_cif_keeps_selected_model_and_author_residue_ids(tmp_path: Path):
    structure = MMCIFParser(QUIET=True).get_structure("example", EXAMPLE)
    first = structure[0]
    second = first.copy()
    second.id = 1
    second.serial_num = 2
    chain = second["A"]
    chain.id = "LONG_B"
    residue = next(chain.get_residues())
    residue.id = (" ", 10, "A")
    structure.add(second)
    source = tmp_path / "models.cif"
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(source))
    binary = tmp_path / "models.bcif"
    _binary_cif(source, binary)
    cfg = build_config({"input.model_id": 1})
    loader = ProteinLoader(binary, cfg.input, dssp_bin=None)
    assert loader.available_chains() == ["LONG_B"]
    assert ("LONG_B", (" ", 10, "A")) in loader._mmcif_polypeptide_positions
    assert loader.model.serial_num == 2


@pytest.mark.skipif(not _dssp_available(), reason="DSSP >= 4.5.3 is required")
@pytest.mark.parametrize("extension", [".pdb", ".ent", ".cif", ".mmcif", ".bcif"])
@pytest.mark.parametrize("compressed", [False, True])
def test_mainstream_formats_keep_barrel_features(tmp_path: Path, extension: str, compressed: bool):
    source = tmp_path / f"example{extension}"
    structure = MMCIFParser(QUIET=True).get_structure("example", EXAMPLE)
    if extension in {".pdb", ".ent"}:
        writer = PDBIO()
        writer.set_structure(structure)
        writer.save(str(source))  # Coordinate-only PDB, without SEQRES.
    elif extension == ".bcif":
        _binary_cif(EXAMPLE, source)
    else:
        source.write_bytes(EXAMPLE.read_bytes())
    if compressed:
        compressed_path = source.with_name(source.name + ".gz")
        compressed_path.write_bytes(gzip.compress(source.read_bytes()))
        source.unlink()
        source = compressed_path

    config = build_config({"runtime.prepare_cache_enabled": False, "runtime.log_console": False})
    run = detect(
        str(tmp_path),
        config=config,
        workers=1,
        prepare_workers=1,
        write_csv=False,
        print_summary=False,
        show_progress=False,
    )
    assert len(run.rows) == 1
    row = run.rows[0]
    assert row.filename == source.name
    assert row.author_chain_id == "A"
    assert row.result == "BARREL"
    assert (row.strand_count, row.strand_adjacency_count, row.cycle_strand_count) == (8, 8, 8)
    assert (row.cycle_strand_fraction, row.cycle_rank) == (1.0, 1)


@pytest.mark.skipif(not _dssp_available(), reason="DSSP >= 4.5.3 is required")
def test_coordinate_only_multichain_mmcif_keeps_author_chains(tmp_path: Path):
    structure = MMCIFParser(QUIET=True).get_structure("example", EXAMPLE)
    model = structure[0]
    second = model["A"].copy()
    second.id = "LONG_B"
    for atom in second.get_atoms():
        atom.coord = atom.coord + [100.0, 0.0, 0.0]
    model.add(second)
    source = tmp_path / "two_chains.cif"
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(source))
    cfg = build_config()
    loader = ProteinLoader(source, cfg.input, dssp_bin=cfg.runtime.dssp_bin_path)
    first_result = loader.prepare_chain("A")
    second_result = loader.prepare_chain("LONG_B")
    assert not first_result.failed and not second_result.failed
    assert len(first_result.residues) == len(second_result.residues) == 137
    assert len(first_result.strand_graph.nodes) == len(second_result.strand_graph.nodes) == 8
    assert len(first_result.strand_graph.edges) == len(second_result.strand_graph.edges) == 8
    assert first_result.strand_graph.author_chain_id == "A"
    assert second_result.strand_graph.author_chain_id == "LONG_B"
