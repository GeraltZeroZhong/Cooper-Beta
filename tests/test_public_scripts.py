from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_copy_ok_rejects_escaping_paths(tmp_path: Path):
    copy_ok = importlib.import_module("scripts.copy_ok")
    source_dir = tmp_path / "structures"
    source_dir.mkdir()
    outside_file = tmp_path / "outside.pdb"
    outside_file.write_text("HEADER\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe"):
        copy_ok.find_in_source(source_dir, "../outside.pdb", recursive=True)

    with pytest.raises(ValueError, match="escapes"):
        copy_ok.find_in_source(source_dir, str(outside_file), recursive=True)


def test_copy_ok_uses_source_path_column_for_duplicate_basenames(tmp_path: Path):
    copy_ok = importlib.import_module("scripts.copy_ok")
    source_dir = tmp_path / "structures"
    first = source_dir / "first" / "same.pdb"
    second = source_dir / "second" / "same.pdb"
    destination = tmp_path / "selected"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("HEADER first\n", encoding="utf-8")
    second.write_text("HEADER second\n", encoding="utf-8")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "filename,source_path,result,reason\n"
        f"same.pdb,{second},BARREL,OK\n",
        encoding="utf-8",
    )

    exit_code = copy_ok.main(
        [
            "--csv",
            str(csv_path),
            "--src",
            str(source_dir),
            "--dst",
            str(destination),
            "--no-recursive-search",
        ]
    )

    assert exit_code == 0
    assert (destination / "same.pdb").read_text(encoding="utf-8") == "HEADER second\n"


def test_copy_ok_disambiguates_destination_name_collisions(tmp_path: Path):
    copy_ok = importlib.import_module("scripts.copy_ok")
    source_dir = tmp_path / "structures"
    first = source_dir / "first" / "same.pdb"
    second = source_dir / "second" / "same.pdb"
    destination = tmp_path / "selected"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("HEADER first\n", encoding="utf-8")
    second.write_text("HEADER second\n", encoding="utf-8")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "filename,source_path,result,reason\n"
        f"same.pdb,{first},BARREL,OK\n"
        f"same.pdb,{second},BARREL,OK\n",
        encoding="utf-8",
    )

    exit_code = copy_ok.main(
        [
            "--csv",
            str(csv_path),
            "--src",
            str(source_dir),
            "--dst",
            str(destination),
            "--no-recursive-search",
        ]
    )

    copied_texts = sorted(path.read_text(encoding="utf-8") for path in destination.glob("*.pdb"))
    assert exit_code == 0
    assert copied_texts == ["HEADER first\n", "HEADER second\n"]


def test_copy_ok_returns_nonzero_when_selected_files_are_missing(tmp_path: Path):
    copy_ok = importlib.import_module("scripts.copy_ok")
    source_dir = tmp_path / "structures"
    source_dir.mkdir()
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("filename,result,reason\nmissing.pdb,BARREL,OK\n", encoding="utf-8")

    exit_code = copy_ok.main(
        [
            "--csv",
            str(csv_path),
            "--src",
            str(source_dir),
            "--dst",
            str(tmp_path / "selected"),
        ]
    )

    assert exit_code == 1


def test_blast_annotation_structure_lookup_rejects_unsafe_and_ambiguous_paths(
    tmp_path: Path,
):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    structures_dir = tmp_path / "structures"
    (structures_dir / "a").mkdir(parents=True)
    (structures_dir / "b").mkdir(parents=True)
    (structures_dir / "a" / "same.pdb").write_text("HEADER a\n", encoding="utf-8")
    (structures_dir / "b" / "same.pdb").write_text("HEADER b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe"):
        annotate.resolve_structure_path(
            structures_dir,
            "../same.pdb",
            recursive=True,
            cache={},
        )

    with pytest.raises(ValueError, match="Ambiguous"):
        annotate.resolve_structure_path(
            structures_dir,
            "same.pdb",
            recursive=True,
            cache={},
        )


def test_blast_candidate_builder_keeps_duplicate_basenames_when_source_path_differs(
    tmp_path: Path,
):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    structures_dir = tmp_path / "structures"
    first = structures_dir / "a" / "same.pdb"
    second = structures_dir / "b" / "same.pdb"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    pdb_text = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C
ATOM      3  N   GLY A   2       2.000   0.000   0.000  1.00 80.00           N
ATOM      4  CA  GLY A   2       3.000   0.000   0.000  1.00 80.00           C
END
"""
    first.write_text(pdb_text, encoding="utf-8")
    second.write_text(pdb_text, encoding="utf-8")

    candidates, sequences, duplicates = annotate.build_candidates(
        [
            {"filename": "same.pdb", "source_path": str(first), "chain": "A"},
            {"filename": "same.pdb", "source_path": str(second), "chain": "A"},
        ],
        structures_dir,
        min_query_length=1,
        recursive=False,
    )

    assert duplicates == 0
    assert len(candidates) == 2
    assert len(sequences) == 2
    assert candidates[0].query_id != candidates[1].query_id


def test_blast_annotation_writes_headers_for_empty_outputs(tmp_path: Path):
    annotate = importlib.import_module("scripts.annotate_bfvd_candidates_blastp")
    candidates_csv = tmp_path / "candidates.csv"
    annotations_csv = tmp_path / "annotations.csv"

    annotate.write_candidate_manifest([], candidates_csv)
    annotate.annotate_candidates(
        [],
        {},
        type(
            "Args",
            (),
            {"hit_evalue": 1e-5, "min_pident": 25.0, "min_qcov": 50.0},
        )(),
        annotations_csv,
    )

    assert candidates_csv.read_text(encoding="utf-8").startswith("query_id,filename,chain")
    assert annotations_csv.read_text(encoding="utf-8").startswith(
        "query_id,filename,chain,source_path"
    )
