from __future__ import annotations

from pathlib import Path

from cooper_beta.config import build_config
from cooper_beta.provenance import build_run_manifest


def test_run_manifest_records_input_file_hash(tmp_path: Path):
    input_file = tmp_path / "model.pdb"
    input_file.write_text("HEADER\n", encoding="utf-8")

    manifest = build_run_manifest(
        config=build_config({"input.path": str(input_file)}),
        input_files=[str(input_file)],
        output_path=str(tmp_path / "results.csv"),
    )

    state = manifest["input_file_state"][0]
    assert state["path"] == str(input_file.resolve())
    assert state["exists"] is True
    assert state["size"] == len("HEADER\n")
    assert isinstance(state["sha256"], str)
    assert len(state["sha256"]) == 64
