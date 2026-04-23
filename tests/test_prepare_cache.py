from __future__ import annotations

from pathlib import Path

from cooper_beta.config import build_config
from cooper_beta.pipeline_workers import prepare_one_file


class FakeChain:
    def __init__(self, chain_id: str):
        self.id = chain_id


class FakeLoader:
    calls = 0

    def __init__(self, file_path, model_id=0, dssp_bin=None, fail_on_dssp_error=True):
        del model_id, dssp_bin, fail_on_dssp_error
        type(self).calls += 1
        self.file_path = file_path
        self.model = [FakeChain("A")]

    def get_chain_data(self, chain_id):
        assert chain_id == "A"
        return [
            {"res_id": 1, "coord": [0.0, 0.0, 0.0], "is_sheet": True},
            {"res_id": 2, "coord": [1.0, 0.0, 0.0], "is_sheet": True},
        ]


def test_prepare_one_file_reuses_cached_payloads(tmp_path: Path, monkeypatch):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n")

    cfg = build_config(
        {
            "input.min_chain_residues": 1,
            "runtime.prepare_cache_enabled": True,
            "runtime.prepare_cache_dir": str(tmp_path / "cache"),
        }
    )

    monkeypatch.setattr("cooper_beta.pipeline_workers.ProteinLoader", FakeLoader)
    FakeLoader.calls = 0

    first = prepare_one_file(str(input_file), cfg)
    second = prepare_one_file(str(input_file), cfg)

    assert isinstance(first, list)
    assert first == second
    assert FakeLoader.calls == 1


def test_prepare_cache_invalidates_when_file_changes(tmp_path: Path, monkeypatch):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n")

    cfg = build_config(
        {
            "input.min_chain_residues": 1,
            "runtime.prepare_cache_enabled": True,
            "runtime.prepare_cache_dir": str(tmp_path / "cache"),
        }
    )

    monkeypatch.setattr("cooper_beta.pipeline_workers.ProteinLoader", FakeLoader)
    FakeLoader.calls = 0

    first = prepare_one_file(str(input_file), cfg)
    input_file.write_text("HEADER UPDATED\n")
    second = prepare_one_file(str(input_file), cfg)

    assert isinstance(first, list)
    assert isinstance(second, list)
    assert FakeLoader.calls == 2
