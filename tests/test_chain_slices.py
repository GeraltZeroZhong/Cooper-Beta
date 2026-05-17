from __future__ import annotations

from pathlib import Path

import numpy as np

import cooper_beta.chain_slices as chain_slices
from cooper_beta import ChainSliceBundle, extract_chain_slices
from cooper_beta.config import AppConfig
from cooper_beta.models import AnalysisReport, LayerDiagnostic


def _ring_slice(point_count: int = 8) -> dict[float, list[tuple[float, ...]]]:
    angles = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    return {
        0.0: [
            (10.0 * np.cos(angle), 10.0 * np.sin(angle), float(index) + 0.5, float(index))
            for index, angle in enumerate(angles)
        ]
    }


def _install_loader_stub(monkeypatch, residues_data):
    class DummyLoader:
        def __init__(
            self,
            file_path,
            *,
            model_id=0,
            dssp_bin=None,
            fail_on_dssp_error=True,
            strict_chain=True,
        ):
            self.file_path = file_path
            self.model_id = model_id
            self.dssp_bin = dssp_bin
            self.fail_on_dssp_error = fail_on_dssp_error
            self.strict_chain = strict_chain

        def get_chain_data(self, chain_id, *, strict_chain=None):
            assert chain_id == "A"
            assert strict_chain is True
            return residues_data

    monkeypatch.setattr(chain_slices, "ProteinLoader", DummyLoader)


def test_extract_chain_slices_returns_requested_artifacts(monkeypatch, tmp_path: Path):
    cfg = AppConfig()
    residues_data = [
        {"coord": (float(index), float(index % 3), float(index * 2)), "is_sheet": True}
        for index in range(8)
    ]
    raw_slices = _ring_slice()
    _install_loader_stub(monkeypatch, residues_data)
    monkeypatch.setattr(
        chain_slices,
        "select_alignment_slices",
        lambda aligner, coordinates, residues, slicer, cfg: raw_slices,
    )

    bundle = extract_chain_slices(
        tmp_path / "toy.pdb",
        chain="A",
        config=cfg,
        include_raw_axis_slices=True,
        include_core_slices=True,
        include_layer_diagnostics=True,
    )

    assert isinstance(bundle, ChainSliceBundle)
    assert bundle.filename == "toy.pdb"
    assert bundle.chain == "A"
    assert bundle.chain_residues == 8
    assert bundle.sheet_residues == 8
    assert bundle.informative_slices == 1
    assert bundle.raw_axis_slices == raw_slices
    assert bundle.core_slices is not None
    assert bundle.core_slices[0.0] == [point[:3] for point in raw_slices[0.0]]
    assert bundle.layer_diagnostics is not None
    assert len(bundle.layer_diagnostics) == 1
    assert isinstance(bundle.layer_diagnostics[0], LayerDiagnostic)
    assert bundle.layer_diagnostics[0].valid is True
    assert isinstance(bundle.analysis_report, AnalysisReport)


def test_extract_chain_slices_honors_include_flags(monkeypatch, tmp_path: Path):
    cfg = AppConfig()
    residues_data = [
        {"coord": (float(index), float(index % 2), float(index)), "is_sheet": True}
        for index in range(8)
    ]
    _install_loader_stub(monkeypatch, residues_data)
    monkeypatch.setattr(
        chain_slices,
        "select_alignment_slices",
        lambda aligner, coordinates, residues, slicer, cfg: _ring_slice(),
    )

    bundle = extract_chain_slices(tmp_path / "toy.pdb", chain="A", config=cfg)

    assert bundle.raw_axis_slices is None
    assert bundle.core_slices is None
    assert bundle.layer_diagnostics is None
    assert bundle.analysis_report is None
    assert bundle.informative_slices == 1


def test_extract_chain_slices_returns_empty_bundle_with_too_few_sheet_residues(
    monkeypatch,
    tmp_path: Path,
):
    cfg = AppConfig()
    residues_data = [
        {"coord": (0.0, 0.0, 0.0), "is_sheet": True},
        {"coord": (1.0, 0.0, 0.0), "is_sheet": True},
        {"coord": (2.0, 0.0, 0.0), "is_sheet": False},
    ]
    _install_loader_stub(monkeypatch, residues_data)

    bundle = extract_chain_slices(
        tmp_path / "short.pdb",
        chain="A",
        config=cfg,
        include_raw_axis_slices=True,
        include_layer_diagnostics=True,
    )

    assert bundle.raw_axis_slices == {}
    assert bundle.informative_slices == 0
    assert bundle.sheet_residues == 2
    assert bundle.layer_diagnostics == []
    assert bundle.analysis_report is not None
    assert bundle.analysis_report.message == "No active slices were found."
