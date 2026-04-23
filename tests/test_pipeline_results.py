from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import cooper_beta.pipeline_workers as pipeline_workers
from cooper_beta.config import AppConfig
from cooper_beta.constants import RESULT_BARREL, RESULT_FILTERED_OUT


def _residue(x: float, y: float, z: float, *, is_sheet: bool) -> dict[str, object]:
    return {"coord": (x, y, z), "is_sheet": is_sheet}


def _install_analysis_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    report: dict[str, object],
    slice_count: int,
) -> None:
    class DummyAligner:
        def fit(self, coordinates):
            self.coordinates = coordinates
            self.center = np.zeros(3, dtype=float)
            self.rotation_matrix = np.eye(3, dtype=float)

        def transform(self, coordinates):
            return coordinates

    class DummySlicer:
        def __init__(self, *, step_size: float, fill_sheet_hole_length: int):
            self.step_size = step_size
            self.fill_sheet_hole_length = fill_sheet_hole_length

        def slice_structure(self, aligned_coordinates, residues_data):
            return {
                float(index): [(float(index), 0.0, float(index) + 0.5, float(index))]
                for index in range(slice_count)
            }

    class DummyAnalyzer:
        def __init__(self, analyzer_cfg):
            self.analyzer_cfg = analyzer_cfg

        def analyze(self, slices):
            return deepcopy(report)

    monkeypatch.setattr(pipeline_workers, "PCAAligner", DummyAligner)
    monkeypatch.setattr(pipeline_workers, "ProteinSlicer", DummySlicer)
    monkeypatch.setattr(pipeline_workers, "BarrelAnalyzer", DummyAnalyzer)


def test_analyze_chain_payload_marks_prefiltered_chain_as_filtered_out():
    cfg = AppConfig()
    payload = {
        "filename": "example.pdb",
        "chain": "A",
        "residues_data": [
            _residue(float(index), 0.0, 0.0, is_sheet=False)
            for index in range(cfg.input.min_chain_residues - 1)
        ],
    }

    row = pipeline_workers.analyze_chain_payload(payload, cfg)

    assert row["result"] == RESULT_FILTERED_OUT
    assert row["result_stage"] == "prefilter"
    assert row["reason"] == (
        f"Chain residues below threshold ({cfg.input.min_chain_residues - 1} < "
        f"{cfg.input.min_chain_residues})"
    )
    assert row["chain_residues"] == cfg.input.min_chain_residues - 1
    assert row["sheet_residues"] == 0
    assert row["informative_slices"] == 0


def test_prepare_one_file_keeps_short_chains_for_filtered_reporting(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = AppConfig()
    short_length = cfg.input.min_chain_residues - 1

    class DummyLoader:
        def __init__(self, file_path: str, dssp_bin=None, fail_on_dssp_error: bool = True):
            self.file_path = file_path
            self.dssp_bin = dssp_bin
            self.fail_on_dssp_error = fail_on_dssp_error
            self.model = [SimpleNamespace(id="A")]

        def get_chain_data(self, chain_id: str):
            assert chain_id == "A"
            return [
                _residue(float(index), 0.0, 0.0, is_sheet=False)
                for index in range(short_length)
            ]

    monkeypatch.setattr(pipeline_workers, "ProteinLoader", DummyLoader)
    monkeypatch.setattr(pipeline_workers, "load_prepare_payloads", lambda file_path, cfg: None)
    monkeypatch.setattr(
        pipeline_workers,
        "store_prepare_payloads",
        lambda file_path, cfg, payloads: None,
    )

    payloads = pipeline_workers.prepare_one_file("example.pdb", cfg)

    assert isinstance(payloads, list)
    assert len(payloads) == 1
    assert payloads[0]["filename"] == "example.pdb"
    assert payloads[0]["chain"] == "A"
    assert len(payloads[0]["residues_data"]) == short_length


def test_analyze_chain_payload_reports_decision_metrics(monkeypatch: pytest.MonkeyPatch):
    cfg = AppConfig()
    cfg.input.min_chain_residues = 4
    cfg.input.min_sheet_residues = 2
    cfg.input.min_informative_slices = 4
    cfg.analyzer.decision.barrel_valid_ratio = 0.60

    payload = {
        "filename": "example.pdb",
        "chain": "A",
        "residues_data": [
            _residue(0.0, 0.0, 0.0, is_sheet=True),
            _residue(1.0, 0.0, 1.0, is_sheet=True),
            _residue(2.0, 0.0, 2.0, is_sheet=True),
            _residue(3.0, 0.0, 3.0, is_sheet=False),
        ],
    }
    report = {
        "score": 0.50,
        "score_adjust": 0.75,
        "valid_layers": 3,
        "total_layers": 4,
        "total_scored_layers": 4,
        "avg_radius": 10.5,
        "layer_details": [
            {"reason": "OK"},
            {"reason": "JUNK(too few intersections: need >= 7)"},
            {"reason": "Nearest-neighbor spacing is irregular"},
            {"reason": "OK"},
        ],
    }
    _install_analysis_stubs(monkeypatch, report=report, slice_count=4)

    row = pipeline_workers.analyze_chain_payload(payload, cfg)

    assert row["result"] == RESULT_BARREL
    assert row["result_stage"] == "decision"
    assert row["reason"] == "OK"
    assert row["decision_basis"] == "adjusted"
    assert row["decision_score"] == pytest.approx(0.75)
    assert row["decision_threshold"] == pytest.approx(0.60)
    assert row["score_raw"] == pytest.approx(0.50)
    assert row["score_adjust"] == pytest.approx(0.75)
    assert row["valid_layers"] == 3
    assert row["scored_layers"] == 4
    assert row["total_layers"] == 4
    assert row["valid_layer_frac"] == pytest.approx(0.75)
    assert row["scored_layer_frac"] == pytest.approx(1.0)
    assert row["junk_layers"] == 1
    assert row["invalid_layers"] == 1
    assert row["avg_radius"] == pytest.approx(10.5)
    assert row["sheet_residues"] == 3
    assert row["informative_slices"] == 4
    assert row["all_adjusted_layers"] == 4
    assert row["all_layers"] == 4


def test_analyze_chain_payload_uses_raw_score_when_configured(monkeypatch: pytest.MonkeyPatch):
    cfg = AppConfig()
    cfg.input.min_chain_residues = 4
    cfg.input.min_sheet_residues = 2
    cfg.input.min_informative_slices = 3
    cfg.analyzer.decision.use_adjusted_score = False
    cfg.analyzer.decision.barrel_valid_ratio = 0.70

    payload = {
        "filename": "example.pdb",
        "chain": "B",
        "residues_data": [
            _residue(0.0, 0.0, 0.0, is_sheet=True),
            _residue(1.0, 0.0, 1.0, is_sheet=True),
            _residue(2.0, 0.0, 2.0, is_sheet=False),
            _residue(3.0, 0.0, 3.0, is_sheet=True),
        ],
    }
    report = {
        "score": 0.80,
        "score_adjust": 0.20,
        "valid_layers": 4,
        "total_layers": 4,
        "total_scored_layers": 4,
        "layer_details": [{"reason": "OK"} for _ in range(4)],
    }
    _install_analysis_stubs(monkeypatch, report=report, slice_count=4)

    row = pipeline_workers.analyze_chain_payload(payload, cfg)

    assert row["result"] == RESULT_BARREL
    assert row["decision_basis"] == "raw"
    assert row["decision_score"] == pytest.approx(0.80)
    assert row["score_adjust"] == pytest.approx(0.20)
    assert row["score_raw"] == pytest.approx(0.80)


def test_select_alignment_slices_prefers_refined_best_axis(monkeypatch: pytest.MonkeyPatch):
    class DummyAligner:
        center = np.zeros(3, dtype=float)
        rotation_matrix = np.eye(3, dtype=float)

        def transform(self, coordinates):
            return np.asarray(coordinates, dtype=float)

    class DummySlicer:
        def slice_structure(self, aligned_coordinates, residues_data):
            label = int(round(float(np.max(np.asarray(aligned_coordinates, dtype=float)[:, 2]))))
            return {0.0: [(float(label), 0.0, 0.5, 0.0)]}

    monkeypatch.setattr(
        pipeline_workers,
        "_candidate_axis_rotations",
        lambda rotation_matrix: [
            np.diag([1.0, 1.0, 1.0]),
            np.diag([1.0, 1.0, 2.0]),
            np.diag([1.0, 1.0, 3.0]),
        ],
    )
    monkeypatch.setattr(
        pipeline_workers,
        "_refinement_rotations",
        lambda base_rotation, angle_deg: [
            base_rotation,
            np.diag([1.0, 1.0, 9.0]),
        ],
    )
    monkeypatch.setattr(
        pipeline_workers,
        "_axis_search_score_key",
        lambda slices, minimum_points: {
            1: (1, 1.0, 1.0, 0),
            2: (4, 4.0, 4.0, 0),
            3: (2, 2.0, 2.0, 0),
            9: (6, 6.0, 6.0, 0),
        }[int(slices[0.0][0][0])],
    )

    cfg = AppConfig()
    cfg.analyzer.axis_search.enabled = True
    cfg.analyzer.axis_search.refine.enabled = True
    cfg.analyzer.axis_search.refine.angle_deg = 5.0

    slices = pipeline_workers._select_alignment_slices(
        DummyAligner(),
        np.array([[0.0, 0.0, 1.0]], dtype=float),
        [_residue(0.0, 0.0, 0.0, is_sheet=True)],
        DummySlicer(),
        cfg,
    )

    assert slices[0.0][0][0] == pytest.approx(9.0)
