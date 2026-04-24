from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import cooper_beta.pipeline_workers as pipeline_workers
from cooper_beta.config import AppConfig
from cooper_beta.constants import RESULT_BARREL, RESULT_FILTERED_OUT, RESULT_NON_BARREL


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
    cfg.analyzer.decision.min_scored_layers = 0

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


def test_analyze_chain_payload_requires_minimum_scored_layers(monkeypatch: pytest.MonkeyPatch):
    cfg = AppConfig()
    cfg.input.min_chain_residues = 4
    cfg.input.min_sheet_residues = 2
    cfg.input.min_informative_slices = 4
    cfg.analyzer.decision.barrel_valid_ratio = 0.60
    cfg.analyzer.decision.min_scored_layers = 5

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
        "layer_details": [{"reason": "OK"} for _ in range(4)],
    }
    _install_analysis_stubs(monkeypatch, report=report, slice_count=4)

    row = pipeline_workers.analyze_chain_payload(payload, cfg)

    assert row["result"] == RESULT_NON_BARREL
    assert row["decision_score"] == pytest.approx(0.75)
    assert row["reason"] == (
        "Too few scored slices for a stable decision "
        "(4/4 = 1.00, need > 0.31 and >= 5 layers)"
    )


def test_analyze_chain_payload_blocks_short_low_sheet_wide_radius(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = AppConfig()
    cfg.input.min_chain_residues = 20
    cfg.input.min_sheet_residues = 10
    cfg.input.min_informative_slices = 4
    cfg.analyzer.decision.barrel_valid_ratio = 0.85
    cfg.analyzer.decision.min_scored_layer_frac = 0.31
    cfg.analyzer.decision.min_scored_layers = 7

    payload = {
        "filename": "wide-short-chain.pdb",
        "chain": "A",
        "residues_data": [
            _residue(float(index), 0.0, float(index), is_sheet=index < 86)
            for index in range(174)
        ],
    }
    report = {
        "score": 15.0 / 26.0,
        "score_adjust": 1.0,
        "valid_layers": 15,
        "total_layers": 26,
        "total_scored_layers": 15,
        "avg_radius": 16.5,
        "layer_details": [{"reason": "OK"} for _ in range(15)],
    }
    _install_analysis_stubs(monkeypatch, report=report, slice_count=26)

    row = pipeline_workers.analyze_chain_payload(payload, cfg)

    assert row["result"] == RESULT_NON_BARREL
    assert row["decision_score"] == pytest.approx(1.0)
    assert row["reason"].startswith(
        "Short low-sheet chain has an unusually large fitted barrel radius"
    )


def test_exception_layer_switch_disables_low_sheet_wide_guard(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = AppConfig()
    cfg.input.min_chain_residues = 20
    cfg.input.min_sheet_residues = 10
    cfg.input.min_informative_slices = 4
    cfg.analyzer.decision.barrel_valid_ratio = 0.85
    cfg.analyzer.decision.min_scored_layer_frac = 0.31
    cfg.analyzer.decision.min_scored_layers = 7
    cfg.analyzer.decision.exception_layer_enabled = False

    payload = {
        "filename": "wide-short-chain.pdb",
        "chain": "A",
        "residues_data": [
            _residue(float(index), 0.0, float(index), is_sheet=index < 86)
            for index in range(174)
        ],
    }
    report = {
        "score": 15.0 / 26.0,
        "score_adjust": 1.0,
        "valid_layers": 15,
        "total_layers": 26,
        "total_scored_layers": 15,
        "avg_radius": 16.5,
        "layer_details": [{"reason": "OK"} for _ in range(15)],
    }
    _install_analysis_stubs(monkeypatch, report=report, slice_count=26)

    row = pipeline_workers.analyze_chain_payload(payload, cfg)

    assert row["result"] == RESULT_BARREL
    assert row["reason"] == "OK"


def test_analyze_chain_payload_rescues_high_confidence_small_barrel(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = AppConfig()
    cfg.input.min_chain_residues = 4
    cfg.input.min_sheet_residues = 2
    cfg.input.min_informative_slices = 4
    cfg.analyzer.decision.barrel_valid_ratio = 0.625001
    cfg.analyzer.decision.min_scored_layer_frac = 0.04
    cfg.analyzer.decision.min_scored_layers = 7
    cfg.analyzer.decision.small_barrel_rescue.enabled = True
    cfg.analyzer.decision.small_barrel_rescue.min_score = 0.999
    cfg.analyzer.decision.small_barrel_rescue.min_scored_layers = 5
    cfg.analyzer.decision.small_barrel_rescue.min_total_layers = 25
    cfg.analyzer.decision.small_barrel_rescue.max_avg_radius = 10.5

    payload = {
        "filename": "small-barrel.pdb",
        "chain": "A",
        "residues_data": [
            _residue(0.0, 0.0, 0.0, is_sheet=True),
            _residue(1.0, 0.0, 1.0, is_sheet=True),
            _residue(2.0, 0.0, 2.0, is_sheet=True),
            _residue(3.0, 0.0, 3.0, is_sheet=False),
        ],
    }
    report = {
        "score": 5.0 / 30.0,
        "score_adjust": 1.0,
        "valid_layers": 5,
        "total_layers": 30,
        "total_scored_layers": 5,
        "avg_radius": 8.0,
        "layer_details": [{"reason": "OK"} for _ in range(5)],
    }
    _install_analysis_stubs(monkeypatch, report=report, slice_count=30)

    row = pipeline_workers.analyze_chain_payload(payload, cfg)

    assert row["result"] == RESULT_BARREL
    assert row["reason"] == "OK"
    assert row["decision_score"] == pytest.approx(1.0)
    assert row["scored_layers"] == 5


def test_exception_layer_switch_disables_small_barrel_rescue(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = AppConfig()
    cfg.input.min_chain_residues = 4
    cfg.input.min_sheet_residues = 2
    cfg.input.min_informative_slices = 4
    cfg.analyzer.decision.barrel_valid_ratio = 0.625001
    cfg.analyzer.decision.min_scored_layer_frac = 0.04
    cfg.analyzer.decision.min_scored_layers = 7
    cfg.analyzer.decision.exception_layer_enabled = False
    cfg.analyzer.decision.small_barrel_rescue.enabled = True
    cfg.analyzer.decision.small_barrel_rescue.min_score = 0.999
    cfg.analyzer.decision.small_barrel_rescue.min_scored_layers = 5
    cfg.analyzer.decision.small_barrel_rescue.min_total_layers = 25
    cfg.analyzer.decision.small_barrel_rescue.max_avg_radius = 10.5

    payload = {
        "filename": "small-barrel.pdb",
        "chain": "A",
        "residues_data": [
            _residue(0.0, 0.0, 0.0, is_sheet=True),
            _residue(1.0, 0.0, 1.0, is_sheet=True),
            _residue(2.0, 0.0, 2.0, is_sheet=True),
            _residue(3.0, 0.0, 3.0, is_sheet=False),
        ],
    }
    report = {
        "score": 5.0 / 30.0,
        "score_adjust": 1.0,
        "valid_layers": 5,
        "total_layers": 30,
        "total_scored_layers": 5,
        "avg_radius": 8.0,
        "layer_details": [{"reason": "OK"} for _ in range(5)],
    }
    _install_analysis_stubs(monkeypatch, report=report, slice_count=30)

    row = pipeline_workers.analyze_chain_payload(payload, cfg)

    assert row["result"] == RESULT_NON_BARREL
    assert row["reason"] == (
        "Too few scored slices for a stable decision "
        "(5/30 = 0.17, need > 0.04 and >= 7 layers)"
    )


def test_analyze_chain_payload_rescues_compact_high_confidence_barrel(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = AppConfig()
    cfg.input.min_chain_residues = 20
    cfg.input.min_sheet_residues = 10
    cfg.input.min_informative_slices = 4
    cfg.analyzer.decision.barrel_valid_ratio = 0.625001
    cfg.analyzer.decision.min_scored_layer_frac = 0.04
    cfg.analyzer.decision.min_scored_layers = 7
    cfg.analyzer.decision.small_barrel_rescue.enabled = True
    cfg.analyzer.decision.small_barrel_rescue.compact_enabled = True

    payload = {
        "filename": "compact-barrel.pdb",
        "chain": "A",
        "residues_data": [
            _residue(float(index), 0.0, float(index), is_sheet=index < 63)
            for index in range(210)
        ],
    }
    report = {
        "score": 4.0 / 19.0,
        "score_adjust": 1.0,
        "valid_layers": 4,
        "total_layers": 19,
        "total_scored_layers": 4,
        "avg_radius": 12.3,
        "layer_details": [{"reason": "OK"} for _ in range(4)],
    }
    _install_analysis_stubs(monkeypatch, report=report, slice_count=19)

    row = pipeline_workers.analyze_chain_payload(payload, cfg)

    assert row["result"] == RESULT_BARREL
    assert row["reason"] == "OK"
    assert row["scored_layers"] == 4


def test_analyze_chain_payload_rescues_sparse_high_confidence_barrel(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = AppConfig()
    cfg.input.min_chain_residues = 20
    cfg.input.min_sheet_residues = 10
    cfg.input.min_informative_slices = 4
    cfg.analyzer.decision.barrel_valid_ratio = 0.625001
    cfg.analyzer.decision.min_scored_layer_frac = 0.04
    cfg.analyzer.decision.min_scored_layers = 7
    cfg.analyzer.decision.small_barrel_rescue.enabled = True
    cfg.analyzer.decision.small_barrel_rescue.sparse_enabled = True

    payload = {
        "filename": "sparse-barrel.pdb",
        "chain": "A",
        "residues_data": [
            _residue(float(index), 0.0, float(index), is_sheet=index < 76)
            for index in range(170)
        ],
    }
    report = {
        "score": 3.0 / 35.0,
        "score_adjust": 1.0,
        "valid_layers": 3,
        "total_layers": 35,
        "total_scored_layers": 3,
        "avg_radius": 8.8,
        "layer_details": [{"reason": "OK"} for _ in range(3)],
    }
    _install_analysis_stubs(monkeypatch, report=report, slice_count=35)

    row = pipeline_workers.analyze_chain_payload(payload, cfg)

    assert row["result"] == RESULT_BARREL
    assert row["reason"] == "OK"
    assert row["scored_layers"] == 3


def test_analyze_chain_payload_rescues_soft_nn_near_miss(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = AppConfig()
    cfg.input.min_chain_residues = 20
    cfg.input.min_sheet_residues = 10
    cfg.input.min_informative_slices = 4
    cfg.analyzer.decision.barrel_valid_ratio = 0.625001
    cfg.analyzer.decision.min_scored_layer_frac = 0.04
    cfg.analyzer.decision.min_scored_layers = 7
    cfg.analyzer.decision.near_miss_rescue.enabled = True
    cfg.analyzer.decision.near_miss_rescue.soft_nn_enabled = True
    cfg.analyzer.rules.angle.max_gap_deg = 160.0
    cfg.analyzer.rules.angle.order.max_mean_circ_dist_norm = 0.10

    payload = {
        "filename": "soft-nn-barrel.pdb",
        "chain": "A",
        "residues_data": [
            _residue(float(index), 0.0, float(index), is_sheet=index < 69)
            for index in range(238)
        ],
    }
    soft_nn_layer = {
        "reason": "JUNK(irregular nearest-neighbor spacing)",
        "nn_inlier_frac": 0.714,
        "nn_cv": 0.10,
        "angle_max_gap_deg": 90.0,
        "order_local_frac": 1.0,
        "order_mean_circ_dist_norm": 0.0,
    }
    report = {
        "score": 0.0,
        "score_adjust": 0.0,
        "valid_layers": 0,
        "total_layers": 35,
        "total_scored_layers": 0,
        "avg_radius": 0.0,
        "layer_details": [soft_nn_layer.copy() for _ in range(3)],
    }
    _install_analysis_stubs(monkeypatch, report=report, slice_count=35)

    row = pipeline_workers.analyze_chain_payload(payload, cfg)

    assert row["result"] == RESULT_BARREL
    assert row["reason"] == "OK"


def test_analyze_chain_payload_does_not_soft_nn_rescue_scored_sparse_barrel(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = AppConfig()
    cfg.input.min_chain_residues = 20
    cfg.input.min_sheet_residues = 10
    cfg.input.min_informative_slices = 4
    cfg.analyzer.decision.barrel_valid_ratio = 0.625001
    cfg.analyzer.decision.min_scored_layer_frac = 0.04
    cfg.analyzer.decision.min_scored_layers = 7
    cfg.analyzer.decision.near_miss_rescue.enabled = True
    cfg.analyzer.decision.near_miss_rescue.soft_nn_enabled = True
    cfg.analyzer.rules.angle.max_gap_deg = 160.0
    cfg.analyzer.rules.angle.order.max_mean_circ_dist_norm = 0.10

    payload = {
        "filename": "sparse-should-stay-blocked.pdb",
        "chain": "A",
        "residues_data": [
            _residue(float(index), 0.0, float(index), is_sheet=index < 60)
            for index in range(238)
        ],
    }
    soft_nn_layer = {
        "reason": "JUNK(irregular nearest-neighbor spacing)",
        "nn_inlier_frac": 0.714,
        "nn_cv": 0.10,
        "angle_max_gap_deg": 90.0,
        "order_local_frac": 1.0,
        "order_mean_circ_dist_norm": 0.0,
    }
    report = {
        "score": 4.0 / 35.0,
        "score_adjust": 1.0,
        "valid_layers": 4,
        "total_layers": 35,
        "total_scored_layers": 4,
        "avg_radius": 7.5,
        "layer_details": [soft_nn_layer.copy() for _ in range(3)],
    }
    _install_analysis_stubs(monkeypatch, report=report, slice_count=35)

    row = pipeline_workers.analyze_chain_payload(payload, cfg)

    assert row["result"] == RESULT_NON_BARREL


def test_analyze_chain_payload_rescues_compact_partner_near_miss(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = AppConfig()
    cfg.input.min_chain_residues = 20
    cfg.input.min_sheet_residues = 10
    cfg.input.min_informative_slices = 4
    cfg.analyzer.decision.barrel_valid_ratio = 0.625001
    cfg.analyzer.decision.min_scored_layer_frac = 0.04
    cfg.analyzer.decision.min_scored_layers = 7
    cfg.analyzer.decision.near_miss_rescue.enabled = True
    cfg.analyzer.decision.near_miss_rescue.compact_partner_enabled = True

    payload = {
        "filename": "compact-partner.pdb",
        "chain": "A",
        "residues_data": [
            _residue(float(index), 0.0, float(index), is_sheet=index < 71)
            for index in range(195)
        ],
    }
    report = {
        "score": 1.0 / 18.0,
        "score_adjust": 0.5,
        "valid_layers": 1,
        "total_layers": 18,
        "total_scored_layers": 2,
        "avg_radius": 9.3,
        "layer_details": [
            {"reason": "OK"},
            {"reason": "Local sequence/angle order mismatch"},
        ],
    }
    _install_analysis_stubs(monkeypatch, report=report, slice_count=18)

    row = pipeline_workers.analyze_chain_payload(payload, cfg)

    assert row["result"] == RESULT_BARREL
    assert row["reason"] == "OK"


def test_analyze_chain_payload_rescues_large_partner_near_miss(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = AppConfig()
    cfg.input.min_chain_residues = 20
    cfg.input.min_sheet_residues = 10
    cfg.input.min_informative_slices = 4
    cfg.analyzer.decision.barrel_valid_ratio = 0.625001
    cfg.analyzer.decision.min_scored_layer_frac = 0.04
    cfg.analyzer.decision.min_scored_layers = 7
    cfg.analyzer.decision.near_miss_rescue.enabled = True
    cfg.analyzer.decision.near_miss_rescue.large_partner_enabled = True

    payload = {
        "filename": "large-partner.pdb",
        "chain": "A",
        "residues_data": [
            _residue(float(index), 0.0, float(index), is_sheet=index < 256)
            for index in range(724)
        ],
    }
    report = {
        "score": 16.0 / 65.0,
        "score_adjust": 16.0 / 29.0,
        "valid_layers": 16,
        "total_layers": 65,
        "total_scored_layers": 29,
        "avg_radius": 20.0,
        "layer_details": [{"reason": "OK"} for _ in range(16)],
    }
    _install_analysis_stubs(monkeypatch, report=report, slice_count=65)

    row = pipeline_workers.analyze_chain_payload(payload, cfg)

    assert row["result"] == RESULT_BARREL
    assert row["reason"] == "OK"


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
        lambda slices, minimum_points, cfg: {
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


def test_select_alignment_slices_prefers_geometry_over_dense_bad_axis(
    monkeypatch: pytest.MonkeyPatch,
):
    class DummyAligner:
        center = np.zeros(3, dtype=float)
        rotation_matrix = np.eye(3, dtype=float)

        def transform(self, coordinates):
            return np.asarray(coordinates, dtype=float)

    class DummySlicer:
        def slice_structure(self, aligned_coordinates, residues_data):
            label = int(round(float(np.max(np.asarray(aligned_coordinates, dtype=float)[:, 2]))))
            point_count = {1: 8, 2: 12, 9: 10}[label]
            return {
                0.0: [
                    (float(label), float(index), float(index) + 0.5, float(index))
                    for index in range(point_count)
                ]
            }

    monkeypatch.setattr(
        pipeline_workers,
        "_candidate_axis_rotations",
        lambda rotation_matrix: [
            np.diag([1.0, 1.0, 1.0]),
            np.diag([1.0, 1.0, 2.0]),
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
        "nearest_neighbor_spacing_stats",
        lambda points_xy: (1.0, 0.0, 0.0, 1.0),
    )
    monkeypatch.setattr(
        pipeline_workers,
        "angular_gap_stats",
        lambda points_xy: (
            {1: 100.0, 2: 160.0, 9: 100.0}[int(round(float(points_xy[0, 0])))],
            0.0,
            int(points_xy.shape[0]),
            0.0,
            0.0,
        ),
    )
    monkeypatch.setattr(
        pipeline_workers,
        "sequence_angle_order_stats",
        lambda points, order_cfg: {
            "order_local_frac": {1: 1.0, 2: 0.5, 9: 1.0}[int(round(float(points[0, 0])))],
            "order_mean_circ_dist_norm": {1: 0.0, 2: 0.3, 9: 0.0}[int(round(float(points[0, 0])))],
        },
    )

    cfg = AppConfig()
    cfg.analyzer.axis_search.enabled = True
    cfg.analyzer.axis_search.refine.enabled = True
    cfg.analyzer.axis_search.refine.angle_deg = 5.0
    cfg.analyzer.fit.min_points_per_slice = 7
    cfg.analyzer.decision.min_intersections_for_scoring = 7
    cfg.analyzer.rules.angle.max_gap_deg = 120.0
    cfg.analyzer.rules.angle.order.min_local_frac = 1.0
    cfg.analyzer.rules.angle.order.max_mean_circ_dist_norm = 0.0

    slices = pipeline_workers._select_alignment_slices(
        DummyAligner(),
        np.array([[0.0, 0.0, 1.0]], dtype=float),
        [_residue(0.0, 0.0, 0.0, is_sheet=True)],
        DummySlicer(),
        cfg,
    )

    assert slices[0.0][0][0] == pytest.approx(9.0)
