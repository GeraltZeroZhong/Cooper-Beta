from __future__ import annotations

import numpy as np
import pytest

from cooper_beta.analysis_utils import angular_gap_stats, sequence_angle_order_stats
from cooper_beta.analyzer import BarrelAnalyzer
from cooper_beta.config import AnalyzerConfig, AngleOrderRuleConfig
from cooper_beta.slicer import ProteinSlicer

STRICT_ORDER_CONFIG = AngleOrderRuleConfig(
    enabled=True,
    local_step_max=1,
    min_local_frac=1.0,
    max_mean_circ_dist_norm=0.0,
)


def test_angular_and_order_stats_recenter_after_filtering_outlier():
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring = np.column_stack(
        [
            10.0 * np.cos(angles),
            10.0 * np.sin(angles),
            np.arange(8, dtype=float) + 0.5,
        ]
    )
    outlier = np.array([[100.0, 0.0, 8.5]])
    points = np.vstack([ring, outlier])

    gap_stats = angular_gap_stats(points[:, :2])
    assert gap_stats is not None
    assert gap_stats[0] == pytest.approx(45.0)
    assert gap_stats[2] == 8
    assert gap_stats[3] == pytest.approx(0.0, abs=1e-9)
    assert gap_stats[4] == pytest.approx(0.0, abs=1e-9)

    order_stats = sequence_angle_order_stats(points, STRICT_ORDER_CONFIG)
    assert order_stats is not None
    assert order_stats["order_used_n"] == 8
    assert order_stats["order_local_frac"] == pytest.approx(1.0)
    assert order_stats["order_mean_circ_dist_norm"] == pytest.approx(0.0)


def test_slice_structure_uses_only_segments_inside_same_sheet_run():
    slicer = ProteinSlicer(step_size=1.0, fill_sheet_hole_length=0)
    aligned_coords = np.array(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    residues_data = [
        {"is_sheet": False},
        {"is_sheet": True},
        {"is_sheet": True},
        {"is_sheet": False},
    ]

    slices = slicer.slice_structure(aligned_coords, residues_data)

    assert sorted(slices) == [0.0, 1.0]
    assert slices[0.0] == [(1.0, 0.0, 1.5, 0.0)]
    assert slices[1.0] == [(2.0, 0.0, 1.5, 0.0)]


def test_sequence_angle_order_collapses_multiple_intersections_per_strand():
    strand_angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    points = []
    for strand_id, angle in enumerate(strand_angles):
        seq_pos = float(strand_id) + 0.5
        for radius in (9.5, 10.5):
            points.append(
                (
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    seq_pos,
                    float(strand_id),
                )
            )

    order_stats = sequence_angle_order_stats(np.asarray(points, dtype=float), STRICT_ORDER_CONFIG)

    assert order_stats is not None
    assert order_stats["order_used_n"] == 6
    assert order_stats["order_local_frac"] == pytest.approx(1.0)
    assert order_stats["order_mean_circ_dist_norm"] == pytest.approx(0.0)


def test_analyzer_trims_terminal_sequence_outlier_when_core_is_more_barrel_like():
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    base = [
        (12.0 * np.cos(angle), 8.0 * np.sin(angle), float(index) + 0.5, float(index))
        for index, angle in enumerate(angles)
    ]
    points = base + [(30.0, 0.0, 12.5, 12.0)]

    report = BarrelAnalyzer(AnalyzerConfig()).analyze({0.0: points})
    layer = report["layer_details"][0]

    assert layer["valid"] is True
    assert layer["reason"] == "OK"
    assert layer["raw_n_points"] == 13
    assert layer["collapsed_n_points"] == 13
    assert layer["n_points"] == 12
    assert layer["seq_core_trim_left"] == 0
    assert layer["seq_core_trim_right"] == 1
    assert layer["fit"]["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert layer["fit"]["a"] == pytest.approx(12.0, abs=1e-6)
    assert layer["fit"]["b"] == pytest.approx(8.0, abs=1e-6)


def test_analyzer_keeps_clean_slice_without_terminal_trimming():
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    points = [
        (12.0 * np.cos(angle), 8.0 * np.sin(angle), float(index) + 0.5, float(index))
        for index, angle in enumerate(angles)
    ]

    report = BarrelAnalyzer(AnalyzerConfig()).analyze({0.0: points})
    layer = report["layer_details"][0]

    assert layer["valid"] is True
    assert layer["reason"] == "OK"
    assert layer["n_points"] == 12
    assert layer["seq_core_trim_left"] == 0
    assert layer["seq_core_trim_right"] == 0
