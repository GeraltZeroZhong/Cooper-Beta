from __future__ import annotations

import numpy as np
import pytest

from cooper_beta.analyzer import BarrelAnalyzer
from cooper_beta.config import AnalyzerConfig, LeastSquaresConfig
from cooper_beta.ellipse import fit_rotated_ellipse


def _ellipse_points(
    *,
    center: tuple[float, float] = (1.5, -2.0),
    axis_a: float = 12.0,
    axis_b: float = 8.0,
    theta: float = 0.35,
    count: int = 12,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x = axis_a * np.cos(angles)
    y = axis_b * np.sin(angles)
    rotated_x = x * cos_theta - y * sin_theta + center[0]
    rotated_y = x * sin_theta + y * cos_theta + center[1]
    return np.column_stack([rotated_x, rotated_y])


def test_fit_rotated_ellipse_reports_geometric_rmse_on_clean_data():
    points = _ellipse_points()

    fit = fit_rotated_ellipse(points, LeastSquaresConfig())

    assert fit is not None
    assert fit["xc"] == pytest.approx(1.5, abs=1e-6)
    assert fit["yc"] == pytest.approx(-2.0, abs=1e-6)
    assert fit["a"] == pytest.approx(12.0, abs=1e-6)
    assert fit["b"] == pytest.approx(8.0, abs=1e-6)
    assert fit["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert fit["rmse_all"] == pytest.approx(0.0, abs=1e-6)
    assert fit["inlier_frac"] == pytest.approx(1.0)


def test_fit_rotated_ellipse_is_robust_to_nonterminal_anomalous_point():
    points = _ellipse_points()
    anomalous = points.copy()
    anomalous[5] = np.array([2.0, -1.0], dtype=float)

    fit = fit_rotated_ellipse(anomalous, LeastSquaresConfig())

    assert fit is not None
    assert fit["xc"] == pytest.approx(1.5, abs=0.2)
    assert fit["yc"] == pytest.approx(-2.0, abs=0.2)
    assert fit["a"] == pytest.approx(12.0, abs=0.3)
    assert fit["b"] == pytest.approx(8.0, abs=0.3)
    assert fit["rmse"] == pytest.approx(fit["rmse_all"])
    assert fit["rmse"] < 2.5
    assert fit["median_abs_dist"] < 0.2
    assert fit["inlier_frac"] == pytest.approx(11.0 / 12.0)


def test_analyzer_keeps_nonterminal_anomaly_without_terminal_trimming():
    points_xy = _ellipse_points(center=(0.0, 0.0), theta=0.0)
    points = [
        (float(x), float(y), float(index) + 0.5, float(index))
        for index, (x, y) in enumerate(points_xy)
    ]
    points[5] = (1.0, 0.5, 5.5, 5.0)

    report = BarrelAnalyzer(AnalyzerConfig()).analyze({0.0: points})
    layer = report["layer_details"][0]

    assert layer["valid"] is True
    assert layer["reason"] == "OK"
    assert layer["seq_core_trim_left"] == 0
    assert layer["seq_core_trim_right"] == 0
    assert layer["fit_inlier_frac"] == pytest.approx(11.0 / 12.0)
    assert layer["fit"]["rmse"] == pytest.approx(layer["fit_rmse_all"])
    assert layer["fit"]["a"] == pytest.approx(12.0, abs=0.3)
    assert layer["fit"]["b"] == pytest.approx(8.0, abs=0.3)


def test_fit_rotated_ellipse_keeps_soft_support_diagnostics_for_multiple_bad_points():
    points = _ellipse_points(center=(0.0, 0.0), theta=0.0)
    for index in (2, 4, 6, 8):
        points[index] = np.array([0.5 * float(index), -0.25 * float(index)], dtype=float)

    fit = fit_rotated_ellipse(points, LeastSquaresConfig())

    assert fit is not None
    assert fit["used_n"] < float(len(points))
    assert fit["inlier_frac"] < 1.0
    assert fit["rmse"] > fit["median_abs_dist"]
