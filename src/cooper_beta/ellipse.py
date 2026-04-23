from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .config import LeastSquaresConfig
from .constants import COVARIANCE_FLOOR


def _ellipse_residuals(
    params: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    """Residuals of a rotated implicit ellipse."""
    center_x, center_y, axis_a, axis_b, theta = params

    delta_x = x_values - center_x
    delta_y = y_values - center_y

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotated_x = delta_x * cos_theta + delta_y * sin_theta
    rotated_y = -delta_x * sin_theta + delta_y * cos_theta

    return (rotated_x / axis_a) ** 2 + (rotated_y / axis_b) ** 2 - 1.0


def fit_rotated_ellipse(
    points_xy: np.ndarray,
    least_squares_config: LeastSquaresConfig,
) -> dict[str, float] | None:
    """Fit a rotated ellipse to slice intersections."""
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("`points_xy` must have shape `(N, 2)`.")

    x_values = pts[:, 0]
    y_values = pts[:, 1]

    center_x = float(np.mean(x_values))
    center_y = float(np.mean(y_values))
    centered_points = pts - np.array([center_x, center_y], dtype=float)

    covariance = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    axis_a = float(2.0 * np.sqrt(max(eigenvalues[1], COVARIANCE_FLOOR)))
    axis_b = float(2.0 * np.sqrt(max(eigenvalues[0], COVARIANCE_FLOOR)))
    longest_axis_vector = eigenvectors[:, 1]
    theta = float(np.arctan2(longest_axis_vector[1], longest_axis_vector[0]))

    initial_guess = [center_x, center_y, axis_a, axis_b, theta]

    try:
        result = least_squares(
            _ellipse_residuals,
            initial_guess,
            args=(x_values, y_values),
            method=least_squares_config.method,
            loss=least_squares_config.loss,
            f_scale=float(least_squares_config.f_scale),
        )
    except Exception:
        return None

    fitted_center_x, fitted_center_y, fitted_axis_a, fitted_axis_b, fitted_theta = result.x
    fitted_axis_a = abs(float(fitted_axis_a))
    fitted_axis_b = abs(float(fitted_axis_b))
    if fitted_axis_b > fitted_axis_a:
        fitted_axis_a, fitted_axis_b = fitted_axis_b, fitted_axis_a

    rmse = float(np.sqrt(np.mean(result.fun**2)))
    return {
        "xc": float(fitted_center_x),
        "yc": float(fitted_center_y),
        "a": float(fitted_axis_a),
        "b": float(fitted_axis_b),
        "theta": float(fitted_theta),
        "rmse": float(rmse),
    }
