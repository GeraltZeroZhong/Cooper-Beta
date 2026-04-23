from __future__ import annotations

from math import pi

import numpy as np
from scipy.optimize import least_squares

from .config import LeastSquaresConfig
from .constants import COVARIANCE_FLOOR, ROBUST_SIGMA_SCALE

_EPS = 1e-12
_ANGLE_SEEDS = (-0.5 * pi, 0.0, 0.5 * pi, pi)
_NEWTON_MAX_ITER = 32
_NEWTON_STEP_CLIP = 0.5


def _wrap_angle(theta: float) -> float:
    """Wrap an angle to ``[-pi, pi)`` for stable reporting."""
    return float((theta + pi) % (2.0 * pi) - pi)


def _decode_params(raw_params: np.ndarray) -> tuple[float, float, float, float, float]:
    """Decode unconstrained parameters into a positive-axis ellipse."""
    center_x = float(raw_params[0])
    center_y = float(raw_params[1])
    axis_a = float(np.exp(np.clip(raw_params[2], -20.0, 20.0)))
    axis_b = float(np.exp(np.clip(raw_params[3], -20.0, 20.0)))
    theta = float(raw_params[4])
    return center_x, center_y, axis_a, axis_b, theta


def _canonicalize_ellipse(
    center_x: float,
    center_y: float,
    axis_a: float,
    axis_b: float,
    theta: float,
) -> tuple[float, float, float, float, float]:
    """Return the same ellipse with ``a >= b`` and canonical orientation."""
    axis_a = max(float(axis_a), np.sqrt(COVARIANCE_FLOOR))
    axis_b = max(float(axis_b), np.sqrt(COVARIANCE_FLOOR))
    if axis_b > axis_a:
        axis_a, axis_b = axis_b, axis_a
        theta += 0.5 * pi
    return float(center_x), float(center_y), float(axis_a), float(axis_b), _wrap_angle(theta)


def _rotate_points(
    points_xy: np.ndarray,
    center_x: float,
    center_y: float,
    theta: float,
) -> np.ndarray:
    """Translate to the ellipse center and rotate into the ellipse frame."""
    pts = np.asarray(points_xy, dtype=float)
    delta_x = pts[:, 0] - float(center_x)
    delta_y = pts[:, 1] - float(center_y)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotated_x = delta_x * cos_theta + delta_y * sin_theta
    rotated_y = -delta_x * sin_theta + delta_y * cos_theta
    return np.column_stack([rotated_x, rotated_y])


def _point_distance_squared(phi: float, point_x: float, point_y: float, axis_a: float, axis_b: float) -> float:
    ellipse_x = axis_a * np.cos(phi)
    ellipse_y = axis_b * np.sin(phi)
    return float((ellipse_x - point_x) ** 2 + (ellipse_y - point_y) ** 2)


def _closest_parameter(point_x: float, point_y: float, axis_a: float, axis_b: float) -> float:
    """
    Return the ellipse parameter of the closest point to ``(point_x, point_y)``.

    We use several Newton starts around the scaled polar angle and keep the best
    converged solution. This is fast for the small slice sizes in this project
    while remaining much closer to the physical orthogonal distance than an
    algebraic residual.
    """
    if (abs(point_x) < _EPS) and (abs(point_y) < _EPS):
        return 0.5 * pi

    base = float(np.arctan2(axis_a * point_y, axis_b * point_x))
    best_phi = base
    best_distance = float("inf")

    for seed in _ANGLE_SEEDS:
        phi = base + seed
        for _ in range(_NEWTON_MAX_ITER):
            sin_phi = float(np.sin(phi))
            cos_phi = float(np.cos(phi))
            function_value = (
                (axis_b * axis_b - axis_a * axis_a) * sin_phi * cos_phi
                + axis_a * point_x * sin_phi
                - axis_b * point_y * cos_phi
            )
            derivative = (
                (axis_b * axis_b - axis_a * axis_a) * (cos_phi * cos_phi - sin_phi * sin_phi)
                + axis_a * point_x * cos_phi
                + axis_b * point_y * sin_phi
            )

            if abs(derivative) < _EPS:
                break

            step = float(np.clip(function_value / derivative, -_NEWTON_STEP_CLIP, _NEWTON_STEP_CLIP))
            phi -= step
            if abs(step) < 1e-10:
                break

        distance = _point_distance_squared(phi, point_x, point_y, axis_a, axis_b)
        if distance < best_distance:
            best_distance = distance
            best_phi = phi

    return float(best_phi)


def signed_geometric_distances(
    points_xy: np.ndarray,
    params: tuple[float, float, float, float, float],
) -> np.ndarray:
    """
    Return signed point-to-ellipse distances in Angstrom-like coordinate units.

    The sign is positive outside the ellipse and negative inside.
    """
    center_x, center_y, axis_a, axis_b, theta = params
    rotated = _rotate_points(points_xy, center_x, center_y, theta)

    distances = np.empty(rotated.shape[0], dtype=float)
    for index, (point_x, point_y) in enumerate(rotated):
        phi = _closest_parameter(float(point_x), float(point_y), axis_a, axis_b)
        closest_x = axis_a * np.cos(phi)
        closest_y = axis_b * np.sin(phi)
        distance = float(np.hypot(point_x - closest_x, point_y - closest_y))
        algebraic_sign = (point_x / axis_a) ** 2 + (point_y / axis_b) ** 2 - 1.0
        sign = 1.0 if algebraic_sign >= 0.0 else -1.0
        distances[index] = sign * distance

    return distances


def _geometric_residuals(raw_params: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    return signed_geometric_distances(points_xy, _decode_params(raw_params))


def _initial_guess(points_xy: np.ndarray) -> np.ndarray:
    """Build a robust ellipse initialization from trimmed second moments."""
    pts = np.asarray(points_xy, dtype=float)
    median_center = np.median(pts, axis=0)
    radial_distance = np.sqrt(np.sum((pts - median_center) ** 2, axis=1))

    median_radius = float(np.median(radial_distance))
    mad_radius = float(np.median(np.abs(radial_distance - median_radius)))
    sigma_radius = float(ROBUST_SIGMA_SCALE * mad_radius)

    if sigma_radius > _EPS:
        keep_mask = np.abs(radial_distance - median_radius) <= (3.0 * sigma_radius)
        trimmed = pts[keep_mask]
        if trimmed.shape[0] < 5:
            trimmed = pts
    else:
        trimmed = pts

    center_x, center_y = np.mean(trimmed, axis=0)
    centered = trimmed - np.array([center_x, center_y], dtype=float)

    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_vector = eigenvectors[:, 1]
    theta = float(np.arctan2(major_vector[1], major_vector[0]))

    axis_a = float(np.sqrt(max(2.0 * eigenvalues[1], COVARIANCE_FLOOR)))
    axis_b = float(np.sqrt(max(2.0 * eigenvalues[0], COVARIANCE_FLOOR)))
    axis_a = max(axis_a, np.sqrt(COVARIANCE_FLOOR))
    axis_b = max(axis_b, np.sqrt(COVARIANCE_FLOOR))
    return np.array(
        [center_x, center_y, np.log(axis_a), np.log(axis_b), theta],
        dtype=float,
    )


def _fit_once(
    points_xy: np.ndarray,
    initial_guess: np.ndarray,
    least_squares_config: LeastSquaresConfig,
) -> np.ndarray | None:
    try:
        result = least_squares(
            _geometric_residuals,
            np.asarray(initial_guess, dtype=float),
            args=(np.asarray(points_xy, dtype=float),),
            method=least_squares_config.method,
            loss=least_squares_config.loss,
            f_scale=float(least_squares_config.f_scale),
        )
    except Exception:
        return None

    if not result.success:
        return None
    return np.asarray(result.x, dtype=float)


def _distance_threshold(distances: np.ndarray, least_squares_config: LeastSquaresConfig) -> float:
    absolute_distances = np.abs(np.asarray(distances, dtype=float))
    median_distance = float(np.median(absolute_distances))
    mad_distance = float(np.median(np.abs(absolute_distances - median_distance)))
    sigma = float(ROBUST_SIGMA_SCALE * mad_distance)
    lower = 1.5 * float(least_squares_config.f_scale)
    upper = 2.5 * float(least_squares_config.f_scale)
    robust_band = median_distance + (3.0 * sigma)
    return max(
        lower,
        min(upper, robust_band),
        1e-3,
    )


def fit_rotated_ellipse(
    points_xy: np.ndarray,
    least_squares_config: LeastSquaresConfig,
) -> dict[str, float] | None:
    """
    Fit a rotated ellipse using robust geometric distances.

    The returned ``rmse`` is the RMSE of signed point-to-ellipse distances across
    all retained core points, so it is expressed in the same units as the input
    coordinates. ``inlier_frac`` is kept only as a soft diagnostic derived from
    the final residual distribution; it is not used to refit the ellipse.
    """
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("`points_xy` must have shape `(N, 2)`.")
    if pts.shape[0] < 5:
        return None

    initial_guess = _initial_guess(pts)
    raw_params = _fit_once(pts, initial_guess, least_squares_config)
    if raw_params is None:
        return None

    params = _canonicalize_ellipse(*_decode_params(raw_params))

    all_distances = signed_geometric_distances(pts, params)
    rmse = float(np.sqrt(np.mean(all_distances**2))) if all_distances.size else float("inf")
    rmse_all = rmse
    median_abs_distance = float(np.median(np.abs(all_distances))) if all_distances.size else float("inf")
    threshold = _distance_threshold(all_distances, least_squares_config)
    inlier_mask = np.abs(all_distances) <= threshold
    inlier_count = int(np.sum(inlier_mask))

    center_x, center_y, axis_a, axis_b, theta = params
    return {
        "xc": float(center_x),
        "yc": float(center_y),
        "a": float(axis_a),
        "b": float(axis_b),
        "theta": float(theta),
        "rmse": float(rmse),
        "rmse_all": float(rmse_all),
        "median_abs_dist": float(median_abs_distance),
        "used_n": float(inlier_count),
        "inlier_frac": float(inlier_count / len(pts)),
    }
