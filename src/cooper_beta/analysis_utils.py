from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .config import AngleOrderRuleConfig
from .constants import (
    EPSILON,
    FULL_ROTATION_DEG,
    MIN_ANGULAR_GAP_POINTS,
    MIN_NEAREST_NEIGHBOR_POINTS,
    MIN_SEQUENCE_ANGLE_ORDER_POINTS,
    RAD_TO_DEG,
    ROBUST_SIGMA_SCALE,
    THREE_SIGMA_MULTIPLIER,
    TOLERANCE,
)


def robust_center(points_xy: np.ndarray) -> tuple[float, float]:
    """Return the mean center of a slice cross section."""
    pts = np.asarray(points_xy, dtype=float)
    center = np.mean(pts, axis=0)
    return float(center[0]), float(center[1])


def nearest_neighbor_spacing_stats(
    points_xy: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Return robust nearest-neighbor spacing statistics for one slice."""
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < MIN_NEAREST_NEIGHBOR_POINTS:
        return None

    tree = cKDTree(pts)
    distances, _ = tree.query(pts, k=2)
    nearest_neighbor = distances[:, 1]

    median_distance = float(np.median(nearest_neighbor))
    mad_distance = float(np.median(np.abs(nearest_neighbor - median_distance)))
    robust_sigma = float(ROBUST_SIGMA_SCALE * mad_distance)

    if median_distance <= TOLERANCE:
        robust_cv = float("inf")
    else:
        robust_cv = float(robust_sigma / median_distance)

    if robust_sigma < EPSILON:
        return median_distance, 0.0, 0.0, 1.0

    inliers = np.abs(nearest_neighbor - median_distance) <= (
        THREE_SIGMA_MULTIPLIER * robust_sigma
    )
    inlier_fraction = float(np.mean(inliers)) if len(inliers) else 0.0
    return median_distance, robust_sigma, robust_cv, inlier_fraction


def angular_gap_stats(
    points_xy: np.ndarray,
) -> tuple[float, float, int, float, float] | None:
    """Return max angular gap, coverage, used count, and center coordinates."""
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < MIN_ANGULAR_GAP_POINTS:
        return None

    center_x, center_y = robust_center(pts)
    delta_x = pts[:, 0] - center_x
    delta_y = pts[:, 1] - center_y
    radius = np.sqrt(delta_x * delta_x + delta_y * delta_y)

    median_radius = float(np.median(radius))
    mad_radius = float(np.median(np.abs(radius - median_radius)))
    radius_sigma = float(ROBUST_SIGMA_SCALE * mad_radius)

    if radius_sigma > TOLERANCE:
        keep_mask = np.abs(radius - median_radius) <= (THREE_SIGMA_MULTIPLIER * radius_sigma)
        filtered_points = pts[keep_mask]
    else:
        filtered_points = pts

    if filtered_points.shape[0] < MIN_ANGULAR_GAP_POINTS:
        return None

    filtered_dx = filtered_points[:, 0] - center_x
    filtered_dy = filtered_points[:, 1] - center_y
    angle = np.sort(np.arctan2(filtered_dy, filtered_dx))

    differences = np.diff(angle)
    wrap_difference = (angle[0] + (2.0 * np.pi)) - angle[-1]
    circular_differences = np.concatenate([differences, [wrap_difference]])

    max_gap_radians = float(np.max(circular_differences))
    max_gap_degrees = max_gap_radians * RAD_TO_DEG
    coverage_degrees = FULL_ROTATION_DEG - max_gap_degrees

    return (
        max_gap_degrees,
        coverage_degrees,
        int(filtered_points.shape[0]),
        center_x,
        center_y,
    )


def best_circular_affine_fit_cost(angle_pos_by_seq: np.ndarray) -> float:
    """Return normalized sequence/angle-order mismatch under shift and reversal."""
    positions = np.asarray(angle_pos_by_seq, dtype=int)
    count = int(positions.size)
    if count <= 1:
        return 0.0

    normalization = max(1.0, count / 2.0)
    best_cost = float("inf")

    base_order = np.arange(count, dtype=int)
    for direction in (1, -1):
        directed_order = (direction * base_order) % count
        for shift in range(count):
            predicted = (directed_order + shift) % count
            distance = np.abs(positions - predicted)
            distance = np.minimum(distance, count - distance)
            candidate_cost = float(np.mean(distance)) / normalization
            if candidate_cost < best_cost:
                best_cost = candidate_cost

    return float(min(1.0, max(0.0, best_cost)))


def sequence_angle_order_stats(
    points: np.ndarray,
    order_config: AngleOrderRuleConfig,
) -> dict[str, float | int] | None:
    """Compute sequence-order versus angle-order consistency on one slice."""
    pts = np.asarray(points, dtype=float)
    if (
        pts.ndim != 2
        or pts.shape[0] < MIN_SEQUENCE_ANGLE_ORDER_POINTS
        or pts.shape[1] < 3
    ):
        return None

    xy = pts[:, :2]
    sequence_position = pts[:, 2]

    center_x, center_y = robust_center(xy)
    delta_x = xy[:, 0] - center_x
    delta_y = xy[:, 1] - center_y
    radius = np.sqrt(delta_x * delta_x + delta_y * delta_y)

    median_radius = float(np.median(radius))
    mad_radius = float(np.median(np.abs(radius - median_radius)))
    radius_sigma = float(ROBUST_SIGMA_SCALE * mad_radius)
    if radius_sigma > TOLERANCE:
        keep_mask = np.abs(radius - median_radius) <= (THREE_SIGMA_MULTIPLIER * radius_sigma)
    else:
        keep_mask = np.ones_like(radius, dtype=bool)

    if int(np.sum(keep_mask)) < MIN_SEQUENCE_ANGLE_ORDER_POINTS:
        return None

    filtered_xy = xy[keep_mask]
    filtered_sequence_position = sequence_position[keep_mask]
    count = int(filtered_xy.shape[0])

    sequence_order = np.argsort(filtered_sequence_position)
    filtered_dx = filtered_xy[:, 0] - center_x
    filtered_dy = filtered_xy[:, 1] - center_y
    angle = (np.arctan2(filtered_dy, filtered_dx) * RAD_TO_DEG) % FULL_ROTATION_DEG
    angle_order = np.argsort(angle)

    position_in_angle_order = np.empty(count, dtype=int)
    position_in_angle_order[angle_order] = np.arange(count, dtype=int)
    angle_position_by_sequence = position_in_angle_order[sequence_order]

    local_steps = []
    for index in range(count - 1):
        step = abs(
            int(angle_position_by_sequence[index + 1]) - int(angle_position_by_sequence[index])
        )
        local_steps.append(min(step, count - step))
    local_steps_array = np.asarray(local_steps, dtype=float)

    local_fraction = (
        float(np.mean(local_steps_array <= float(order_config.local_step_max)))
        if local_steps_array.size
        else 1.0
    )
    mean_step = float(np.mean(local_steps_array)) if local_steps_array.size else 0.0
    max_step = float(np.max(local_steps_array)) if local_steps_array.size else 0.0

    mean_circular_distance = float(best_circular_affine_fit_cost(angle_position_by_sequence))

    sequence_order_xy = filtered_xy[sequence_order]
    euclidean_steps = np.diff(sequence_order_xy, axis=0)
    sequence_neighbor_distance = np.sqrt(np.sum(euclidean_steps * euclidean_steps, axis=1))
    if sequence_neighbor_distance.size:
        median_neighbor_distance = float(np.median(sequence_neighbor_distance))
        mad_neighbor_distance = float(
            np.median(np.abs(sequence_neighbor_distance - median_neighbor_distance))
        )
        robust_sigma = float(ROBUST_SIGMA_SCALE * mad_neighbor_distance)
        robust_cv = (
            float(robust_sigma / median_neighbor_distance)
            if median_neighbor_distance > TOLERANCE
            else float("inf")
        )
    else:
        median_neighbor_distance = 0.0
        robust_cv = 0.0

    return {
        "order_used_n": int(count),
        "order_local_frac": float(local_fraction),
        "order_mean_step": float(mean_step),
        "order_max_step": float(max_step),
        "order_mean_circ_dist_norm": float(mean_circular_distance),
        "seq_neighbor_dist_median": float(median_neighbor_distance),
        "seq_neighbor_dist_robust_cv": float(robust_cv),
    }
