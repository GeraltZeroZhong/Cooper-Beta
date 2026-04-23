from __future__ import annotations

from copy import deepcopy

import numpy as np

from .analysis_utils import (
    angular_gap_stats,
    nearest_neighbor_spacing_stats,
    sequence_angle_order_stats,
)
from .config import AnalyzerConfig
from .ellipse import fit_rotated_ellipse

LEGACY_ANALYZER_OVERRIDE_PATHS = {
    "min_points": ("fit", "min_points_per_slice"),
    "max_rmse": ("fit", "max_rmse"),
    "min_axis": ("fit", "min_axis"),
    "max_axis": ("fit", "max_axis"),
    "max_flattening": ("fit", "max_flattening"),
    "valid_ratio": ("decision", "barrel_valid_ratio"),
    "lsq_method": ("fit", "least_squares", "method"),
    "loss": ("fit", "least_squares", "loss"),
    "f_scale": ("fit", "least_squares", "f_scale"),
    "min_intersections_for_scoring": ("decision", "min_intersections_for_scoring"),
    "nn_rule_enabled": ("rules", "nearest_neighbor", "enabled"),
    "nn_max_robust_cv": ("rules", "nearest_neighbor", "max_robust_cv"),
    "nn_min_inlier_frac": ("rules", "nearest_neighbor", "min_inlier_frac"),
    "nn_fail_as_junk": ("rules", "nearest_neighbor", "fail_as_junk"),
    "angle_rule_enabled": ("rules", "angle", "enabled"),
    "angle_max_gap_deg": ("rules", "angle", "max_gap_deg"),
    "angle_order_rule_enabled": ("rules", "angle", "order", "enabled"),
    "angle_order_local_step_max": ("rules", "angle", "order", "local_step_max"),
    "angle_order_min_local_frac": ("rules", "angle", "order", "min_local_frac"),
    "angle_order_max_mean_circ_dist_norm": (
        "rules",
        "angle",
        "order",
        "max_mean_circ_dist_norm",
    ),
    "angle_fail_as_junk": ("rules", "angle", "fail_as_junk"),
}


class BarrelAnalyzer:
    """
    Analyze slice intersections with ellipse fitting plus geometric consistency rules.
    """

    def __init__(self, config: AnalyzerConfig | None = None, **legacy_overrides):
        self.config = deepcopy(config or AnalyzerConfig())
        self._apply_legacy_overrides(legacy_overrides)

    def _apply_legacy_overrides(self, overrides: dict[str, object]) -> None:
        for override_name, override_value in overrides.items():
            if override_name not in LEGACY_ANALYZER_OVERRIDE_PATHS:
                raise TypeError(f"Unknown analyzer override: {override_name}")

            target = self.config
            path = LEGACY_ANALYZER_OVERRIDE_PATHS[override_name]
            for part in path[:-1]:
                target = getattr(target, part)
            setattr(target, path[-1], override_value)

    def fit_slice(self, points: list[tuple[float, ...]] | np.ndarray) -> dict[str, float] | None:
        """Fit a rotated ellipse to a slice."""
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < self.config.fit.min_points_per_slice:
            return None
        return fit_rotated_ellipse(pts[:, :2], self.config.fit.least_squares)

    def _layer_result(
        self,
        z_value: float,
        point_count: int,
        *,
        fit: dict[str, float] | None,
        valid: bool,
        reason: str,
        nn_stats: tuple[float, float, float, float] | None = None,
        angle_stats: tuple[float, float, int, float, float] | None = None,
        order_stats: dict[str, float | int] | None = None,
    ) -> dict[str, object]:
        return {
            "z": float(z_value),
            "n_points": int(point_count),
            "fit": fit,
            "valid": bool(valid),
            "reason": reason,
            "nn_median": nn_stats[0] if nn_stats else None,
            "nn_sigma": nn_stats[1] if nn_stats else None,
            "nn_cv": nn_stats[2] if nn_stats else None,
            "nn_inlier_frac": nn_stats[3] if nn_stats else None,
            "angle_max_gap_deg": angle_stats[0] if angle_stats else None,
            "angle_coverage_deg": angle_stats[1] if angle_stats else None,
            "angle_used_n": angle_stats[2] if angle_stats else None,
            "angle_center_x": angle_stats[3] if angle_stats else None,
            "angle_center_y": angle_stats[4] if angle_stats else None,
            "order_used_n": order_stats.get("order_used_n") if order_stats else None,
            "order_local_frac": order_stats.get("order_local_frac") if order_stats else None,
            "order_mean_step": order_stats.get("order_mean_step") if order_stats else None,
            "order_max_step": order_stats.get("order_max_step") if order_stats else None,
            "order_mean_circ_dist_norm": (
                order_stats.get("order_mean_circ_dist_norm") if order_stats else None
            ),
            "seq_neighbor_dist_median": (
                order_stats.get("seq_neighbor_dist_median") if order_stats else None
            ),
            "seq_neighbor_dist_robust_cv": (
                order_stats.get("seq_neighbor_dist_robust_cv") if order_stats else None
            ),
        }

    def _evaluate_nn_rule(
        self,
        points_xy: np.ndarray,
    ) -> tuple[tuple[float, float, float, float] | None, bool]:
        nn_cfg = self.config.rules.nearest_neighbor
        if not nn_cfg.enabled:
            return None, True

        stats = nearest_neighbor_spacing_stats(points_xy)
        if stats is None:
            return None, True

        _, _, robust_cv, inlier_fraction = stats
        is_valid = (robust_cv <= nn_cfg.max_robust_cv) and (
            inlier_fraction >= nn_cfg.min_inlier_frac
        )
        return stats, is_valid

    def _evaluate_angle_rule(
        self,
        points: list[tuple[float, ...]] | np.ndarray,
        points_xy: np.ndarray,
    ) -> tuple[
        tuple[float, float, int, float, float] | None,
        dict[str, float | int] | None,
        bool,
        str | None,
    ]:
        angle_cfg = self.config.rules.angle
        if not angle_cfg.enabled:
            return None, None, True, None

        angle_stats = angular_gap_stats(points_xy)
        if angle_stats is not None:
            max_gap_deg = angle_stats[0]
            gap_ok = max_gap_deg <= angle_cfg.max_gap_deg
        else:
            gap_ok = True

        order_stats = None
        order_ok = True
        failure_reason = None
        if angle_cfg.order.enabled:
            order_stats = sequence_angle_order_stats(np.asarray(points, dtype=float), angle_cfg.order)
            if order_stats is not None:
                local_ok = order_stats["order_local_frac"] >= angle_cfg.order.min_local_frac
                global_ok = (
                    order_stats["order_mean_circ_dist_norm"]
                    <= angle_cfg.order.max_mean_circ_dist_norm
                )
                order_ok = local_ok and global_ok
                if not local_ok:
                    failure_reason = "Local sequence/angle order mismatch"
                elif not global_ok:
                    failure_reason = "Global sequence/angle order mismatch"

        if not gap_ok:
            failure_reason = "Angular gap is too large"

        return angle_stats, order_stats, (gap_ok and order_ok), failure_reason

    def analyze(self, slices_dict: dict[float, list[tuple[float, ...]]]) -> dict[str, object]:
        """
        Analyze active slices and return per-layer diagnostics plus aggregate scores.
        """
        results: list[dict[str, object]] = []
        valid_layers = 0
        valid_scored_layers = 0
        valid_radii: list[float] = []

        active_layers = [z_value for z_value in sorted(slices_dict) if slices_dict[z_value]]
        total_layers = len(active_layers)
        if total_layers == 0:
            return {
                "is_barrel": False,
                "score": 0.0,
                "score_adjust": 0.0,
                "msg": "No active slices were found.",
            }

        minimum_points_for_scoring = max(
            self.config.decision.min_intersections_for_scoring,
            self.config.fit.min_points_per_slice,
        )
        total_scored_layers = 0

        for z_value in active_layers:
            points = slices_dict[z_value]
            point_count = len(points)

            if point_count < minimum_points_for_scoring:
                results.append(
                    self._layer_result(
                        z_value,
                        point_count,
                        fit=None,
                        valid=False,
                        reason=(
                            f"JUNK(too few intersections: need >= {minimum_points_for_scoring})"
                        ),
                    )
                )
                continue

            points_xy = np.asarray([(point[0], point[1]) for point in points], dtype=float)

            nn_stats, nn_ok = self._evaluate_nn_rule(points_xy)
            if (not nn_ok) and self.config.rules.nearest_neighbor.fail_as_junk:
                results.append(
                    self._layer_result(
                        z_value,
                        point_count,
                        fit=None,
                        valid=False,
                        reason="JUNK(irregular nearest-neighbor spacing)",
                        nn_stats=nn_stats,
                    )
                )
                continue

            angle_stats, order_stats, angle_ok, angle_failure_reason = self._evaluate_angle_rule(
                points,
                points_xy,
            )
            if (not angle_ok) and self.config.rules.angle.fail_as_junk:
                results.append(
                    self._layer_result(
                        z_value,
                        point_count,
                        fit=None,
                        valid=False,
                        reason=f"JUNK({angle_failure_reason or 'Angle rule failed'})",
                        nn_stats=nn_stats,
                        angle_stats=angle_stats,
                        order_stats=order_stats,
                    )
                )
                continue

            total_scored_layers += 1

            if (not nn_ok) and (not self.config.rules.nearest_neighbor.fail_as_junk):
                results.append(
                    self._layer_result(
                        z_value,
                        point_count,
                        fit=None,
                        valid=False,
                        reason="Nearest-neighbor spacing is irregular",
                        nn_stats=nn_stats,
                        angle_stats=angle_stats,
                        order_stats=order_stats,
                    )
                )
                continue

            if (not angle_ok) and (not self.config.rules.angle.fail_as_junk):
                results.append(
                    self._layer_result(
                        z_value,
                        point_count,
                        fit=None,
                        valid=False,
                        reason=angle_failure_reason or "Angle rule failed",
                        nn_stats=nn_stats,
                        angle_stats=angle_stats,
                        order_stats=order_stats,
                    )
                )
                continue

            fit = self.fit_slice(points)
            if fit is None:
                reason = "Ellipse fit failed"
                is_valid = False
            else:
                axis_a = fit["a"]
                axis_b = fit["b"]
                rmse = fit["rmse"]
                fit_cfg = self.config.fit

                if rmse > fit_cfg.max_rmse:
                    reason = f"Ellipse fit RMSE exceeds {fit_cfg.max_rmse}"
                    is_valid = False
                elif axis_a < fit_cfg.min_axis or axis_b < fit_cfg.min_axis:
                    reason = "Ellipse axis is below the minimum threshold"
                    is_valid = False
                elif axis_a > fit_cfg.max_axis or axis_b > fit_cfg.max_axis:
                    reason = "Ellipse axis exceeds the maximum threshold"
                    is_valid = False
                elif (axis_a / axis_b) > fit_cfg.max_flattening:
                    reason = "Slice is too flattened"
                    is_valid = False
                else:
                    reason = "OK"
                    is_valid = True
                    valid_radii.append((axis_a + axis_b) / 2.0)

            if is_valid:
                valid_layers += 1
                valid_scored_layers += 1

            results.append(
                self._layer_result(
                    z_value,
                    point_count,
                    fit=fit,
                    valid=is_valid,
                    reason=reason,
                    nn_stats=nn_stats,
                    angle_stats=angle_stats,
                    order_stats=order_stats,
                )
            )

        score = (valid_layers / total_layers) if total_layers else 0.0
        score_adjust = (valid_scored_layers / total_scored_layers) if total_scored_layers else 0.0
        average_radius = float(np.mean(valid_radii)) if valid_radii else 0.0

        return {
            "is_barrel": None,
            "score": float(score),
            "score_adjust": float(score_adjust),
            "valid_layers": int(valid_scored_layers),
            "total_layers": int(total_layers),
            "total_scored_layers": int(total_scored_layers),
            "avg_radius": float(average_radius),
            "layer_details": results,
        }
