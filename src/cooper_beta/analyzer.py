from __future__ import annotations

from copy import deepcopy

import numpy as np

from .analysis_utils import (
    angular_gap_stats,
    collapse_points_by_strand,
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
    "sequence_core_rule_enabled": ("rules", "sequence_core", "enabled"),
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

    def _minimum_points_for_scoring(self) -> int:
        return max(
            self.config.decision.min_intersections_for_scoring,
            self.config.fit.min_points_per_slice,
        )

    def _layer_result(
        self,
        z_value: float,
        point_count: int,
        *,
        fit: dict[str, float] | None,
        valid: bool,
        reason: str,
        raw_point_count: int | None = None,
        collapsed_point_count: int | None = None,
        trim_left: int = 0,
        trim_right: int = 0,
        nn_stats: tuple[float, float, float, float] | None = None,
        angle_stats: tuple[float, float, int, float, float] | None = None,
        order_stats: dict[str, float | int] | None = None,
    ) -> dict[str, object]:
        return {
            "z": float(z_value),
            "n_points": int(point_count),
            "raw_n_points": int(raw_point_count) if raw_point_count is not None else int(point_count),
            "collapsed_n_points": (
                int(collapsed_point_count) if collapsed_point_count is not None else int(point_count)
            ),
            "seq_core_trim_left": int(trim_left),
            "seq_core_trim_right": int(trim_right),
            "fit": fit,
            "valid": bool(valid),
            "reason": reason,
            "fit_rmse_all": fit.get("rmse_all") if fit else None,
            "fit_median_abs_dist": fit.get("median_abs_dist") if fit else None,
            "fit_used_n": fit.get("used_n") if fit else None,
            "fit_inlier_frac": fit.get("inlier_frac") if fit else None,
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

    def _evaluate_slice_points(
        self,
        points: list[tuple[float, ...]] | np.ndarray,
    ) -> dict[str, object]:
        pts = np.asarray(points, dtype=float)
        point_count = int(pts.shape[0]) if pts.ndim == 2 else 0
        minimum_points_for_scoring = self._minimum_points_for_scoring()

        base_result: dict[str, object] = {
            "points": pts,
            "point_count": point_count,
            "fit": None,
            "valid": False,
            "scored": False,
            "reason": f"JUNK(too few intersections: need >= {minimum_points_for_scoring})",
            "nn_stats": None,
            "angle_stats": None,
            "order_stats": None,
        }
        if pts.ndim != 2 or point_count < minimum_points_for_scoring:
            return base_result

        points_xy = np.asarray(pts[:, :2], dtype=float)

        nn_stats, nn_ok = self._evaluate_nn_rule(points_xy)
        angle_stats, order_stats, angle_ok, angle_failure_reason = self._evaluate_angle_rule(
            pts,
            points_xy,
        )

        base_result.update(
            {
                "nn_stats": nn_stats,
                "angle_stats": angle_stats,
                "order_stats": order_stats,
            }
        )

        if (not nn_ok) and self.config.rules.nearest_neighbor.fail_as_junk:
            base_result["reason"] = "JUNK(irregular nearest-neighbor spacing)"
            return base_result

        if (not angle_ok) and self.config.rules.angle.fail_as_junk:
            base_result["reason"] = f"JUNK({angle_failure_reason or 'Angle rule failed'})"
            return base_result

        base_result["scored"] = True

        if not nn_ok:
            base_result["reason"] = "Nearest-neighbor spacing is irregular"
            return base_result

        if not angle_ok:
            base_result["reason"] = angle_failure_reason or "Angle rule failed"
            return base_result

        fit = self.fit_slice(pts)
        base_result["fit"] = fit
        if fit is None:
            base_result["reason"] = "Ellipse fit failed"
            return base_result

        axis_a = fit["a"]
        axis_b = fit["b"]
        rmse = fit["rmse"]
        fit_cfg = self.config.fit

        if rmse > fit_cfg.max_rmse:
            base_result["reason"] = f"Ellipse fit RMSE exceeds {fit_cfg.max_rmse}"
            return base_result
        if axis_a < fit_cfg.min_axis or axis_b < fit_cfg.min_axis:
            base_result["reason"] = "Ellipse axis is below the minimum threshold"
            return base_result
        if axis_a > fit_cfg.max_axis or axis_b > fit_cfg.max_axis:
            base_result["reason"] = "Ellipse axis exceeds the maximum threshold"
            return base_result
        if (axis_a / axis_b) > fit_cfg.max_flattening:
            base_result["reason"] = "Slice is too flattened"
            return base_result

        base_result["valid"] = True
        base_result["reason"] = "OK"
        return base_result

    @staticmethod
    def _candidate_sort_key(
        evaluation: dict[str, object],
        *,
        trim_left: int,
        trim_right: int,
    ) -> tuple[int, float, float, float, float, int, int, int]:
        fit = evaluation.get("fit") or {}
        nn_stats = evaluation.get("nn_stats")
        order_stats = evaluation.get("order_stats") or {}
        return (
            int(trim_left + trim_right),
            float(fit.get("rmse", float("inf"))),
            1.0 - float(nn_stats[3]) if nn_stats else float("inf"),
            float(order_stats.get("order_mean_circ_dist_norm", 0.0)),
            1.0 - float(order_stats.get("order_local_frac", 1.0)),
            -int(evaluation.get("point_count", 0)),
            int(trim_left),
            int(trim_right),
        )

    def _material_core_improvement(
        self,
        full_evaluation: dict[str, object],
        candidate_evaluation: dict[str, object],
    ) -> bool:
        if not candidate_evaluation["valid"]:
            return False
        if not full_evaluation["valid"]:
            return True

        full_fit = full_evaluation.get("fit") or {}
        candidate_fit = candidate_evaluation.get("fit") or {}
        fit_improvement = (
            float(full_fit.get("rmse", float("inf")))
            - float(candidate_fit.get("rmse", float("inf")))
            >= max(1e-3, 0.02 * float(self.config.fit.max_rmse))
        )

        full_nn_stats = full_evaluation.get("nn_stats")
        candidate_nn_stats = candidate_evaluation.get("nn_stats")
        inlier_improvement = (
            candidate_nn_stats is not None
            and full_nn_stats is not None
            and (float(candidate_nn_stats[3]) - float(full_nn_stats[3]) >= 0.05)
        )

        full_order_stats = full_evaluation.get("order_stats") or {}
        candidate_order_stats = candidate_evaluation.get("order_stats") or {}
        order_improvement = (
            float(full_order_stats.get("order_mean_circ_dist_norm", 0.0))
            - float(candidate_order_stats.get("order_mean_circ_dist_norm", 0.0))
            >= 0.05
        ) or (
            float(candidate_order_stats.get("order_local_frac", 1.0))
            - float(full_order_stats.get("order_local_frac", 1.0))
            >= 0.10
        )

        return fit_improvement or inlier_improvement or order_improvement

    def _select_sequence_core(
        self,
        points: list[tuple[float, ...]] | np.ndarray,
    ) -> tuple[np.ndarray, dict[str, int], dict[str, object]]:
        raw_points = np.asarray(points, dtype=float)
        collapsed_points = collapse_points_by_strand(raw_points)
        if collapsed_points.ndim != 2:
            collapsed_points = raw_points

        if collapsed_points.shape[1] >= 4:
            full_points = collapsed_points[:, :3]
            ordered_points = collapsed_points[
                np.argsort(collapsed_points[:, 2], kind="mergesort")
            ]
        else:
            full_points = collapsed_points[:, :3]
            ordered_points = full_points[np.argsort(full_points[:, 2], kind="mergesort")]

        diagnostics = {
            "raw_point_count": int(raw_points.shape[0]) if raw_points.ndim == 2 else 0,
            "collapsed_point_count": int(full_points.shape[0]) if full_points.ndim == 2 else 0,
            "trim_left": 0,
            "trim_right": 0,
        }

        full_evaluation = self._evaluate_slice_points(full_points)
        best_points = full_points
        best_evaluation = full_evaluation
        best_key = None

        if not self.config.rules.sequence_core.enabled:
            return best_points, diagnostics, best_evaluation

        minimum_points_for_scoring = self._minimum_points_for_scoring()
        total_points = int(ordered_points.shape[0]) if ordered_points.ndim == 2 else 0
        if total_points <= minimum_points_for_scoring:
            return best_points, diagnostics, best_evaluation

        for start in range(total_points):
            for stop in range(start + minimum_points_for_scoring, total_points + 1):
                if start == 0 and stop == total_points:
                    continue

                candidate = ordered_points[start:stop]
                candidate_points = candidate[:, :3] if candidate.shape[1] >= 4 else candidate
                evaluation = self._evaluate_slice_points(candidate_points)
                if not self._material_core_improvement(full_evaluation, evaluation):
                    continue

                key = self._candidate_sort_key(
                    evaluation,
                    trim_left=start,
                    trim_right=total_points - stop,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_points = candidate_points
                    best_evaluation = evaluation
                    diagnostics["trim_left"] = start
                    diagnostics["trim_right"] = total_points - stop

        return best_points, diagnostics, best_evaluation

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

        total_scored_layers = 0

        for z_value in active_layers:
            points = slices_dict[z_value]
            selected_points, core_stats, evaluation = self._select_sequence_core(points)
            point_count = int(evaluation["point_count"])

            if evaluation["scored"]:
                total_scored_layers += 1

            fit = evaluation["fit"]
            is_valid = bool(evaluation["valid"])
            reason = str(evaluation["reason"])
            if is_valid and fit is not None:
                valid_layers += 1
                valid_scored_layers += 1
                valid_radii.append((fit["a"] + fit["b"]) / 2.0)

            results.append(
                self._layer_result(
                    z_value,
                    point_count,
                    fit=fit,
                    valid=is_valid,
                    reason=reason,
                    raw_point_count=core_stats["raw_point_count"],
                    collapsed_point_count=core_stats["collapsed_point_count"],
                    trim_left=core_stats["trim_left"],
                    trim_right=core_stats["trim_right"],
                    nn_stats=evaluation["nn_stats"],
                    angle_stats=evaluation["angle_stats"],
                    order_stats=evaluation["order_stats"],
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
