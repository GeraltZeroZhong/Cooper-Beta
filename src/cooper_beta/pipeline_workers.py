from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .alignment import PCAAligner
from .analysis_utils import (
    angular_gap_stats,
    collapse_points_by_strand,
    nearest_neighbor_spacing_stats,
    sequence_angle_order_stats,
)
from .analyzer import BarrelAnalyzer
from .config import AppConfig
from .constants import (
    RESULT_BARREL,
    RESULT_ERROR,
    RESULT_FILTERED_OUT,
    RESULT_NON_BARREL,
)
from .loader import ProteinLoader
from .prepare_cache import load_prepare_payloads, store_prepare_payloads
from .slicer import ProteinSlicer

try:
    from tqdm.auto import tqdm

    tqdm_write = tqdm.write
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

    def tqdm_write(message: str) -> None:
        print(message)


def _format_below_threshold_reason(metric_name: str, observed: int, threshold: int) -> str:
    return f"{metric_name} below threshold ({observed} < {threshold})"


def _axis_search_minimum_points(cfg: AppConfig) -> int:
    return max(
        int(cfg.analyzer.decision.min_intersections_for_scoring),
        int(cfg.analyzer.fit.min_points_per_slice),
    )


def _axis_search_score_key(
    slices: dict[float, list[tuple[float, ...]]],
    minimum_points: int,
    cfg: AppConfig,
) -> tuple[int | float, ...]:
    """Rank candidate axis rotations using lightweight geometric rule proxies."""
    rules_cfg = cfg.analyzer.rules
    collapsed_counts: list[int] = []
    full_pass_layers = 0
    gap_pass_layers = 0
    order_pass_layers = 0
    nn_pass_layers = 0
    gap_margins: list[float] = []
    order_local_fracs: list[float] = []
    order_mean_circ_dists: list[float] = []
    nn_inlier_fracs: list[float] = []
    duplicate_penalty = 0

    for points in slices.values():
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue

        collapsed = collapse_points_by_strand(pts)
        collapsed_count = int(collapsed.shape[0]) if collapsed.ndim == 2 else 0
        collapsed_counts.append(collapsed_count)

        if pts.shape[1] >= 4:
            unique_strands = int(np.unique(pts[:, 3].astype(int)).size)
        else:
            unique_strands = int(pts.shape[0])
        duplicate_penalty += int(pts.shape[0]) - unique_strands

        if collapsed_count < minimum_points:
            continue

        collapsed_points = np.asarray(collapsed, dtype=float)
        collapsed_xy = np.asarray(collapsed_points[:, :2], dtype=float)

        nn_ok = True
        nn_inlier_frac = 1.0
        if rules_cfg.nearest_neighbor.enabled:
            nn_stats = nearest_neighbor_spacing_stats(collapsed_xy)
            if nn_stats is not None:
                _, _, robust_cv, inlier_frac = nn_stats
                nn_inlier_frac = float(inlier_frac)
                nn_ok = (
                    float(robust_cv) <= float(rules_cfg.nearest_neighbor.max_robust_cv)
                    and nn_inlier_frac >= float(rules_cfg.nearest_neighbor.min_inlier_frac)
                )
        nn_inlier_fracs.append(nn_inlier_frac)
        if nn_ok:
            nn_pass_layers += 1

        gap_ok = True
        gap_margin = 0.0
        order_ok = True
        order_local_frac = 1.0
        order_mean_circ = 0.0
        if rules_cfg.angle.enabled:
            angle_stats = angular_gap_stats(collapsed_xy)
            if angle_stats is not None:
                max_gap_deg = float(angle_stats[0])
                gap_margin = float(rules_cfg.angle.max_gap_deg) - max_gap_deg
                gap_ok = max_gap_deg <= float(rules_cfg.angle.max_gap_deg)

            if rules_cfg.angle.order.enabled:
                order_stats = sequence_angle_order_stats(collapsed_points, rules_cfg.angle.order)
                if order_stats is not None:
                    order_local_frac = float(order_stats["order_local_frac"])
                    order_mean_circ = float(order_stats["order_mean_circ_dist_norm"])
                    order_ok = (
                        order_local_frac >= float(rules_cfg.angle.order.min_local_frac)
                        and order_mean_circ
                        <= float(rules_cfg.angle.order.max_mean_circ_dist_norm)
                    )

        gap_margins.append(gap_margin)
        order_local_fracs.append(order_local_frac)
        order_mean_circ_dists.append(order_mean_circ)

        if gap_ok:
            gap_pass_layers += 1
        if order_ok:
            order_pass_layers += 1
        if nn_ok and gap_ok and order_ok:
            full_pass_layers += 1

    if not collapsed_counts:
        return (0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0)

    collapsed_array = np.asarray(collapsed_counts, dtype=float)
    layers_at_threshold = int(np.sum(collapsed_array >= float(minimum_points)))
    median_gap_margin = float(np.median(gap_margins)) if gap_margins else 0.0
    median_order_local = float(np.median(order_local_fracs)) if order_local_fracs else 0.0
    median_order_mean_circ = (
        float(np.median(order_mean_circ_dists)) if order_mean_circ_dists else 0.0
    )
    median_nn_inlier = float(np.median(nn_inlier_fracs)) if nn_inlier_fracs else 0.0
    return (
        full_pass_layers,
        gap_pass_layers,
        order_pass_layers,
        nn_pass_layers,
        median_gap_margin,
        median_order_local,
        -median_order_mean_circ,
        median_nn_inlier,
        layers_at_threshold,
        float(np.median(collapsed_array)),
        float(np.mean(collapsed_array)),
        -int(duplicate_penalty),
    )


def _proper_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation_matrix, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("Expected a 3x3 rotation matrix.")
    if float(np.linalg.det(rotation)) < 0.0:
        rotation = rotation.copy()
        rotation[:, 0] *= -1.0
    return rotation


def _candidate_axis_rotations(rotation_matrix: np.ndarray) -> list[np.ndarray]:
    base_rotation = np.asarray(rotation_matrix, dtype=float)
    permutations = (
        (0, 1, 2),  # largest PCA axis -> Z
        (0, 2, 1),  # middle PCA axis -> Z
        (1, 2, 0),  # smallest PCA axis -> Z
    )
    return [
        _proper_rotation_matrix(base_rotation[:, permutation])
        for permutation in permutations
    ]


def _local_axis_rotation(angle_deg: float, axis: str) -> np.ndarray:
    theta = math.radians(float(angle_deg))
    cosine = math.cos(theta)
    sine = math.sin(theta)

    if axis == "x":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
            dtype=float,
        )
    if axis == "y":
        return np.array(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=float,
        )
    raise ValueError(f"Unknown local rotation axis: {axis}")


def _refinement_rotations(base_rotation: np.ndarray, angle_deg: float) -> list[np.ndarray]:
    if angle_deg <= 0.0:
        return [_proper_rotation_matrix(base_rotation)]

    refined = [_proper_rotation_matrix(base_rotation)]
    for axis in ("x", "y"):
        for sign in (-1.0, 1.0):
            refined.append(
                _proper_rotation_matrix(
                    np.asarray(base_rotation, dtype=float)
                    @ _local_axis_rotation(sign * angle_deg, axis)
                )
            )
    return refined


def _aligned_coordinates_for_rotation(
    coordinates: np.ndarray,
    *,
    center: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    return np.dot(np.asarray(coordinates, dtype=float) - np.asarray(center, dtype=float), rotation_matrix)


def _select_alignment_slices(
    aligner: PCAAligner,
    coordinates: np.ndarray,
    residues_data: list[dict[str, object]],
    slicer: ProteinSlicer,
    cfg: AppConfig,
) -> dict[float, list[tuple[float, ...]]]:
    axis_cfg = cfg.analyzer.axis_search
    if not axis_cfg.enabled:
        aligned_coordinates = aligner.transform(coordinates)
        return slicer.slice_structure(aligned_coordinates, residues_data)

    center = getattr(aligner, "center", None)
    rotation_matrix = getattr(aligner, "rotation_matrix", None)
    if center is None or rotation_matrix is None:
        raise RuntimeError("Aligner is not fitted. Call fit() before selecting a PCA axis.")

    minimum_points = _axis_search_minimum_points(cfg)

    def best_slices_for_rotations(
        rotations: Iterable[np.ndarray],
    ) -> tuple[np.ndarray, dict[float, list[tuple[float, ...]]], tuple[int | float, ...]]:
        best_rotation: np.ndarray | None = None
        best_slice_dict: dict[float, list[tuple[float, ...]]] | None = None
        best_key: tuple[int | float, ...] | None = None

        for candidate_rotation in rotations:
            aligned_coordinates = _aligned_coordinates_for_rotation(
                coordinates,
                center=np.asarray(center, dtype=float),
                rotation_matrix=np.asarray(candidate_rotation, dtype=float),
            )
            slice_dict = slicer.slice_structure(aligned_coordinates, residues_data)
            score_key = _axis_search_score_key(
                slice_dict,
                minimum_points,
                cfg,
            )
            if best_key is None or score_key > best_key:
                best_rotation = np.asarray(candidate_rotation, dtype=float)
                best_slice_dict = slice_dict
                best_key = score_key

        if best_rotation is None or best_slice_dict is None or best_key is None:
            raise RuntimeError("Axis search produced no candidate rotations.")
        return best_rotation, best_slice_dict, best_key

    best_rotation, best_slices, _ = best_slices_for_rotations(
        _candidate_axis_rotations(np.asarray(rotation_matrix, dtype=float))
    )

    if axis_cfg.refine.enabled and float(axis_cfg.refine.angle_deg) > 0.0:
        _, best_slices, _ = best_slices_for_rotations(
            _refinement_rotations(best_rotation, float(axis_cfg.refine.angle_deg))
        )

    return best_slices


def analyze_chain_payload(payload: dict[str, object], cfg: AppConfig) -> dict[str, object]:
    """
    Analyze one chain payload and return the final row written to the results table.
    """
    filename = str(payload.get("filename", ""))
    chain_id = str(payload.get("chain", ""))
    residues_data = list(payload.get("residues_data", []) or [])
    min_chain_residues = cfg.input.min_chain_residues
    min_sheet_residues = cfg.input.min_sheet_residues
    min_informative_slices = cfg.input.min_informative_slices
    decision_cfg = cfg.analyzer.decision
    default_decision_basis = "adjusted" if decision_cfg.use_adjusted_score else "raw"
    chain_residue_count = len(residues_data)
    sheet_residue_coords = [
        residue["coord"] for residue in residues_data if residue.get("is_sheet", False)
    ]
    sheet_residue_count = len(sheet_residue_coords)

    def build_row(
        result: str,
        reason: str,
        report: dict[str, object] | None = None,
        *,
        result_stage: str,
        informative_slices: int | None = None,
    ) -> dict[str, object]:
        report = report or {}
        score_raw = float(report.get("score", 0.0))
        score_adjust = float(report.get("score_adjust", 0.0))
        decision_basis = str(report.get("decision_basis", default_decision_basis))
        decision_score = float(
            report.get(
                "decision_score",
                score_adjust if decision_basis == "adjusted" else score_raw,
            )
        )
        total_layers = int(report.get("total_layers", informative_slices or 0))
        scored_layers = int(report.get("total_scored_layers", 0))
        valid_layers = int(report.get("valid_layers", 0))
        valid_layer_frac = float(
            report.get(
                "valid_layer_frac",
                (valid_layers / total_layers) if total_layers else 0.0,
            )
        )
        scored_layer_frac = float(
            report.get(
                "scored_layer_frac",
                (scored_layers / total_layers) if total_layers else 0.0,
            )
        )
        informative_slice_count = int(
            report.get("informative_slices", informative_slices if informative_slices is not None else total_layers)
        )
        return {
            "filename": filename,
            "chain": chain_id,
            "result": result,
            "result_stage": result_stage,
            "decision_score": decision_score,
            "decision_basis": decision_basis,
            "decision_threshold": float(
                report.get("decision_threshold", decision_cfg.barrel_valid_ratio)
            ),
            "score_raw": score_raw,
            "score_adjust": score_adjust,
            "valid_layers": valid_layers,
            "scored_layers": scored_layers,
            "total_layers": total_layers,
            "valid_layer_frac": valid_layer_frac,
            "scored_layer_frac": scored_layer_frac,
            "junk_layers": int(report.get("junk_layers", 0)),
            "invalid_layers": int(report.get("invalid_layers", 0)),
            "avg_radius": float(report.get("avg_radius", 0.0)),
            "chain_residues": int(report.get("chain_residues", chain_residue_count)),
            "sheet_residues": int(report.get("sheet_residues", sheet_residue_count)),
            "informative_slices": informative_slice_count,
            "reason": reason,
            # Keep the legacy names for downstream notebooks and older CSV consumers.
            "all_adjusted_layers": scored_layers,
            "all_layers": total_layers,
        }

    if chain_residue_count < min_chain_residues:
        return build_row(
            RESULT_FILTERED_OUT,
            _format_below_threshold_reason(
                "Chain residues",
                chain_residue_count,
                min_chain_residues,
            ),
            result_stage="prefilter",
        )

    all_coordinates = np.array([residue["coord"] for residue in residues_data], dtype=float)
    if sheet_residue_count < min_sheet_residues:
        return build_row(
            RESULT_FILTERED_OUT,
            _format_below_threshold_reason(
                "Beta-sheet residues",
                sheet_residue_count,
                min_sheet_residues,
            ),
            result_stage="prefilter",
        )
    sheet_coordinates = np.array(sheet_residue_coords, dtype=float)
    slicer = ProteinSlicer(
        step_size=cfg.slicer.step_size,
        fill_sheet_hole_length=cfg.slicer.fill_sheet_hole_length,
    )

    try:
        aligner = PCAAligner()
        aligner.fit(sheet_coordinates)
        slices = _select_alignment_slices(
            aligner,
            all_coordinates,
            residues_data,
            slicer,
            cfg,
        )
    except Exception:
        return build_row(RESULT_ERROR, "Alignment failed", result_stage="error")

    informative_slice_count = len(slices)
    if informative_slice_count < min_informative_slices:
        return build_row(
            RESULT_FILTERED_OUT,
            _format_below_threshold_reason(
                "Informative slices",
                informative_slice_count,
                min_informative_slices,
            ),
            result_stage="prefilter",
            informative_slices=informative_slice_count,
        )

    analyzer = BarrelAnalyzer(cfg.analyzer)
    try:
        report = analyzer.analyze(slices)
    except Exception:
        fallback = {"total_layers": informative_slice_count}
        return build_row(
            RESULT_ERROR,
            "Slice analyzer crashed unexpectedly",
            fallback,
            result_stage="error",
            informative_slices=informative_slice_count,
        )

    report = dict(report)
    score_raw = float(report.get("score", 0.0))
    score_adjust = float(report.get("score_adjust", 0.0))
    final_score = score_adjust if decision_cfg.use_adjusted_score else score_raw

    total_layers = int(report.get("total_layers", 0))
    total_scored_layers = int(report.get("total_scored_layers", 0))
    valid_layers = int(report.get("valid_layers", 0))
    avg_radius = float(report.get("avg_radius", 0.0))
    layer_details = list(report.get("layer_details", []) or [])
    scored_layer_frac = (total_scored_layers / total_layers) if total_layers else 0.0
    exceptions_enabled = bool(decision_cfg.exception_layer_enabled)
    enough_scored_layers = True
    if decision_cfg.use_adjusted_score:
        enough_scored_layers = (total_layers > 0) and (
            total_scored_layers > total_layers * float(decision_cfg.min_scored_layer_frac)
            and total_scored_layers >= int(decision_cfg.min_scored_layers)
        )

    rescue_cfg = decision_cfg.small_barrel_rescue
    rescued_small_barrel = (
        decision_cfg.use_adjusted_score
        and exceptions_enabled
        and (not enough_scored_layers)
        and bool(rescue_cfg.enabled)
        and (
            (
                final_score >= float(rescue_cfg.min_score)
                and total_scored_layers >= int(rescue_cfg.min_scored_layers)
                and total_layers >= int(rescue_cfg.min_total_layers)
                and 0.0 < avg_radius <= float(rescue_cfg.max_avg_radius)
            )
            or (
                bool(rescue_cfg.compact_enabled)
                and final_score >= float(rescue_cfg.compact_min_score)
                and total_scored_layers >= int(rescue_cfg.compact_min_scored_layers)
                and total_layers >= int(rescue_cfg.compact_min_total_layers)
                and total_layers <= int(rescue_cfg.compact_max_total_layers)
                and chain_residue_count >= int(rescue_cfg.compact_min_chain_residues)
                and sheet_residue_count >= int(rescue_cfg.compact_min_sheet_residues)
                and 0.0 < avg_radius <= float(rescue_cfg.compact_max_avg_radius)
            )
            or (
                bool(rescue_cfg.sparse_enabled)
                and final_score >= float(rescue_cfg.sparse_min_score)
                and total_scored_layers >= int(rescue_cfg.sparse_min_scored_layers)
                and total_layers >= int(rescue_cfg.sparse_min_total_layers)
                and chain_residue_count >= int(rescue_cfg.sparse_min_chain_residues)
                and chain_residue_count <= int(rescue_cfg.sparse_max_chain_residues)
                and sheet_residue_count >= int(rescue_cfg.sparse_min_sheet_residues)
                and 0.0 < avg_radius <= float(rescue_cfg.sparse_max_avg_radius)
            )
        )
    )

    near_miss_cfg = decision_cfg.near_miss_rescue
    soft_nn_layers = 0
    if exceptions_enabled and near_miss_cfg.enabled and near_miss_cfg.soft_nn_enabled:
        angle_cfg = cfg.analyzer.rules.angle
        for layer in layer_details:
            if layer.get("reason") != "JUNK(irregular nearest-neighbor spacing)":
                continue
            nn_inlier_frac = layer.get("nn_inlier_frac")
            nn_cv = layer.get("nn_cv")
            angle_gap = layer.get("angle_max_gap_deg")
            order_local = layer.get("order_local_frac")
            order_mean_circ = layer.get("order_mean_circ_dist_norm")
            if (
                nn_inlier_frac is not None
                and nn_cv is not None
                and angle_gap is not None
                and order_local is not None
                and order_mean_circ is not None
                and float(nn_inlier_frac) >= float(near_miss_cfg.soft_nn_min_inlier_frac)
                and float(nn_cv) <= float(near_miss_cfg.soft_nn_max_robust_cv)
                and float(angle_gap) <= float(angle_cfg.max_gap_deg)
                and float(order_local) >= float(angle_cfg.order.min_local_frac)
                and float(order_mean_circ)
                <= float(angle_cfg.order.max_mean_circ_dist_norm)
            ):
                soft_nn_layers += 1

    rescued_near_miss = (
        decision_cfg.use_adjusted_score
        and exceptions_enabled
        and bool(near_miss_cfg.enabled)
        and (
            (
                bool(near_miss_cfg.soft_nn_enabled)
                and total_scored_layers == 0
                and soft_nn_layers >= int(near_miss_cfg.soft_nn_min_layers)
                and total_layers >= int(near_miss_cfg.soft_nn_min_total_layers)
                and total_layers <= int(near_miss_cfg.soft_nn_max_total_layers)
                and chain_residue_count >= int(near_miss_cfg.soft_nn_min_chain_residues)
                and chain_residue_count <= int(near_miss_cfg.soft_nn_max_chain_residues)
                and sheet_residue_count >= int(near_miss_cfg.soft_nn_min_sheet_residues)
                and sheet_residue_count <= int(near_miss_cfg.soft_nn_max_sheet_residues)
            )
            or (
                bool(near_miss_cfg.compact_partner_enabled)
                and final_score >= float(near_miss_cfg.compact_partner_min_score)
                and valid_layers >= int(near_miss_cfg.compact_partner_min_valid_layers)
                and total_scored_layers
                >= int(near_miss_cfg.compact_partner_min_scored_layers)
                and total_layers >= int(near_miss_cfg.compact_partner_min_total_layers)
                and total_layers <= int(near_miss_cfg.compact_partner_max_total_layers)
                and chain_residue_count
                >= int(near_miss_cfg.compact_partner_min_chain_residues)
                and chain_residue_count
                <= int(near_miss_cfg.compact_partner_max_chain_residues)
                and sheet_residue_count
                >= int(near_miss_cfg.compact_partner_min_sheet_residues)
                and sheet_residue_count
                <= int(near_miss_cfg.compact_partner_max_sheet_residues)
                and avg_radius >= float(near_miss_cfg.compact_partner_min_avg_radius)
                and avg_radius <= float(near_miss_cfg.compact_partner_max_avg_radius)
            )
            or (
                bool(near_miss_cfg.large_partner_enabled)
                and final_score >= float(near_miss_cfg.large_partner_min_score)
                and valid_layers >= int(near_miss_cfg.large_partner_min_valid_layers)
                and total_scored_layers >= int(near_miss_cfg.large_partner_min_scored_layers)
                and total_layers >= int(near_miss_cfg.large_partner_min_total_layers)
                and chain_residue_count >= int(near_miss_cfg.large_partner_min_chain_residues)
                and sheet_residue_count >= int(near_miss_cfg.large_partner_min_sheet_residues)
                and avg_radius >= float(near_miss_cfg.large_partner_min_avg_radius)
                and avg_radius <= float(near_miss_cfg.large_partner_max_avg_radius)
            )
        )
    )

    guard_cfg = decision_cfg.low_sheet_wide_guard
    blocked_low_sheet_wide = (
        decision_cfg.use_adjusted_score
        and exceptions_enabled
        and bool(guard_cfg.enabled)
        and chain_residue_count <= int(guard_cfg.max_chain_residues)
        and sheet_residue_count <= int(guard_cfg.max_sheet_residues)
        and total_layers >= int(guard_cfg.min_total_layers)
        and total_scored_layers >= int(guard_cfg.min_scored_layers)
        and avg_radius >= float(guard_cfg.min_avg_radius)
    )

    threshold_pass = final_score >= decision_cfg.barrel_valid_ratio
    passes_barrel_decision = (
        ((enough_scored_layers or rescued_small_barrel) and threshold_pass)
        or rescued_near_miss
    )
    is_barrel = passes_barrel_decision and not blocked_low_sheet_wide

    invalid_reasons: list[str] = []
    junk_reasons: list[str] = []
    for layer in layer_details:
        reason = str(layer.get("reason", "") or "")
        if not reason or reason == "OK":
            continue
        if reason.startswith("JUNK"):
            junk_reasons.append(reason)
        else:
            invalid_reasons.append(reason)

    if invalid_reasons:
        main_reason = Counter(invalid_reasons).most_common(1)[0][0]
    elif junk_reasons:
        main_reason = Counter(junk_reasons).most_common(1)[0][0]
    else:
        main_reason = "Too few valid slices"

    report.update(
        {
            "decision_score": final_score,
            "decision_basis": default_decision_basis,
            "decision_threshold": float(decision_cfg.barrel_valid_ratio),
            "valid_layer_frac": (int(report.get("valid_layers", 0)) / total_layers)
            if total_layers
            else 0.0,
            "scored_layer_frac": scored_layer_frac,
            "junk_layers": len(junk_reasons),
            "invalid_layers": len(invalid_reasons),
            "chain_residues": chain_residue_count,
            "sheet_residues": sheet_residue_count,
            "informative_slices": informative_slice_count,
        }
    )

    if is_barrel:
        reason = "OK"
    elif passes_barrel_decision and blocked_low_sheet_wide:
        reason = (
            "Short low-sheet chain has an unusually large fitted barrel radius "
            f"(chain residues {chain_residue_count} <= {int(guard_cfg.max_chain_residues)}, "
            f"sheet residues {sheet_residue_count} <= {int(guard_cfg.max_sheet_residues)}, "
            f"avg radius {avg_radius:.2f} >= {float(guard_cfg.min_avg_radius):.2f})"
        )
    elif not enough_scored_layers:
        reason = (
            "Too few scored slices for a stable decision "
            f"({total_scored_layers}/{total_layers} = {scored_layer_frac:.2f}, "
            f"need > {float(decision_cfg.min_scored_layer_frac):.2f} "
            f"and >= {int(decision_cfg.min_scored_layers)} layers)"
        )
    else:
        reason = main_reason

    return build_row(
        RESULT_BARREL if is_barrel else RESULT_NON_BARREL,
        reason,
        report,
        result_stage="decision",
        informative_slices=informative_slice_count,
    )


def prepare_one_file(file_path: str, cfg: AppConfig) -> list[dict[str, object]] | dict[str, str]:
    """
    Parse a structure once, run DSSP once, and produce per-chain payloads.
    """
    filename = os.path.basename(file_path)
    try:
        cached_payloads = load_prepare_payloads(file_path, cfg)
        if cached_payloads is not None:
            return cached_payloads

        loader = ProteinLoader(
            file_path,
            dssp_bin=cfg.runtime.dssp_bin_path,
            fail_on_dssp_error=cfg.runtime.fail_on_dssp_error,
        )
    except Exception as exc:
        return {"_error": f"{filename}: {exc}"}

    payloads: list[dict[str, object]] = []
    for chain in loader.model:
        chain_id = chain.id
        try:
            residues_data = loader.get_chain_data(chain_id)
        except Exception as exc:
            return {"_error": f"{filename}: {exc}"}

        payloads.append(
            {
                "filename": filename,
                "chain": chain_id,
                "residues_data": residues_data,
            }
        )

    store_prepare_payloads(file_path, cfg, payloads)
    return payloads


def collect_payloads(
    files: Iterable[str],
    cfg: AppConfig,
    prepare_workers: int,
) -> list[dict[str, object]]:
    """
    Run the preparation phase and collect chain-level payloads.
    """
    file_list = list(files)
    payloads: list[dict[str, object]] = []

    if prepare_workers <= 1:
        for file_path in tqdm(file_list, desc="Preparing", unit="file"):
            result = prepare_one_file(file_path, cfg)
            if isinstance(result, dict) and result.get("_error"):
                tqdm_write(f"  [X] Load failed: {result['_error']}")
                continue
            payloads.extend(result)
        return payloads

    with ProcessPoolExecutor(max_workers=prepare_workers) as executor:
        futures = [executor.submit(prepare_one_file, file_path, cfg) for file_path in file_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preparing", unit="file"):
            try:
                result = future.result()
            except Exception as exc:
                tqdm_write(f"  [X] Prepare worker failed: {exc}")
                continue

            if isinstance(result, dict) and result.get("_error"):
                tqdm_write(f"  [X] Load failed: {result['_error']}")
                continue
            payloads.extend(result)

    return payloads


def run_analysis(
    payloads: list[dict[str, object]],
    cfg: AppConfig,
    workers: int,
) -> list[dict[str, object]]:
    """
    Run the CPU-heavy chain analysis stage.
    """
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(analyze_chain_payload, payload, cfg) for payload in payloads]
        with tqdm(total=len(futures), desc="Analyzing", unit="chain") as progress_bar:
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        {
                            "filename": "",
                            "chain": "",
                            "result": RESULT_ERROR,
                            "result_stage": "error",
                            "reason": f"Worker crashed: {exc}",
                        }
                    )
                finally:
                    progress_bar.update(1)
    return results
