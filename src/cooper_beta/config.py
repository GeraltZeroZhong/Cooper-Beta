from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .constants import (
    DEFAULT_ALLOWED_SUFFIXES,
    DEFAULT_FILL_SHEET_HOLE_LENGTH,
    DEFAULT_INPUT_PATH,
    DEFAULT_MIN_CHAIN_RESIDUES,
    DEFAULT_MIN_INFORMATIVE_SLICES,
    DEFAULT_MIN_SHEET_RESIDUES,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_SLICE_STEP_SIZE,
)
from .exceptions import ConfigValidationError


@dataclass
class RuntimeConfig:
    workers: int | None = None
    prepare_workers: int | None = None
    prepare_batch_size: int = 16
    analysis_batch_size: int = 64
    cpu_reserve: int = 1
    dssp_bin_path: str | None = None
    fail_on_dssp_error: bool = True
    prepare_cache_enabled: bool = True
    prepare_cache_dir: str | None = None
    check_env: bool = False


@dataclass
class InputConfig:
    path: str = DEFAULT_INPUT_PATH
    allowed_suffixes: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_SUFFIXES))
    min_chain_residues: int = DEFAULT_MIN_CHAIN_RESIDUES
    min_sheet_residues: int = DEFAULT_MIN_SHEET_RESIDUES
    min_informative_slices: int = DEFAULT_MIN_INFORMATIVE_SLICES


@dataclass
class OutputConfig:
    csv_path: str = DEFAULT_OUTPUT_CSV
    summary_limit: int = 50


@dataclass
class SlicerConfig:
    step_size: float = DEFAULT_SLICE_STEP_SIZE
    fill_sheet_hole_length: int = DEFAULT_FILL_SHEET_HOLE_LENGTH


@dataclass
class LeastSquaresConfig:
    method: str = "trf"
    loss: str = "soft_l1"
    f_scale: float = 1.0


@dataclass
class EllipseFitConfig:
    min_points_per_slice: int = 7
    max_rmse: float = 3.0
    min_axis: float = 3.0
    max_axis: float = 199.0
    max_flattening: float = 3.5
    least_squares: LeastSquaresConfig = field(default_factory=LeastSquaresConfig)


@dataclass
class SmallBarrelRescueConfig:
    enabled: bool = False
    min_score: float = 0.999
    min_scored_layers: int = 5
    min_total_layers: int = 25
    max_avg_radius: float = 10.5
    compact_enabled: bool = False
    compact_min_score: float = 0.999
    compact_min_scored_layers: int = 4
    compact_min_total_layers: int = 18
    compact_max_total_layers: int = 24
    compact_min_chain_residues: int = 120
    compact_min_sheet_residues: int = 60
    compact_max_avg_radius: float = 12.5
    sparse_enabled: bool = False
    sparse_min_score: float = 0.85
    sparse_min_scored_layers: int = 3
    sparse_min_total_layers: int = 35
    sparse_min_chain_residues: int = 160
    sparse_max_chain_residues: int = 1000
    sparse_min_sheet_residues: int = 70
    sparse_max_avg_radius: float = 35.0


@dataclass
class NearMissRescueConfig:
    enabled: bool = False
    soft_nn_enabled: bool = False
    soft_nn_min_layers: int = 3
    soft_nn_min_inlier_frac: float = 0.70
    soft_nn_max_robust_cv: float = 0.20
    soft_nn_min_total_layers: int = 30
    soft_nn_max_total_layers: int = 40
    soft_nn_min_chain_residues: int = 220
    soft_nn_max_chain_residues: int = 260
    soft_nn_min_sheet_residues: int = 60
    soft_nn_max_sheet_residues: int = 80
    compact_partner_enabled: bool = False
    compact_partner_min_score: float = 0.50
    compact_partner_min_valid_layers: int = 1
    compact_partner_min_scored_layers: int = 2
    compact_partner_min_total_layers: int = 16
    compact_partner_max_total_layers: int = 22
    compact_partner_min_chain_residues: int = 180
    compact_partner_max_chain_residues: int = 220
    compact_partner_min_sheet_residues: int = 65
    compact_partner_max_sheet_residues: int = 85
    compact_partner_min_avg_radius: float = 8.0
    compact_partner_max_avg_radius: float = 13.0
    large_partner_enabled: bool = False
    large_partner_min_score: float = 0.55
    large_partner_min_valid_layers: int = 11
    large_partner_min_scored_layers: int = 14
    large_partner_min_total_layers: int = 35
    large_partner_min_chain_residues: int = 250
    large_partner_min_sheet_residues: int = 140
    large_partner_min_avg_radius: float = 0.0
    large_partner_max_avg_radius: float = 26.0


@dataclass
class LowSheetWideGuardConfig:
    enabled: bool = True
    max_chain_residues: int = 220
    max_sheet_residues: int = 100
    min_avg_radius: float = 13.0
    min_total_layers: int = 20
    min_scored_layers: int = 7


@dataclass
class DecisionConfig:
    barrel_valid_ratio: float = 0.85
    use_adjusted_score: bool = True
    min_intersections_for_scoring: int = 7
    min_scored_layer_frac: float = 0.31
    min_scored_layers: int = 9
    exception_layer_enabled: bool = True
    small_barrel_rescue: SmallBarrelRescueConfig = field(
        default_factory=SmallBarrelRescueConfig
    )
    near_miss_rescue: NearMissRescueConfig = field(default_factory=NearMissRescueConfig)
    low_sheet_wide_guard: LowSheetWideGuardConfig = field(
        default_factory=LowSheetWideGuardConfig
    )


@dataclass
class AxisSearchRefineConfig:
    enabled: bool = True
    angle_deg: float = 5.0


@dataclass
class AxisSearchConfig:
    enabled: bool = True
    refine: AxisSearchRefineConfig = field(default_factory=AxisSearchRefineConfig)


@dataclass
class NearestNeighborRuleConfig:
    enabled: bool = True
    max_robust_cv: float = 0.40
    min_inlier_frac: float = 0.75
    fail_as_junk: bool = True


@dataclass
class AngleOrderRuleConfig:
    enabled: bool = True
    local_step_max: int = 1
    min_local_frac: float = 1.0
    max_mean_circ_dist_norm: float = 0.0


@dataclass
class AngleRuleConfig:
    enabled: bool = True
    max_gap_deg: float = 80.0
    fail_as_junk: bool = False
    order: AngleOrderRuleConfig = field(default_factory=AngleOrderRuleConfig)


@dataclass
class SequenceCoreRuleConfig:
    enabled: bool = True


@dataclass
class AnalyzerRulesConfig:
    nearest_neighbor: NearestNeighborRuleConfig = field(default_factory=NearestNeighborRuleConfig)
    angle: AngleRuleConfig = field(default_factory=AngleRuleConfig)
    sequence_core: SequenceCoreRuleConfig = field(default_factory=SequenceCoreRuleConfig)


@dataclass
class AnalyzerConfig:
    fit: EllipseFitConfig = field(default_factory=EllipseFitConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    axis_search: AxisSearchConfig = field(default_factory=AxisSearchConfig)
    rules: AnalyzerRulesConfig = field(default_factory=AnalyzerRulesConfig)


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    slicer: SlicerConfig = field(default_factory=SlicerConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)


LEGACY_OVERRIDE_PATHS = {
    "DSSP_BIN_PATH": "runtime.dssp_bin_path",
    "PREPARE_BATCH_SIZE": "runtime.prepare_batch_size",
    "ANALYSIS_BATCH_SIZE": "runtime.analysis_batch_size",
    "PREPARE_CACHE_ENABLED": "runtime.prepare_cache_enabled",
    "PREPARE_CACHE_DIR": "runtime.prepare_cache_dir",
    "SLICE_STEP_SIZE": "slicer.step_size",
    "MIN_POINTS_PER_SLICE": "analyzer.fit.min_points_per_slice",
    "MAX_FIT_RMSE": "analyzer.fit.max_rmse",
    "MIN_AXIS": "analyzer.fit.min_axis",
    "MAX_AXIS": "analyzer.fit.max_axis",
    "MAX_FLATTENING": "analyzer.fit.max_flattening",
    "LSQ_METHOD": "analyzer.fit.least_squares.method",
    "LSQ_LOSS": "analyzer.fit.least_squares.loss",
    "LSQ_F_SCALE": "analyzer.fit.least_squares.f_scale",
    "BARREL_VALID_RATIO": "analyzer.decision.barrel_valid_ratio",
    "MIN_INTERSECTIONS_FOR_SCORING": "analyzer.decision.min_intersections_for_scoring",
    "USE_ADJUSTED_SCORE": "analyzer.decision.use_adjusted_score",
    "MIN_SCORED_LAYER_FRAC": "analyzer.decision.min_scored_layer_frac",
    "MIN_SCORED_LAYERS": "analyzer.decision.min_scored_layers",
    "EXCEPTION_LAYER_ENABLED": "analyzer.decision.exception_layer_enabled",
    "SMALL_BARREL_RESCUE_ENABLED": "analyzer.decision.small_barrel_rescue.enabled",
    "NEAR_MISS_RESCUE_ENABLED": "analyzer.decision.near_miss_rescue.enabled",
    "LOW_SHEET_WIDE_GUARD_ENABLED": "analyzer.decision.low_sheet_wide_guard.enabled",
    "AXIS_SEARCH_ENABLED": "analyzer.axis_search.enabled",
    "AXIS_SEARCH_REFINE_ENABLED": "analyzer.axis_search.refine.enabled",
    "AXIS_SEARCH_REFINE_ANGLE_DEG": "analyzer.axis_search.refine.angle_deg",
    "NN_RULE_ENABLED": "analyzer.rules.nearest_neighbor.enabled",
    "NN_MAX_ROBUST_CV": "analyzer.rules.nearest_neighbor.max_robust_cv",
    "NN_MIN_INLIER_FRAC": "analyzer.rules.nearest_neighbor.min_inlier_frac",
    "NN_FAIL_AS_JUNK": "analyzer.rules.nearest_neighbor.fail_as_junk",
    "ANGLE_RULE_ENABLED": "analyzer.rules.angle.enabled",
    "ANGLE_MAX_GAP_DEG": "analyzer.rules.angle.max_gap_deg",
    "ANGLE_ORDER_RULE_ENABLED": "analyzer.rules.angle.order.enabled",
    "ANGLE_ORDER_LOCAL_STEP_MAX": "analyzer.rules.angle.order.local_step_max",
    "ANGLE_ORDER_MIN_LOCAL_FRAC": "analyzer.rules.angle.order.min_local_frac",
    "ANGLE_ORDER_MAX_MEAN_CIRC_DIST_NORM": "analyzer.rules.angle.order.max_mean_circ_dist_norm",
    "ANGLE_FAIL_AS_JUNK": "analyzer.rules.angle.fail_as_junk",
    "SEQUENCE_CORE_RULE_ENABLED": "analyzer.rules.sequence_core.enabled",
    "MIN_CHAIN_RESIDUES": "input.min_chain_residues",
    "MIN_SHEET_RESIDUES": "input.min_sheet_residues",
    "MIN_INFORMATIVE_SLICES": "input.min_informative_slices",
    "SUMMARY_LIMIT": "output.summary_limit",
}


class Config:
    """Legacy flat configuration shim kept for backward compatibility."""


def _register_schema() -> None:
    cs = ConfigStore.instance()
    if getattr(_register_schema, "_done", False):
        return
    cs.store(name="cooper_beta_schema", node=AppConfig)
    _register_schema._done = True


def _to_override_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_to_override_value(v) for v in value)
        return f"[{inner}]"
    return str(value)


def normalize_overrides(overrides: Mapping[str, Any] | list[str] | None = None) -> list[str]:
    if overrides is None:
        return []
    if isinstance(overrides, list):
        return list(overrides)

    normalized: list[str] = []
    for key, value in overrides.items():
        target_key = LEGACY_OVERRIDE_PATHS.get(key, key)
        normalized.append(f"{target_key}={_to_override_value(value)}")
    return normalized


def compose_config(
    overrides: Mapping[str, Any] | list[str] | None = None,
    *,
    config_name: str = "config",
) -> DictConfig:
    _register_schema()
    with initialize_config_module(config_module="cooper_beta.conf", version_base=None):
        file_cfg = compose(config_name=config_name, overrides=normalize_overrides(overrides))
    structured_cfg = OmegaConf.structured(AppConfig)
    return OmegaConf.merge(structured_cfg, file_cfg)


def build_config(
    overrides: Mapping[str, Any] | list[str] | None = None,
    *,
    config_name: str = "config",
) -> AppConfig:
    cfg = compose_config(overrides, config_name=config_name)
    app_cfg = OmegaConf.to_object(cfg)
    if not isinstance(app_cfg, AppConfig):
        raise TypeError("Hydra returned an unexpected configuration object.")
    validate_config(app_cfg)
    sync_legacy_config(app_cfg)
    return app_cfg


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ConfigValidationError(f"`{name}` must be greater than 0.")


def _require_non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ConfigValidationError(f"`{name}` must be greater than or equal to 0.")


def _require_ratio(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigValidationError(f"`{name}` must be between 0 and 1.")


def _require_ordered_range(name_min: str, min_value: int | float, name_max: str, max_value: int | float) -> None:
    if max_value < min_value:
        raise ConfigValidationError(f"`{name_max}` must be >= `{name_min}`.")


def validate_config(cfg: AppConfig) -> None:
    """Validate user-editable configuration values before running analysis."""
    if cfg.runtime.workers is not None:
        _require_positive("runtime.workers", int(cfg.runtime.workers))
    if cfg.runtime.prepare_workers is not None:
        _require_positive("runtime.prepare_workers", int(cfg.runtime.prepare_workers))
    _require_positive("runtime.prepare_batch_size", int(cfg.runtime.prepare_batch_size))
    _require_positive("runtime.analysis_batch_size", int(cfg.runtime.analysis_batch_size))
    _require_non_negative("runtime.cpu_reserve", int(cfg.runtime.cpu_reserve))

    if not cfg.input.allowed_suffixes:
        raise ConfigValidationError("`input.allowed_suffixes` must contain at least one suffix.")
    for suffix in cfg.input.allowed_suffixes:
        if not str(suffix).startswith("."):
            raise ConfigValidationError("Each `input.allowed_suffixes` value must start with '.'.")
    _require_non_negative("input.min_chain_residues", int(cfg.input.min_chain_residues))
    _require_non_negative("input.min_sheet_residues", int(cfg.input.min_sheet_residues))
    _require_non_negative("input.min_informative_slices", int(cfg.input.min_informative_slices))

    _require_positive("slicer.step_size", float(cfg.slicer.step_size))
    _require_non_negative("slicer.fill_sheet_hole_length", int(cfg.slicer.fill_sheet_hole_length))

    fit = cfg.analyzer.fit
    _require_positive("analyzer.fit.min_points_per_slice", int(fit.min_points_per_slice))
    _require_positive("analyzer.fit.max_rmse", float(fit.max_rmse))
    _require_positive("analyzer.fit.min_axis", float(fit.min_axis))
    _require_positive("analyzer.fit.max_axis", float(fit.max_axis))
    if float(fit.max_axis) < float(fit.min_axis):
        raise ConfigValidationError("`analyzer.fit.max_axis` must be >= `analyzer.fit.min_axis`.")
    if float(fit.max_flattening) < 1.0:
        raise ConfigValidationError("`analyzer.fit.max_flattening` must be >= 1.")
    _require_positive("analyzer.fit.least_squares.f_scale", float(fit.least_squares.f_scale))

    decision = cfg.analyzer.decision
    _require_ratio("analyzer.decision.barrel_valid_ratio", float(decision.barrel_valid_ratio))
    _require_positive(
        "analyzer.decision.min_intersections_for_scoring",
        int(decision.min_intersections_for_scoring),
    )
    _require_ratio("analyzer.decision.min_scored_layer_frac", float(decision.min_scored_layer_frac))
    _require_non_negative("analyzer.decision.min_scored_layers", int(decision.min_scored_layers))

    small = decision.small_barrel_rescue
    _require_ratio("analyzer.decision.small_barrel_rescue.min_score", float(small.min_score))
    _require_non_negative(
        "analyzer.decision.small_barrel_rescue.min_scored_layers",
        int(small.min_scored_layers),
    )
    _require_non_negative(
        "analyzer.decision.small_barrel_rescue.min_total_layers",
        int(small.min_total_layers),
    )
    _require_non_negative(
        "analyzer.decision.small_barrel_rescue.max_avg_radius",
        float(small.max_avg_radius),
    )
    _require_ratio(
        "analyzer.decision.small_barrel_rescue.compact_min_score",
        float(small.compact_min_score),
    )
    _require_non_negative(
        "analyzer.decision.small_barrel_rescue.compact_min_scored_layers",
        int(small.compact_min_scored_layers),
    )
    _require_non_negative(
        "analyzer.decision.small_barrel_rescue.compact_min_total_layers",
        int(small.compact_min_total_layers),
    )
    _require_non_negative(
        "analyzer.decision.small_barrel_rescue.compact_max_total_layers",
        int(small.compact_max_total_layers),
    )
    _require_ordered_range(
        "analyzer.decision.small_barrel_rescue.compact_min_total_layers",
        int(small.compact_min_total_layers),
        "analyzer.decision.small_barrel_rescue.compact_max_total_layers",
        int(small.compact_max_total_layers),
    )
    _require_non_negative(
        "analyzer.decision.small_barrel_rescue.compact_min_chain_residues",
        int(small.compact_min_chain_residues),
    )
    _require_non_negative(
        "analyzer.decision.small_barrel_rescue.compact_min_sheet_residues",
        int(small.compact_min_sheet_residues),
    )
    _require_non_negative(
        "analyzer.decision.small_barrel_rescue.compact_max_avg_radius",
        float(small.compact_max_avg_radius),
    )
    _require_ratio(
        "analyzer.decision.small_barrel_rescue.sparse_min_score",
        float(small.sparse_min_score),
    )
    _require_non_negative(
        "analyzer.decision.small_barrel_rescue.sparse_min_scored_layers",
        int(small.sparse_min_scored_layers),
    )
    _require_non_negative(
        "analyzer.decision.small_barrel_rescue.sparse_min_total_layers",
        int(small.sparse_min_total_layers),
    )
    _require_non_negative(
        "analyzer.decision.small_barrel_rescue.sparse_min_chain_residues",
        int(small.sparse_min_chain_residues),
    )
    _require_non_negative(
        "analyzer.decision.small_barrel_rescue.sparse_max_chain_residues",
        int(small.sparse_max_chain_residues),
    )
    _require_ordered_range(
        "analyzer.decision.small_barrel_rescue.sparse_min_chain_residues",
        int(small.sparse_min_chain_residues),
        "analyzer.decision.small_barrel_rescue.sparse_max_chain_residues",
        int(small.sparse_max_chain_residues),
    )
    _require_non_negative(
        "analyzer.decision.small_barrel_rescue.sparse_min_sheet_residues",
        int(small.sparse_min_sheet_residues),
    )
    _require_non_negative(
        "analyzer.decision.small_barrel_rescue.sparse_max_avg_radius",
        float(small.sparse_max_avg_radius),
    )

    near = decision.near_miss_rescue
    _require_non_negative(
        "analyzer.decision.near_miss_rescue.soft_nn_min_layers",
        int(near.soft_nn_min_layers),
    )
    _require_ratio(
        "analyzer.decision.near_miss_rescue.soft_nn_min_inlier_frac",
        float(near.soft_nn_min_inlier_frac),
    )
    _require_non_negative(
        "analyzer.decision.near_miss_rescue.soft_nn_max_robust_cv",
        float(near.soft_nn_max_robust_cv),
    )
    for prefix in ("soft_nn", "compact_partner"):
        min_total = getattr(near, f"{prefix}_min_total_layers")
        max_total = getattr(near, f"{prefix}_max_total_layers")
        min_chain = getattr(near, f"{prefix}_min_chain_residues")
        max_chain = getattr(near, f"{prefix}_max_chain_residues")
        min_sheet = getattr(near, f"{prefix}_min_sheet_residues")
        max_sheet = getattr(near, f"{prefix}_max_sheet_residues")
        _require_non_negative(f"analyzer.decision.near_miss_rescue.{prefix}_min_total_layers", int(min_total))
        _require_non_negative(f"analyzer.decision.near_miss_rescue.{prefix}_max_total_layers", int(max_total))
        _require_ordered_range(
            f"analyzer.decision.near_miss_rescue.{prefix}_min_total_layers",
            int(min_total),
            f"analyzer.decision.near_miss_rescue.{prefix}_max_total_layers",
            int(max_total),
        )
        _require_non_negative(f"analyzer.decision.near_miss_rescue.{prefix}_min_chain_residues", int(min_chain))
        _require_non_negative(f"analyzer.decision.near_miss_rescue.{prefix}_max_chain_residues", int(max_chain))
        _require_ordered_range(
            f"analyzer.decision.near_miss_rescue.{prefix}_min_chain_residues",
            int(min_chain),
            f"analyzer.decision.near_miss_rescue.{prefix}_max_chain_residues",
            int(max_chain),
        )
        _require_non_negative(f"analyzer.decision.near_miss_rescue.{prefix}_min_sheet_residues", int(min_sheet))
        _require_non_negative(f"analyzer.decision.near_miss_rescue.{prefix}_max_sheet_residues", int(max_sheet))
        _require_ordered_range(
            f"analyzer.decision.near_miss_rescue.{prefix}_min_sheet_residues",
            int(min_sheet),
            f"analyzer.decision.near_miss_rescue.{prefix}_max_sheet_residues",
            int(max_sheet),
        )
    _require_ratio(
        "analyzer.decision.near_miss_rescue.compact_partner_min_score",
        float(near.compact_partner_min_score),
    )
    _require_non_negative(
        "analyzer.decision.near_miss_rescue.compact_partner_min_valid_layers",
        int(near.compact_partner_min_valid_layers),
    )
    _require_non_negative(
        "analyzer.decision.near_miss_rescue.compact_partner_min_scored_layers",
        int(near.compact_partner_min_scored_layers),
    )
    _require_non_negative(
        "analyzer.decision.near_miss_rescue.compact_partner_min_avg_radius",
        float(near.compact_partner_min_avg_radius),
    )
    _require_non_negative(
        "analyzer.decision.near_miss_rescue.compact_partner_max_avg_radius",
        float(near.compact_partner_max_avg_radius),
    )
    _require_ordered_range(
        "analyzer.decision.near_miss_rescue.compact_partner_min_avg_radius",
        float(near.compact_partner_min_avg_radius),
        "analyzer.decision.near_miss_rescue.compact_partner_max_avg_radius",
        float(near.compact_partner_max_avg_radius),
    )
    _require_ratio(
        "analyzer.decision.near_miss_rescue.large_partner_min_score",
        float(near.large_partner_min_score),
    )
    _require_non_negative(
        "analyzer.decision.near_miss_rescue.large_partner_min_valid_layers",
        int(near.large_partner_min_valid_layers),
    )
    _require_non_negative(
        "analyzer.decision.near_miss_rescue.large_partner_min_scored_layers",
        int(near.large_partner_min_scored_layers),
    )
    _require_non_negative(
        "analyzer.decision.near_miss_rescue.large_partner_min_total_layers",
        int(near.large_partner_min_total_layers),
    )
    _require_non_negative(
        "analyzer.decision.near_miss_rescue.large_partner_min_chain_residues",
        int(near.large_partner_min_chain_residues),
    )
    _require_non_negative(
        "analyzer.decision.near_miss_rescue.large_partner_min_sheet_residues",
        int(near.large_partner_min_sheet_residues),
    )
    _require_non_negative(
        "analyzer.decision.near_miss_rescue.large_partner_min_avg_radius",
        float(near.large_partner_min_avg_radius),
    )
    _require_non_negative(
        "analyzer.decision.near_miss_rescue.large_partner_max_avg_radius",
        float(near.large_partner_max_avg_radius),
    )
    _require_ordered_range(
        "analyzer.decision.near_miss_rescue.large_partner_min_avg_radius",
        float(near.large_partner_min_avg_radius),
        "analyzer.decision.near_miss_rescue.large_partner_max_avg_radius",
        float(near.large_partner_max_avg_radius),
    )

    guard = decision.low_sheet_wide_guard
    _require_non_negative("analyzer.decision.low_sheet_wide_guard.max_chain_residues", int(guard.max_chain_residues))
    _require_non_negative("analyzer.decision.low_sheet_wide_guard.max_sheet_residues", int(guard.max_sheet_residues))
    _require_non_negative("analyzer.decision.low_sheet_wide_guard.min_avg_radius", float(guard.min_avg_radius))
    _require_non_negative("analyzer.decision.low_sheet_wide_guard.min_total_layers", int(guard.min_total_layers))
    _require_non_negative("analyzer.decision.low_sheet_wide_guard.min_scored_layers", int(guard.min_scored_layers))

    refine_angle = float(cfg.analyzer.axis_search.refine.angle_deg)
    if not 0.0 <= refine_angle <= 180.0:
        raise ConfigValidationError("`analyzer.axis_search.refine.angle_deg` must be between 0 and 180.")

    nn = cfg.analyzer.rules.nearest_neighbor
    _require_non_negative("analyzer.rules.nearest_neighbor.max_robust_cv", float(nn.max_robust_cv))
    _require_ratio("analyzer.rules.nearest_neighbor.min_inlier_frac", float(nn.min_inlier_frac))

    angle = cfg.analyzer.rules.angle
    if not 0.0 <= float(angle.max_gap_deg) <= 360.0:
        raise ConfigValidationError("`analyzer.rules.angle.max_gap_deg` must be between 0 and 360.")
    _require_non_negative("analyzer.rules.angle.order.local_step_max", int(angle.order.local_step_max))
    _require_ratio("analyzer.rules.angle.order.min_local_frac", float(angle.order.min_local_frac))
    _require_ratio(
        "analyzer.rules.angle.order.max_mean_circ_dist_norm",
        float(angle.order.max_mean_circ_dist_norm),
    )


def sync_legacy_config(cfg: AppConfig) -> None:
    for legacy_name, path in LEGACY_OVERRIDE_PATHS.items():
        value = cfg
        for part in path.split("."):
            value = getattr(value, part)
        setattr(Config, legacy_name, value)


def default_config() -> AppConfig:
    return build_config()


sync_legacy_config(AppConfig())
