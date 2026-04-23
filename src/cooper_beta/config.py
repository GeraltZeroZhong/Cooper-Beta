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


@dataclass
class RuntimeConfig:
    workers: int | None = None
    prepare_workers: int | None = None
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
class DecisionConfig:
    barrel_valid_ratio: float = 0.5
    use_adjusted_score: bool = True
    min_intersections_for_scoring: int = 7
    min_scored_layer_frac: float = 0.20


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
    sync_legacy_config(app_cfg)
    return app_cfg


def sync_legacy_config(cfg: AppConfig) -> None:
    for legacy_name, path in LEGACY_OVERRIDE_PATHS.items():
        value = cfg
        for part in path.split("."):
            value = getattr(value, part)
        setattr(Config, legacy_name, value)


def default_config() -> AppConfig:
    return build_config()


sync_legacy_config(AppConfig())
