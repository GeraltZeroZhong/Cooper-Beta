"""Validator configuration loader.

Design goals
- Single YAML file controls all thresholds.
- Backwards compatible: if YAML is missing or invalid, fall back to in-code defaults.
- Accept either a dict (already-loaded config) or a filesystem path.

The default lookup strategy:
1) If `config` is a path: load it.
2) If `config` is None:
   - if env `BETA_COOPER_CONFIG` points to a file, load it;
   - else try `<repo_root>/validator.yaml` (repo_root inferred from this file);
   - else fall back to defaults.

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml


def _deep_update(base: Dict[str, Any], upd: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested dict (in-place) and return base."""
    for k, v in upd.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            _deep_update(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base


def _repo_root() -> Path:
    # .../beta_cooper/validator/config.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def _default_config_path() -> Optional[Path]:
    env = os.environ.get("BETA_COOPER_CONFIG", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p

    # Prefer repo-root validator.yaml when running from source tree
    p = _repo_root() / "validator.yaml"
    if p.is_file():
        return p

    # Also accept CWD validator.yaml
    p2 = Path.cwd() / "validator.yaml"
    if p2.is_file():
        return p2

    return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def default_config_dict() -> Dict[str, Any]:
    """In-code defaults (mirrors the shipped validator.yaml)."""
    return {
        "extractor": {
            "dssp_timeout_sec": 5,
            "jump_tolerance": 4,
            "min_segment_len": 3,
            "plddt_detection": {
                "enable": True,
                "bfactor_min": 0.0,
                "bfactor_max": 100.0,
                "median_min": 50.0,
                "p90_min": 70.0,
            },
        },
        "analyzer": {
            "min_axis_variance": 1e-5,
            "axis_dominance_fallback_ratio": 1.15,
            "mcd_support_fraction": 0.75,
            "mcd_inlier_percentile": 80,
            "min_core_atoms": 15,
            "score_threshold": 0.60,
            "penalties": {
                "ellipticity": {"thr": 1.8, "mult": 0.6},
                "stability": {"cov_eigen_ratio_thr": 50.0, "mult": 0.05, "max_penalty": 0.30},
                "ring_cv": {"thr_small": 0.20, "thr_large": 0.25, "mult_small": 3.5, "mult_large": 2.5},
                "thickness": {"thr_small": 0.25, "thr_large": 0.30, "mult_small": 3.0, "mult_large": 2.5},
                "kurtosis": {
                    "thick_thr": 0.35,
                    "kurtosis_thr": -1.0,
                    "penalty_if_thick": 0.5,
                    "penalty_else": 0.1,
                },
                "angular_gap": {"thr_deg": 60.0, "mult": 0.01},
                "z_profile": {"thr_cv": 0.30, "penalty": 0.30},
                "fallback": {"penalty": 0.15},
                "segments": {"min_segments": 3, "penalty": 0.25},
                "surface_rmse_norm": {"thr": 0.25, "mult": 1.5},
            },
        },
        "_topology_common": {
            "energy_cutoff": -0.5,
            "min_hbonds_per_edge": 2,
            "nn_dist_tol": 0.75,
            "radial_min_rel_height": 0.15,
        },
        "validator": {
            "confidence_threshold": 0.60,
            "topology_quality": {
                "enable": True,
                "apply_to_confidence": True,
                "components": [
                    {"name": "graph_cyclicity", "weight": 1.0, "transform": "clip01"},
                    {"name": "graph_connected", "weight": 1.0, "transform": "clip01"},
                    {"name": "degree2_fraction", "weight": 1.0, "transform": "clip01"},
                    {"name": "radial_unimodal_score", "weight": 1.0, "transform": "clip01"},
                    {"name": "edge_hbond_cv", "weight": 1.0, "transform": "inv1p"},
                    {"name": "registry_shift_edge_mean_std", "weight": 1.0, "transform": "exp_decay", "scale": 3.0},
                    {"name": "tilt_angle_std_deg", "weight": 1.0, "transform": "exp_decay", "scale": 25.0},
                    {"name": "vector_alt_score", "weight": 1.0, "transform": "neg_clip01"},
                ],
            },
            "plddt_gate": {"enable": True, "threshold": 0.70, "mean_min": 0.70, "pass_fraction_min": 0.85},
            "hard_fail": {
                "cyclicity_required": 1.0,
                "graph_connected_required": 1.0,
                "degree2_fraction_min": 0.75,
                "vector_alt_max": -0.25,
                "radial_unimodal_min": 0.50,
                "tilt_std_max_deg": 30.0,
                "registry_std_max": 4.0,
                "hbond_uniformity": {"mean_min": 2.0, "cv_max": 1.2},
            },
            "enforce_confidence_threshold": True,
        },
    }


def load_config(config: Optional[Union[str, Path, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """Load config from YAML (path) or merge dict override onto defaults."""
    base = default_config_dict()

    # 1) YAML file (if supplied)
    path: Optional[Path] = None
    if isinstance(config, (str, Path)):
        path = Path(config).expanduser().resolve()
    elif config is None:
        path = _default_config_path()

    if path is not None and path.is_file():
        try:
            y = _load_yaml(path)
            _deep_update(base, y)
        except Exception:
            # Keep defaults if YAML parsing fails
            pass

    # 2) dict override
    if isinstance(config, Mapping) and not isinstance(config, (str, Path)):
        _deep_update(base, config)

    return base
