# beta_cooper/validator/__init__.py

import numpy as np

from .config import load_config

from .extractor import BarrelExtractor
from .analyzer import BarrelAnalyzer
from .topology_checks import compute_vector_alt_score, compute_graph_cyclicity_score, compute_topology_graph_metrics

from beta_cooper.geometry.cleaner import BarrelCleaner
from beta_cooper.geometry.topology import BarrelTopology

class BarrelValidator:
    """
    BarrelValidator (Refactored)
    Facade class that orchestrates the validation pipeline:
    1. Extraction (via BarrelExtractor): PDB -> Segments & Coords
    2. Analysis (via BarrelAnalyzer): Geometry -> Metrics & Score
    """

    def __init__(self, pdb_file, chain_id='A', config=None):
        self.pdb_file = pdb_file
        self.target_chain_id = chain_id
        # config can be: None, path to yaml, or dict override
        self.config = load_config(config)

        # Allow YAML to override default chain_id if the caller didn't explicitly set one.
        # (保持向后兼容：如果调用者传入 chain_id，就优先使用调用者的.)
        try:
            if chain_id == 'A':
                cfg_chain = self.config.get('validator', {}).get('chain_id', None)
                if isinstance(cfg_chain, str) and cfg_chain.strip():
                    self.target_chain_id = cfg_chain.strip()
        except Exception:
            pass

    def validate(self):
        """
        Runs the full validation pipeline.
        Returns a dictionary with status, score, metrics, and debug data.
        """
        # --- Step 1: Extraction ---
        # 负责文件读取、OPM 清洗、DSSP 调用、Beta 片段提取
        extractor_cfg = self.config.get("extractor", {}) if isinstance(self.config, dict) else {}
        extractor = BarrelExtractor(self.pdb_file, self.target_chain_id, config=extractor_cfg)
        segments, all_coords = extractor.run()

        # 如果提取失败（例如文件不存在、DSSP 报错），由 Analyzer 生成标准失败响应
        # 或者在这里直接处理。为了统一 metrics 格式，我们让 Analyzer 处理空数据报错。
        
        # --- Step 2: Analysis ---
        # 负责几何计算、异常值剔除、指标统计、打分
        analyzer_cfg = self.config.get("analyzer", {}) if isinstance(self.config, dict) else {}
        analyzer = BarrelAnalyzer(config=analyzer_cfg)
        result = analyzer.run(segments, all_coords)

        # --- Step 2.5: Topology-derived math checks (graph closure & vector alternation) ---
        # 这些指标依赖 strands 的顺序；因此复用 geometry.cleaner+topology 生成 strands。
        topo_metrics = {
            "graph_cyclicity": np.nan,
            "graph_connected": np.nan,
            "degree_mean": np.nan,
            "degree_std": np.nan,
            "degree2_fraction": np.nan,
            "edge_hbond_mean": np.nan,
            "edge_hbond_std": np.nan,
            "edge_hbond_cv": np.nan,
            "edge_hbond_min": np.nan,
            "edge_hbond_max": np.nan,
            "registry_shift_edge_mean_std": np.nan,
            "registry_shift_edge_abs_median_std": np.nan,
            "registry_shift_sign_consistency": np.nan,
            "shear_number_est": np.nan,
            "tilt_angle_mean_deg": np.nan,
            "tilt_angle_std_deg": np.nan,
            "radial_n_peaks": np.nan,
            "radial_unimodal_score": np.nan,
            "vector_alt_score": np.nan,
            "n_strands_topology": np.nan,
            "topology_quality_score": np.nan,
            # Optional pLDDT gate (AlphaFold-like only)
            "plddt_active": np.nan,
            "plddt_mean": np.nan,
            "plddt_pass_fraction": np.nan,
            "plddt_gate_threshold": float(
                (self.config.get("validator", {}).get("plddt_gate", {}) if isinstance(self.config, dict) else {}).get("threshold", 0.7)
            ),
        }
        try:
            if segments is not None and all_coords is not None and len(all_coords) >= 30:
                audit = {}
                cleaner = BarrelCleaner(audit)
                clean_segments, clean_coords = cleaner.run(segments, all_coords)
                if clean_segments and len(clean_coords) > 15:
                    aligned_segments = cleaner.apply_alignment(clean_segments)
                    topology = BarrelTopology(audit)
                    strands = topology.run(aligned_segments)
                    topo_metrics["n_strands_topology"] = float(len(strands))

                    topo_metrics["vector_alt_score"] = compute_vector_alt_score(strands)

                    topo_common = self.config.get("_topology_common", {}) if isinstance(self.config, dict) else {}
                    g = compute_topology_graph_metrics(
                        strands=strands,
                        all_coords=all_coords,
                        all_resids=getattr(extractor, "all_resids", None),
                        dssp_hbonds=getattr(extractor, "dssp_hbonds", None),
                        rotation_matrix=getattr(cleaner, "rotation_matrix", None),
                        centroid=getattr(cleaner, "centroid", None),
                        energy_cutoff=float(topo_common.get("energy_cutoff", -0.5)),
                        min_hbonds_per_edge=int(topo_common.get("min_hbonds_per_edge", 2)),
                        nn_dist_tol=float(topo_common.get("nn_dist_tol", 0.75)),
                        radial_min_rel_height=float(topo_common.get("radial_min_rel_height", 0.15)),
                    )
                    if isinstance(g, dict):
                        topo_metrics.update(g)

                    # Topology quality score (0..1): configurable weighted aggregator.
                    # This is used as a soft gate to reduce false positives on huge datasets.
                    def _tq_transform(v, spec):
                        if v is None or not np.isfinite(v):
                            return None
                        t = str(spec.get("transform", "clip01")).strip()
                        if t == "clip01":
                            return float(np.clip(v, 0.0, 1.0))
                        if t == "inv1p":
                            return float(1.0 / (1.0 + max(0.0, float(v))))
                        if t == "exp_decay":
                            scale = float(spec.get("scale", 1.0))
                            scale = max(scale, 1e-9)
                            return float(np.exp(-max(0.0, float(v)) / scale))
                        if t == "neg_clip01":
                            return float(np.clip(-float(v), 0.0, 1.0))
                        # fallback
                        return float(np.clip(v, 0.0, 1.0))

                    tq_cfg = self.config.get("validator", {}).get("topology_quality", {}) if isinstance(self.config, dict) else {}
                    if bool(tq_cfg.get("enable", True)):
                        items = tq_cfg.get("components", [])
                        num = 0.0
                        den = 0.0
                        if isinstance(items, list):
                            for spec in items:
                                if not isinstance(spec, dict):
                                    continue
                                name = spec.get("name", None)
                                if not name:
                                    continue
                                val = topo_metrics.get(str(name), np.nan)
                                vv = _tq_transform(val, spec)
                                if vv is None:
                                    continue
                                w = float(spec.get("weight", 1.0))
                                if not np.isfinite(w) or w <= 0:
                                    continue
                                num += w * vv
                                den += w
                        if den > 0:
                            topo_metrics["topology_quality_score"] = float(num / den)

                # Optional pLDDT gate (AlphaFold-like only; extractor decides whether active)
                try:
                    if getattr(extractor, "plddt_active", False) and getattr(extractor, "all_plddt", None) is not None:
                        p = np.asarray(getattr(extractor, "all_plddt"), dtype=float)
                        p = p[np.isfinite(p)]
                        topo_metrics["plddt_active"] = 1.0
                        if len(p) > 0:
                            thr = float(topo_metrics.get("plddt_gate_threshold", 0.7))
                            topo_metrics["plddt_mean"] = float(np.mean(p))
                            topo_metrics["plddt_pass_fraction"] = float(np.mean(p >= thr))
                    else:
                        topo_metrics["plddt_active"] = 0.0
                except Exception:
                    pass

        except Exception:
            # Topology checks should never crash validation
            pass

        # 将新指标注入 metrics（并保持 key 在 metrics 内，不污染顶层字段）
        if "metrics" in result and isinstance(result["metrics"], dict):
            result["metrics"].update(topo_metrics)

        # 可选的“硬性”判定：当几何分数通过但拓扑闭合/反平行明显失败时，降低有效性。
        # 这里采用保守阈值，避免在缺少氢键信息/映射不稳时误杀。
        try:
            if result.get("is_valid", False):
                # 1) Apply combined topology quality score as a multiplicative confidence gate
                conf = float(result.get("confidence", 0.0))
                tq_cfg = self.config.get("validator", {}).get("topology_quality", {}) if isinstance(self.config, dict) else {}
                tq = topo_metrics.get("topology_quality_score", np.nan)
                if bool(tq_cfg.get("enable", True)) and bool(tq_cfg.get("apply_to_confidence", True)) and np.isfinite(tq):
                    conf = conf * float(np.clip(tq, 0.0, 1.0))
                    result["confidence"] = conf

                # 2) Optional pLDDT gate (only when extractor marked it as active)
                p_active = topo_metrics.get("plddt_active", np.nan)
                p_mean = topo_metrics.get("plddt_mean", np.nan)
                p_frac = topo_metrics.get("plddt_pass_fraction", np.nan)
                pg_cfg = self.config.get("validator", {}).get("plddt_gate", {}) if isinstance(self.config, dict) else {}
                if bool(pg_cfg.get("enable", True)) and np.isfinite(p_active) and p_active >= 1.0:
                    mean_min = float(pg_cfg.get("mean_min", 0.7))
                    frac_min = float(pg_cfg.get("pass_fraction_min", 0.85))
                    if (np.isfinite(p_mean) and p_mean < mean_min) or (np.isfinite(p_frac) and p_frac < frac_min):
                        result.update({
                            "status": "FAIL_PLDDT_GATE",
                            "is_valid": False,
                            "issue": "PLDDT_BELOW_THRESHOLD",
                            "confidence": min(float(result.get("confidence", 0.0)), 0.30),
                        })

                # 3) Topology structural hard checks (only if still valid)
                if result.get("is_valid", False):
                    gc = topo_metrics.get("graph_cyclicity", np.nan)
                    va = topo_metrics.get("vector_alt_score", np.nan)
                    gconn = topo_metrics.get("graph_connected", np.nan)
                    deg2 = topo_metrics.get("degree2_fraction", np.nan)
                    hcv = topo_metrics.get("edge_hbond_cv", np.nan)
                    hmean = topo_metrics.get("edge_hbond_mean", np.nan)
                    reg_std = topo_metrics.get("registry_shift_edge_mean_std", np.nan)
                    tilt_std = topo_metrics.get("tilt_angle_std_deg", np.nan)
                    uni = topo_metrics.get("radial_unimodal_score", np.nan)

                    hf = self.config.get("validator", {}).get("hard_fail", {}) if isinstance(self.config, dict) else {}
                    cyc_req = float(hf.get("cyclicity_required", 1.0))
                    gconn_req = float(hf.get("graph_connected_required", 1.0))
                    deg2_min = float(hf.get("degree2_fraction_min", 0.75))
                    v_alt_max = float(hf.get("vector_alt_max", -0.25))
                    uni_min = float(hf.get("radial_unimodal_min", 0.50))
                    tilt_max = float(hf.get("tilt_std_max_deg", 30.0))
                    reg_max = float(hf.get("registry_std_max", 4.0))
                    hu = hf.get("hbond_uniformity", {}) if isinstance(hf.get("hbond_uniformity", {}), dict) else {}
                    h_mean_min = float(hu.get("mean_min", 2.0))
                    h_cv_max = float(hu.get("cv_max", 1.2))

                    if np.isfinite(gc) and gc < cyc_req:
                        result.update({
                            "status": "FAIL_CYCLICITY",
                            "is_valid": False,
                            "issue": "TOPOLOGY_NOT_CLOSED",
                            "confidence": min(float(result.get("confidence", 0.0)), 0.35),
                        })
                    elif np.isfinite(gconn) and gconn < gconn_req:
                        result.update({
                            "status": "FAIL_GRAPH_CONNECT",
                            "is_valid": False,
                            "issue": "GRAPH_DISCONNECTED",
                            "confidence": min(float(result.get("confidence", 0.0)), 0.35),
                        })
                    elif np.isfinite(deg2) and deg2 < deg2_min:
                        result.update({
                            "status": "FAIL_DEGREE",
                            "is_valid": False,
                            "issue": "DEGREE_CONSTRAINT",
                            "confidence": min(float(result.get("confidence", 0.0)), 0.35),
                        })
                    elif np.isfinite(va) and va > v_alt_max:
                        result.update({
                            "status": "FAIL_VECTOR_ALT",
                            "is_valid": False,
                            "issue": "VECTOR_DIRECTION_ISSUE",
                            "confidence": min(float(result.get("confidence", 0.0)), 0.35),
                        })
                    elif np.isfinite(uni) and uni < uni_min:
                        result.update({
                            "status": "FAIL_RADIAL_MULTI",
                            "is_valid": False,
                            "issue": "RADIAL_MULTIMODAL",
                            "confidence": min(float(result.get("confidence", 0.0)), 0.35),
                        })
                    elif np.isfinite(tilt_std) and tilt_std > tilt_max:
                        result.update({
                            "status": "FAIL_TILT",
                            "is_valid": False,
                            "issue": "TILT_INCONSISTENT",
                            "confidence": min(float(result.get("confidence", 0.0)), 0.35),
                        })
                    elif np.isfinite(reg_std) and reg_std > reg_max:
                        result.update({
                            "status": "FAIL_REGISTRY",
                            "is_valid": False,
                            "issue": "REGISTRY_INCONSISTENT",
                            "confidence": min(float(result.get("confidence", 0.0)), 0.35),
                        })
                    elif np.isfinite(hcv) and np.isfinite(hmean) and hmean >= h_mean_min and hcv > h_cv_max:
                        result.update({
                            "status": "FAIL_HBOND_UNIFORMITY",
                            "is_valid": False,
                            "issue": "EDGE_HBOND_UNEVEN",
                            "confidence": min(float(result.get("confidence", 0.0)), 0.35),
                        })

                # 4) After scaling, enforce global confidence threshold (configurable)
                vcfg = self.config.get("validator", {}) if isinstance(self.config, dict) else {}
                conf_thr = float(vcfg.get("confidence_threshold", 0.60))
                enforce = bool(vcfg.get("enforce_confidence_threshold", True))
                if enforce and result.get("is_valid", False) and float(result.get("confidence", 0.0)) <= conf_thr:
                    result.update({
                        "status": "FAIL_TOPOLOGY_SCORE",
                        "is_valid": False,
                        "issue": "TOPOLOGY_SCORE_LOW",
                    })
        except Exception:
            pass
        
        # --- Step 3: Combine Results ---
        # 将提取的原始数据 (debug_segments/coords) 注入到结果中
        # 这样上层应用 (如 batch_process.py) 可以拿去给 Geometry 模块使用
        result.update({
            "debug_segments": segments,
            "debug_coords": all_coords
        })
        
        return result