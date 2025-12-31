import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet

class BarrelAnalyzer:
    """
    负责对提取出的 β-折叠原子坐标进行几何分析和质量评估。
    包含 PCA 分析、MCD 异常值检测和几何指标计算。
    """

    def __init__(self, config=None):
        # `config` is the loaded validator config dict; we only read `analyzer` subtree.
        self.config = config if isinstance(config, dict) else {}

    def _cfg(self, key_path, default=None):
        """Fetch nested config value by a dotted key path, e.g. 'penalties.ring_cv.thr_small'."""
        node = self.config
        try:
            for k in str(key_path).split('.'):
                if not isinstance(node, dict) or k not in node:
                    return default
                node = node[k]
            return node
        except Exception:
            return default

    def run(self, segments, all_coords):
        """
        执行分析流程。
        输入: segments (list of arrays), all_coords (N, 3 array)
        输出: 包含评分、状态和详细指标的字典
        """
        metrics = self._get_empty_metrics()
        
        # 1. 基础检查
        if segments is None or all_coords is None:
            return self._fail("FAIL_EXTRACT", "Extraction failed", metrics)

        metrics["n_segments"] = len(segments)
        metrics["n_atoms_total"] = len(all_coords)
        
        if len(all_coords) < 30:
            return self._fail("FAIL_LOW_BETA", f"Insufficient beta atoms ({len(all_coords)}<30)", metrics)

        # 2. PCA & 几何维度检查
        try:
            pca = PCA(n_components=3)
            centered_raw = all_coords - np.mean(all_coords, axis=0)
            aligned = pca.fit_transform(centered_raw)
            vars = pca.explained_variance_ 
            
            min_var = float(self._cfg("min_axis_variance", 1e-5))
            if vars[1] < min_var or vars[2] < min_var: 
                return self._fail("FAIL_DEGENERATE", "Geometry is degenerate (2D/1D)", metrics)

            axis_dominance_ratio = vars[0] / vars[1]
            
            plane_idx = (1, 2)
            z_idx = 0
            plane_source = "PC1_Axis"
            
            # 如果主轴优势不明显 (接近球形)，尝试寻找最佳投影面
            fallback_ratio = float(self._cfg("axis_dominance_fallback_ratio", 1.15))
            if axis_dominance_ratio < fallback_ratio:
                best_plane, best_cv = self._find_best_plane_fallback(aligned, vars, axis_dominance_ratio)
                if best_plane is None:
                    return self._fail("FAIL_AMBIGUOUS", "Shape ambiguous (Fallback failed)", metrics)
                plane_idx = best_plane
                z_idx = list(set([0, 1, 2]) - set(plane_idx))[0]
                plane_source = "Fallback_Search"
            
            metrics["plane_source"] = plane_source

            # 3. MCD 鲁棒内点选择 (Core Selection)
            pts_2d = aligned[:, plane_idx]
            
            support_fraction = float(self._cfg("mcd_support_fraction", 0.75))
            mcd = MinCovDet(support_fraction=support_fraction, random_state=42).fit(pts_2d)
            dist_sq = mcd.mahalanobis(pts_2d)
            inlier_pct = float(self._cfg("mcd_inlier_percentile", 80))
            threshold = np.percentile(dist_sq, inlier_pct)
            inlier_mask = dist_sq <= threshold
            pts_core = pts_2d[inlier_mask]
            
            metrics["n_atoms_core"] = len(pts_core)
            metrics["inlier_ratio"] = len(pts_core) / len(pts_2d) if len(pts_2d) > 0 else 0.0
            
            min_core = int(self._cfg("min_core_atoms", 15))
            if len(pts_core) < min_core:
                return self._fail("FAIL_UNSTABLE", f"Too few core atoms (<{min_core})", metrics)

            center = np.median(pts_core, axis=0)
            is_small_sample = len(pts_core) < 30 
            
            # 4. 详细指标计算
            core_cov = np.cov(pts_core.T)
            evals, _ = np.linalg.eigh(core_cov)
            min_eval = max(evals[0], 1e-9)
            cov_eigen_ratio = evals[1] / min_eval
            ellipticity = np.sqrt(cov_eigen_ratio)
            
            radii = np.linalg.norm(pts_core - center, axis=1)
            avg_r = np.mean(radii) + 1e-6
            ring_cv = np.std(radii) / avg_r
            
            rad_kurtosis = 0.0 if is_small_sample else self._calc_kurtosis_safe(radii)
            
            z_core = aligned[inlier_mask, z_idx]
            local_thickness = self._calc_local_thickness_spatial(radii, z_core)

            # 4.5 二次曲面拟合 (单叶双曲面) RMSE
            # 采用鲁棒拟合：用核心点拟合参数，用全体点评估误差。
            surface_rmse, surface_rmse_norm = self._fit_hyperboloid_rmse(
                aligned=aligned,
                fit_mask=inlier_mask,
                plane_idx=plane_idx,
                z_idx=z_idx
            )
            
            # 角度覆盖度检查 (Angular Gap)
            dx = pts_core[:, 0] - center[0]
            dy = pts_core[:, 1] - center[1]
            angles = np.degrees(np.arctan2(dy, dx)) % 360
            
            n_bins = 18 if is_small_sample else 36
            hist, _ = np.histogram(angles, bins=n_bins, range=(0, 360))
            hist_doubled = np.concatenate([hist, hist])
            max_zeros = 0
            curr_zeros = 0
            for c in hist_doubled:
                if c == 0: curr_zeros += 1
                else: 
                    max_zeros = max(max_zeros, curr_zeros)
                    curr_zeros = 0
            max_zeros = max(max_zeros, curr_zeros)
            max_zeros = min(max_zeros, n_bins)
            angular_gap = max_zeros * (360.0 / n_bins)
            
            # Z轴轮廓稳定性
            try:
                sort_idx = np.argsort(z_core)
                r_sorted = radii[sort_idx]
                n_chunks = max(2, min(4, len(z_core) // 15))
                chunks = np.array_split(r_sorted, n_chunks)
                chunk_means = [np.mean(c) for c in chunks if len(c) > 0]
                z_profile_cv = np.std(chunk_means)/(np.mean(chunk_means)+1e-6) if len(chunk_means)>1 else 0.0
            except:
                z_profile_cv = 1.0

            metrics.update({
                "ellipticity": ellipticity,
                "cov_eigen_ratio": cov_eigen_ratio,
                "ring_cv": ring_cv,
                "rad_kurtosis": rad_kurtosis,
                "local_thickness": local_thickness,
                "angular_gap": angular_gap,
                "n_angular_bins": n_bins,
                "z_profile_cv": z_profile_cv,
                "surface_rmse": surface_rmse,
                "surface_rmse_norm": surface_rmse_norm
            })

            # 5. 评分与惩罚 (Scoring)
            return self._calculate_score(metrics, is_small_sample, plane_source)

        except Exception as e:
             return self._fail("FAIL_MATH", f"Math Error: {str(e)}", metrics)

    def _calculate_score(self, metrics, is_small_sample, plane_source):
        """应用惩罚项计算最终置信度"""
        ellipticity = metrics["ellipticity"]
        cov_eigen_ratio = metrics["cov_eigen_ratio"]
        local_thickness = metrics["local_thickness"]
        rad_kurtosis = metrics["rad_kurtosis"]
        angular_gap = metrics["angular_gap"]
        z_profile_cv = metrics["z_profile_cv"]
        n_segments = metrics["n_segments"]
        ring_cv = metrics["ring_cv"]
        surface_rmse_norm = metrics.get("surface_rmse_norm", np.nan)

        # 计算惩罚
        # --- penalties configured in validator.yaml ---
        e_thr = float(self._cfg("penalties.ellipticity.thr", 1.8))
        e_mult = float(self._cfg("penalties.ellipticity.mult", 0.6))
        if ellipticity > e_thr:
            metrics["pen_ellipticity"] = (ellipticity - e_thr) * e_mult

        s_thr = float(self._cfg("penalties.stability.cov_eigen_ratio_thr", 50.0))
        s_mult = float(self._cfg("penalties.stability.mult", 0.05))
        s_max = float(self._cfg("penalties.stability.max_penalty", 0.30))
        if cov_eigen_ratio > s_thr:
            metrics["pen_stability"] = min(s_max, s_mult * np.log10(cov_eigen_ratio / s_thr))
        
        # [RESTORED] Ring CV Penalty (严格控制圆度)
        cv_thr = float(self._cfg("penalties.ring_cv.thr_small", 0.20)) if is_small_sample else float(self._cfg("penalties.ring_cv.thr_large", 0.25))
        cv_mult = float(self._cfg("penalties.ring_cv.mult_small", 3.5)) if is_small_sample else float(self._cfg("penalties.ring_cv.mult_large", 2.5))
        if ring_cv > cv_thr: 
            metrics["pen_ringcv"] = (ring_cv - cv_thr) * cv_mult
        
        # [RESTORED] Thickness Penalty (严格控制厚度/空心度)
        thick_thr = float(self._cfg("penalties.thickness.thr_small", 0.25)) if is_small_sample else float(self._cfg("penalties.thickness.thr_large", 0.30))
        thick_mult = float(self._cfg("penalties.thickness.mult_small", 3.0)) if is_small_sample else float(self._cfg("penalties.thickness.mult_large", 2.5))
        if local_thickness > thick_thr: 
            metrics["pen_thickness"] = (local_thickness - thick_thr) * thick_mult
        
        if not is_small_sample:
            k_thick_thr = float(self._cfg("penalties.kurtosis.thick_thr", 0.35))
            k_thr = float(self._cfg("penalties.kurtosis.kurtosis_thr", -1.0))
            k_pen_thick = float(self._cfg("penalties.kurtosis.penalty_if_thick", 0.5))
            k_pen_else = float(self._cfg("penalties.kurtosis.penalty_else", 0.1))
            if local_thickness > k_thick_thr and rad_kurtosis < k_thr:
                metrics["pen_kurtosis"] = k_pen_thick
            elif rad_kurtosis < k_thr:
                metrics["pen_kurtosis"] = k_pen_else
        
        gap_thr = float(self._cfg("penalties.angular_gap.thr_deg", 60.0))
        gap_mult = float(self._cfg("penalties.angular_gap.mult", 0.01))
        if angular_gap > gap_thr:
            metrics["pen_gap"] = (angular_gap - gap_thr) * gap_mult

        z_thr = float(self._cfg("penalties.z_profile.thr_cv", 0.30))
        z_pen = float(self._cfg("penalties.z_profile.penalty", 0.30))
        if z_profile_cv > z_thr:
            metrics["pen_z"] = z_pen
        
        fb_pen = float(self._cfg("penalties.fallback.penalty", 0.15))
        if plane_source != "PC1_Axis":
            metrics["pen_fallback"] = fb_pen

        min_segs = int(self._cfg("penalties.segments.min_segments", 3))
        seg_pen = float(self._cfg("penalties.segments.penalty", 0.25))
        if n_segments < min_segs:
            metrics["pen_segments"] = seg_pen

        # Surface RMSE Penalty (单叶双曲面贴合度)
        # 使用归一化误差避免与绝对尺寸耦合。
        if np.isfinite(surface_rmse_norm):
            rmse_thr = float(self._cfg("penalties.surface_rmse_norm.thr", 0.25))
            rmse_mult = float(self._cfg("penalties.surface_rmse_norm.mult", 1.5))
            if surface_rmse_norm > rmse_thr:
                metrics["pen_surface"] = (surface_rmse_norm - rmse_thr) * rmse_mult

        total_penalty = sum(v for k, v in metrics.items() if k.startswith("pen_"))
        score = max(0.0, 1.0 - total_penalty)
        
        # 判定是否有效 (Threshold > 0.6)
        score_thr = float(self._cfg("score_threshold", 0.60))
        is_valid = score > score_thr

        status = "OK" if is_valid else "FAIL_SCORE"
        issue = "None"
        if not is_valid:
            major = {k: v for k, v in metrics.items() if k.startswith("pen_") and v > 0.15}
            if major:
                max_k = max(major, key=major.get)
                issue = max_k.replace("pen_", "").upper() + "_ISSUE"
            else:
                issue = "LOW_CONFIDENCE"

        return {
            "status": status,
            "is_valid": is_valid,
            "confidence": score, 
            "issue": issue,
            "metrics": metrics
        }

    # --- 辅助计算方法 ---

    def _get_empty_metrics(self):
        return {
            "n_segments": np.nan, "n_atoms_total": np.nan, "n_atoms_core": np.nan,
            "inlier_ratio": np.nan, "n_angular_bins": np.nan, "plane_source": "NA",
            "ellipticity": np.nan, "cov_eigen_ratio": np.nan, "ring_cv": np.nan,
            "rad_kurtosis": np.nan, "local_thickness": np.nan, "angular_gap": np.nan,
            "surface_rmse": np.nan, "surface_rmse_norm": np.nan,
            "z_profile_cv": np.nan, "pen_ellipticity": 0.0, "pen_stability": 0.0,
            "pen_ringcv": 0.0, "pen_thickness": 0.0, "pen_kurtosis": 0.0,
            "pen_gap": 0.0, "pen_z": 0.0, "pen_fallback": 0.0, "pen_segments": 0.0,
            "pen_surface": 0.0
        }

    def _fit_hyperboloid_rmse(self, aligned, fit_mask, plane_idx, z_idx):
        """拟合单叶双曲面: x^2/a^2 + y^2/b^2 - z^2/c^2 = 1

        返回：
          - surface_rmse: 近似点到曲面距离 RMSE (Å)
          - surface_rmse_norm: surface_rmse / mean_radius (dimensionless)

        说明：
          使用隐式曲面距离的一阶近似：d \approx |F(p)| / ||\nabla F(p)||
          其中 F(x,y,z) = x^2/a^2 + y^2/b^2 - z^2/c^2 - 1
        """
        try:
            if aligned is None or len(aligned) < 10:
                return np.nan, np.nan

            # 组织坐标：x,y 为横截面平面，z 为轴向
            x = aligned[:, plane_idx[0]]
            y = aligned[:, plane_idx[1]]
            z = aligned[:, z_idx]

            if fit_mask is None or np.sum(fit_mask) < 10:
                fit_mask = np.ones(len(aligned), dtype=bool)

            xf, yf, zf = x[fit_mask], y[fit_mask], z[fit_mask]

            # 线性最小二乘拟合：alpha x^2 + beta y^2 - gamma z^2 = 1
            # 其中 alpha=1/a^2, beta=1/b^2, gamma=1/c^2
            M = np.stack([xf**2, yf**2, -zf**2], axis=1)
            b = np.ones(len(xf))

            # 若退化/病态，直接返回 NaN
            if M.shape[0] < 3:
                return np.nan, np.nan

            params, *_ = np.linalg.lstsq(M, b, rcond=None)
            params = np.asarray(params, dtype=float)
            # 强制正值，避免出现非物理参数
            params = np.clip(np.abs(params), 1e-8, None)
            alpha, beta, gamma = params.tolist()

            a = 1.0 / np.sqrt(alpha)
            b_ = 1.0 / np.sqrt(beta)
            c = 1.0 / np.sqrt(gamma)

            # 评估所有点的近似距离
            F = alpha * x**2 + beta * y**2 - gamma * z**2 - 1.0
            grad = np.sqrt((2 * alpha * x)**2 + (2 * beta * y)**2 + (-2 * gamma * z)**2) + 1e-9
            dist = np.abs(F) / grad
            rmse = float(np.sqrt(np.mean(dist**2))) if len(dist) > 0 else np.nan

            mean_r = float(0.5 * (a + b_))
            rmse_norm = float(rmse / (mean_r + 1e-9)) if np.isfinite(rmse) else np.nan
            return rmse, rmse_norm
        except Exception:
            return np.nan, np.nan

    def _fail(self, status, reason, metrics):
        return {"status": status, "is_valid": False, "confidence": 0.0, "issue": reason, "metrics": metrics}

    def _calc_kurtosis_safe(self, data):
        if len(data) < 30: return 0.0 
        mean = np.mean(data)
        diff = data - mean
        m4 = np.mean(diff**4)
        m2 = np.mean(diff**2)
        if m2 < 1e-6: return 0.0
        return (m4 / (m2**2)) - 3.0
        
    def _calc_local_thickness_spatial(self, radii, z_coords):
        try:
            z_quantiles = np.percentile(z_coords, [0, 25, 50, 75, 100])
            local_thicknesses = []
            for i in range(4):
                z_min, z_max = z_quantiles[i], z_quantiles[i+1]
                if i < 3: mask = (z_coords >= z_min) & (z_coords < z_max)
                else: mask = (z_coords >= z_min) & (z_coords <= z_max + 1e-9)
                chunk_r = radii[mask]
                if len(chunk_r) < 5: continue
                q75, q25 = np.percentile(chunk_r, [75, 25])
                median = np.median(chunk_r) + 1e-6
                local_thicknesses.append((q75 - q25) / median)
            if not local_thicknesses:
                q75, q25 = np.percentile(radii, [75, 25])
                return (q75 - q25) / (np.median(radii) + 1e-6)
            return np.median(local_thicknesses)
        except: return 1.0

    def _find_best_plane_fallback(self, aligned, all_vars, dom_ratio):
        best_cv = float('inf')
        best_plane = None
        if dom_ratio < 1.03: z_thresh_factor = 0.45
        elif dom_ratio < 1.08: z_thresh_factor = 0.60
        else: z_thresh_factor = 0.75
        for pair in [(1, 2), (0, 2), (0, 1)]:
            z_dim = list(set([0, 1, 2]) - set(pair))[0]
            z_var = all_vars[z_dim]
            plane_max_var = max(all_vars[pair[0]], all_vars[pair[1]])
            if z_var < plane_max_var * z_thresh_factor: continue
            pts = aligned[:, pair]
            try:
                support_fraction = float(self._cfg("mcd_support_fraction", 0.75))
                mcd = MinCovDet(support_fraction=support_fraction, random_state=42).fit(pts)
                dist = mcd.mahalanobis(pts)
                inlier_pct = float(self._cfg("mcd_inlier_percentile", 80))
                thresh = np.percentile(dist, inlier_pct)
                mask = dist <= thresh
                if np.sum(mask) < 10: continue
                center = np.median(pts[mask], axis=0) 
                radii = np.linalg.norm(pts[mask] - center, axis=1)
                cv = np.std(radii) / (np.mean(radii) + 1e-6)
                if cv < best_cv:
                    best_cv = cv
                    best_plane = pair
            except: continue
        if best_cv > 0.5: return None, None 
        return best_plane, best_cv