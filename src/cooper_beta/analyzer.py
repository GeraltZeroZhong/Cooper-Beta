import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


class BarrelAnalyzer:
    """
    分析切片数据：通用椭圆拟合 + score_adjust（忽略 junk slices）
    可选规则：
      - NN（最近邻距离均匀性）：过滤噪声/离散点
      - Angle（角度最大缺口 + seq_order vs angle_order 一致性）：抑制 jelly-roll / beta-sandwich
    """

    def __init__(
        self,
        min_points=6,
        max_rmse=2.5,
        min_axis=4.0,
        max_axis=35.0,
        max_flattening=3.0,
        valid_ratio=0.6,
        lsq_method="trf",
        loss="soft_l1",
        f_scale=1.0,
        min_intersections_for_scoring=4,
        # NN rule
        nn_rule_enabled=True,
        nn_max_robust_cv=0.45,
        nn_min_inlier_frac=0.75,
        nn_fail_as_junk=True,
        # Angle rule
        angle_rule_enabled=True,
        angle_max_gap_deg=90.0,
        angle_order_rule_enabled=True,
        angle_order_local_step_max=2,
        angle_order_min_local_frac=0.60,
        angle_order_max_mean_circ_dist_norm=0.25,
        angle_fail_as_junk=False,
    ):
        self.min_points = int(min_points)
        self.max_rmse = float(max_rmse)
        self.min_axis = float(min_axis)
        self.max_axis = float(max_axis)
        self.max_flattening = float(max_flattening)
        self.valid_ratio = float(valid_ratio)

        # Robust least squares settings
        self.lsq_method = lsq_method
        self.loss = loss
        self.f_scale = float(f_scale)

        # score_adjust settings
        self.min_intersections_for_scoring = int(min_intersections_for_scoring)

        # NN rule settings
        self.nn_rule_enabled = bool(nn_rule_enabled)
        self.nn_max_robust_cv = float(nn_max_robust_cv)
        self.nn_min_inlier_frac = float(nn_min_inlier_frac)
        self.nn_fail_as_junk = bool(nn_fail_as_junk)

        # Angle rule settings
        self.angle_rule_enabled = bool(angle_rule_enabled)
        self.angle_max_gap_deg = float(angle_max_gap_deg)
        self.angle_order_rule_enabled = bool(angle_order_rule_enabled)
        self.angle_order_local_step_max = int(angle_order_local_step_max)
        self.angle_order_min_local_frac = float(angle_order_min_local_frac)
        self.angle_order_max_mean_circ_dist_norm = float(angle_order_max_mean_circ_dist_norm)
        self.angle_fail_as_junk = bool(angle_fail_as_junk)

    def _ellipse_residuals(self, params, x, y):
        """
        通用椭圆残差（隐式方程 - 1）。
        params: (xc, yc, a, b, theta)
        """
        xc, yc, a, b, theta = params

        dx = x - xc
        dy = y - yc

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t

        return (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1.0

    def _robust_center(self, points_xy):
        """
        简单鲁棒中心：用均值作为中心（切片点一般近似环，均值足够）。
        若需要更强鲁棒，可替换为几何中位数。
        """
        pts = np.asarray(points_xy, dtype=float)
        c = np.mean(pts, axis=0)
        return float(c[0]), float(c[1])

    def _nn_spacing_stats(self, points_xy):
        """
        最近邻距离均匀性统计（鲁棒）。
        返回 (median, robust_sigma, robust_cv, inlier_frac)。
        robust_sigma = 1.4826 * MAD
        inlier: |d - median| <= 3 * robust_sigma
        """
        pts = np.asarray(points_xy, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return None

        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        nn = dists[:, 1]

        med = float(np.median(nn))
        mad = float(np.median(np.abs(nn - med)))
        robust_sigma = float(1.4826 * mad)

        if med <= 1e-9:
            robust_cv = float("inf")
        else:
            robust_cv = float(robust_sigma / med)

        if robust_sigma < 1e-12:
            return med, 0.0, 0.0, 1.0

        inliers = np.abs(nn - med) <= (3.0 * robust_sigma)
        inlier_frac = float(np.mean(inliers)) if len(inliers) else 0.0
        return med, robust_sigma, robust_cv, inlier_frac

    def _angular_gap_stats(self, points_xy):
        """角度统计：最大角度缺口（max gap）。

        1) 以中心点计算每个点的半径 r、角度 theta
        2) 半径鲁棒过滤（去掉少量离群点，避免“零星点”填补缺口）
        3) 计算排序后的相邻角度差（含首尾环绕） -> max_gap_deg

        返回：
          (max_gap_deg, coverage_deg, used_n, center_x, center_y)
        或 None（点数不足）。
        """
        pts = np.asarray(points_xy, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 5:
            return None

        cx, cy = self._robust_center(pts)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        r = np.sqrt(dx * dx + dy * dy)

        med_r = float(np.median(r))
        mad_r = float(np.median(np.abs(r - med_r)))
        sigma_r = float(1.4826 * mad_r)

        if sigma_r > 1e-9:
            keep = np.abs(r - med_r) <= (3.0 * sigma_r)
            pts2 = pts[keep]
        else:
            pts2 = pts

        if pts2.shape[0] < 5:
            return None

        dx2 = pts2[:, 0] - cx
        dy2 = pts2[:, 1] - cy
        ang = np.arctan2(dy2, dx2)  # [-pi, pi]
        ang = np.sort(ang)

        # 计算相邻角度差（含环绕）
        diffs = np.diff(ang)
        wrap = (ang[0] + 2.0 * np.pi) - ang[-1]
        diffs = np.concatenate([diffs, [wrap]])

        max_gap = float(np.max(diffs))  # radians
        max_gap_deg = max_gap * 180.0 / np.pi
        coverage_deg = 360.0 - max_gap_deg

        return max_gap_deg, coverage_deg, int(pts2.shape[0]), cx, cy

    @staticmethod
    def _best_circular_affine_fit_cost(angle_pos_by_seq):
        """评估 seq_order 与 angle_order 的全局一致性。

        angle_pos_by_seq: 长度 N 的整数数组，每个元素是该 seq_order 在 angle_order 中的名次 [0..N-1]。

        我们允许：
          - 环状 shift（起点不定）
          - 方向翻转（顺/逆时针）

        返回：best_mean_circ_dist_norm ∈ [0,1]
          - 0：完全一致
          - 1：完全不一致（平均圆周距离接近 N/2）
        """
        a = np.asarray(angle_pos_by_seq, dtype=int)
        n = int(a.size)
        if n <= 1:
            return 0.0

        half = max(1.0, n / 2.0)
        best = float("inf")

        k = np.arange(n, dtype=int)
        for direction in (1, -1):
            base = (direction * k) % n
            for shift in range(n):
                pred = (base + shift) % n
                diff = np.abs(a - pred)
                diff = np.minimum(diff, n - diff)
                cost = float(np.mean(diff)) / half
                if cost < best:
                    best = cost

        # 数值稳定：限定到 [0,1]
        return float(min(1.0, max(0.0, best)))

    def _seq_angle_order_stats(self, points):
        """在同一批交点上同时计算 seq_order 与 angle_order，并给出一致性统计。

        输入 points 期望包含 (x, y, seq_pos)；若缺少 seq_pos 或点数不足则返回 None。

        返回 dict：
          - order_used_n
          - order_local_frac
          - order_mean_step
          - order_max_step
          - order_mean_circ_dist_norm
          - seq_neighbor_dist_median
          - seq_neighbor_dist_robust_cv
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 6 or pts.shape[1] < 3:
            return None

        xy = pts[:, :2]
        seq_pos = pts[:, 2]

        # 与 angle gap 相同的半径鲁棒过滤，避免少量离群点扰乱顺序
        cx, cy = self._robust_center(xy)
        dx = xy[:, 0] - cx
        dy = xy[:, 1] - cy
        r = np.sqrt(dx * dx + dy * dy)

        med_r = float(np.median(r))
        mad_r = float(np.median(np.abs(r - med_r)))
        sigma_r = float(1.4826 * mad_r)
        if sigma_r > 1e-9:
            keep = np.abs(r - med_r) <= (3.0 * sigma_r)
        else:
            keep = np.ones_like(r, dtype=bool)

        if int(np.sum(keep)) < 6:
            return None

        xy2 = xy[keep]
        seq2 = seq_pos[keep]

        n = int(xy2.shape[0])
        # seq_order
        seq_order = np.argsort(seq2)

        # angle_order
        dx2 = xy2[:, 0] - cx
        dy2 = xy2[:, 1] - cy
        ang = (np.arctan2(dy2, dx2) * 180.0 / np.pi) % 360.0
        angle_order = np.argsort(ang)

        pos_in_angle = np.empty(n, dtype=int)
        pos_in_angle[angle_order] = np.arange(n, dtype=int)

        angle_pos_by_seq = pos_in_angle[seq_order]

        # 局部一致性：相邻 seq_order 在 angle_order 上的圆周距离
        steps = []
        for i in range(n - 1):
            d = abs(int(angle_pos_by_seq[i + 1]) - int(angle_pos_by_seq[i]))
            d = min(d, n - d)
            steps.append(d)
        steps = np.asarray(steps, dtype=float)
        local_frac = float(np.mean(steps <= float(self.angle_order_local_step_max))) if steps.size else 1.0
        mean_step = float(np.mean(steps)) if steps.size else 0.0
        max_step = float(np.max(steps)) if steps.size else 0.0

        # 全局一致性（考虑 shift/翻转）
        mean_circ_dist_norm = float(self._best_circular_affine_fit_cost(angle_pos_by_seq))

        # 你最初的想法：按 seq_order 相邻点的欧氏距离分布
        seq_xy = xy2[seq_order]
        dxy = np.diff(seq_xy, axis=0)
        dseq = np.sqrt(np.sum(dxy * dxy, axis=1))
        if dseq.size:
            med = float(np.median(dseq))
            mad = float(np.median(np.abs(dseq - med)))
            robust_sigma = float(1.4826 * mad)
            robust_cv = float(robust_sigma / med) if med > 1e-9 else float("inf")
        else:
            med, robust_cv = 0.0, 0.0

        return {
            "order_used_n": int(n),
            "order_local_frac": float(local_frac),
            "order_mean_step": float(mean_step),
            "order_max_step": float(max_step),
            "order_mean_circ_dist_norm": float(mean_circ_dist_norm),
            "seq_neighbor_dist_median": float(med),
            "seq_neighbor_dist_robust_cv": float(robust_cv),
        }

    def fit_slice(self, points):
        """
        对切片点进行通用椭圆拟合（含旋转）。
        返回 dict 或 None。
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < self.min_points:
            return None

        # points 可能包含额外列（如 seq_pos），拟合只使用前两列 (x,y)
        pts_xy = pts[:, :2]
        x = pts_xy[:, 0]
        y = pts_xy[:, 1]

        # PCA 初始化
        xc0 = float(np.mean(x))
        yc0 = float(np.mean(y))
        centered = pts_xy - np.array([xc0, yc0], dtype=float)

        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)

        a0 = float(2.0 * np.sqrt(max(evals[1], 1e-12)))
        b0 = float(2.0 * np.sqrt(max(evals[0], 1e-12)))

        v_long = evecs[:, 1]
        theta0 = float(np.arctan2(v_long[1], v_long[0]))

        x0 = [xc0, yc0, a0, b0, theta0]

        try:
            res = least_squares(
                self._ellipse_residuals,
                x0,
                args=(x, y),
                method=self.lsq_method,
                loss=self.loss,
                f_scale=self.f_scale,
            )
            xc, yc, a, b, theta = res.x
            a, b = abs(a), abs(b)
            if b > a:
                a, b = b, a

            rmse = float(np.sqrt(np.mean(res.fun ** 2)))
            return {"xc": float(xc), "yc": float(yc), "a": float(a), "b": float(b), "theta": float(theta), "rmse": rmse}
        except Exception:
            return None

    def analyze(self, slices_dict):
        """
        主分析函数。

        score:        valid_count / total_layers（所有 active layers）
        score_adjust: valid_count_scored / total_scored_layers（忽略 junk slices）
            junk slices:
              1) n_points < max(min_intersections_for_scoring, min_points)
              2) NN rule 不通过 且 nn_fail_as_junk=True
              3) Angle rule 不通过 且 angle_fail_as_junk=True
        """
        results = []
        valid_count = 0
        valid_count_scored = 0

        sorted_z = sorted(slices_dict.keys())
        active_layers = [z for z in sorted_z if len(slices_dict[z]) > 0]
        total_layers = len(active_layers)

        if total_layers == 0:
            return {"is_barrel": False, "score": 0.0, "score_adjust": 0.0, "msg": "无有效切片数据"}

        min_score_pts = max(self.min_intersections_for_scoring, self.min_points)
        total_scored_layers = 0

        valid_radii = []

        for z in active_layers:
            points = slices_dict[z]
            npts = len(points)

            # 点数过少 -> junk
            if npts < min_score_pts:
                results.append({"z": float(z), "n_points": npts, "fit": None, "valid": False, "reason": f"JUNK(<{min_score_pts} pts)"})
                continue

            pts_xy = [(p[0], p[1]) for p in points]

            # NN rule
            nn_stats = None
            nn_ok = True
            if self.nn_rule_enabled:
                nn_stats = self._nn_spacing_stats(pts_xy)
                if nn_stats is not None:
                    nn_med, nn_sigma, nn_cv, nn_inlier = nn_stats
                    nn_ok = (nn_cv <= self.nn_max_robust_cv) and (nn_inlier >= self.nn_min_inlier_frac)

            if (not nn_ok) and self.nn_fail_as_junk:
                results.append({
                    "z": float(z),
                    "n_points": npts,
                    "fit": None,
                    "valid": False,
                    "reason": "JUNK(NN irregular)",
                    "nn_median": nn_stats[0] if nn_stats else None,
                    "nn_sigma": nn_stats[1] if nn_stats else None,
                    "nn_cv": nn_stats[2] if nn_stats else None,
                    "nn_inlier_frac": nn_stats[3] if nn_stats else None,
                })
                continue

            # Angle rule：最大角度缺口 + seq_order vs angle_order 一致性
            ang_stats = None
            order_stats = None
            ang_ok = True
            ang_fail_reason = None
            if self.angle_rule_enabled:
                # 1) 角度缺口（几何覆盖）
                ang_stats = self._angular_gap_stats(pts_xy)
                if ang_stats is not None:
                    max_gap_deg, coverage_deg, used_n, cx, cy = ang_stats
                    gap_ok = (max_gap_deg <= self.angle_max_gap_deg)
                else:
                    gap_ok = True

                # 2) 顺序一致性（需要 points 包含 seq_pos）
                order_ok = True
                if self.angle_order_rule_enabled:
                    order_stats = self._seq_angle_order_stats(points)
                    if order_stats is not None:
                        local_ok = (order_stats.get("order_local_frac", 1.0) >= self.angle_order_min_local_frac)
                        global_ok = (order_stats.get("order_mean_circ_dist_norm", 0.0) <= self.angle_order_max_mean_circ_dist_norm)
                        order_ok = local_ok and global_ok
                        if not local_ok:
                            ang_fail_reason = "Seq-angle local mismatch"
                        elif not global_ok:
                            ang_fail_reason = "Seq-angle global mismatch"

                if not gap_ok:
                    ang_fail_reason = "Angle gap too large"

                ang_ok = gap_ok and order_ok

            if (not ang_ok) and self.angle_fail_as_junk:
                results.append({
                    "z": float(z),
                    "n_points": npts,
                    "fit": None,
                    "valid": False,
                    "reason": f"JUNK({ang_fail_reason or 'Angle rule'})",
                    "angle_max_gap_deg": ang_stats[0] if ang_stats else None,
                    "angle_coverage_deg": ang_stats[1] if ang_stats else None,
                    "angle_used_n": ang_stats[2] if ang_stats else None,
                    "angle_center_x": ang_stats[3] if ang_stats else None,
                    "angle_center_y": ang_stats[4] if ang_stats else None,
                    "order_used_n": order_stats.get("order_used_n") if order_stats else None,
                    "order_local_frac": order_stats.get("order_local_frac") if order_stats else None,
                    "order_mean_step": order_stats.get("order_mean_step") if order_stats else None,
                    "order_max_step": order_stats.get("order_max_step") if order_stats else None,
                    "order_mean_circ_dist_norm": order_stats.get("order_mean_circ_dist_norm") if order_stats else None,
                    "seq_neighbor_dist_median": order_stats.get("seq_neighbor_dist_median") if order_stats else None,
                    "seq_neighbor_dist_robust_cv": order_stats.get("seq_neighbor_dist_robust_cv") if order_stats else None,
                })
                continue

            # 进入计分分母（点数足够，且如果设置 fail_as_junk 则已通过相关规则）
            total_scored_layers += 1

            # 若规则失败但不视为 junk：计入分母，直接判该层无效
            if (not nn_ok) and (not self.nn_fail_as_junk):
                results.append({
                    "z": float(z),
                    "n_points": npts,
                    "fit": None,
                    "valid": False,
                    "reason": "NN irregular",
                    "nn_median": nn_stats[0] if nn_stats else None,
                    "nn_sigma": nn_stats[1] if nn_stats else None,
                    "nn_cv": nn_stats[2] if nn_stats else None,
                    "nn_inlier_frac": nn_stats[3] if nn_stats else None,
                    "angle_max_gap_deg": ang_stats[0] if ang_stats else None,
                    "angle_coverage_deg": ang_stats[1] if ang_stats else None,
                    "angle_used_n": ang_stats[2] if ang_stats else None,
                    "angle_center_x": ang_stats[3] if ang_stats else None,
                    "angle_center_y": ang_stats[4] if ang_stats else None,
                    "order_used_n": order_stats.get("order_used_n") if order_stats else None,
                    "order_local_frac": order_stats.get("order_local_frac") if order_stats else None,
                    "order_mean_step": order_stats.get("order_mean_step") if order_stats else None,
                    "order_max_step": order_stats.get("order_max_step") if order_stats else None,
                    "order_mean_circ_dist_norm": order_stats.get("order_mean_circ_dist_norm") if order_stats else None,
                    "seq_neighbor_dist_median": order_stats.get("seq_neighbor_dist_median") if order_stats else None,
                    "seq_neighbor_dist_robust_cv": order_stats.get("seq_neighbor_dist_robust_cv") if order_stats else None,
                })
                continue

            if (not ang_ok) and (not self.angle_fail_as_junk):
                results.append({
                    "z": float(z),
                    "n_points": npts,
                    "fit": None,
                    "valid": False,
                    "reason": ang_fail_reason or "Angle rule failed",
                    "nn_median": nn_stats[0] if nn_stats else None,
                    "nn_sigma": nn_stats[1] if nn_stats else None,
                    "nn_cv": nn_stats[2] if nn_stats else None,
                    "nn_inlier_frac": nn_stats[3] if nn_stats else None,
                    "angle_max_gap_deg": ang_stats[0] if ang_stats else None,
                    "angle_coverage_deg": ang_stats[1] if ang_stats else None,
                    "angle_used_n": ang_stats[2] if ang_stats else None,
                    "angle_center_x": ang_stats[3] if ang_stats else None,
                    "angle_center_y": ang_stats[4] if ang_stats else None,
                    "order_used_n": order_stats.get("order_used_n") if order_stats else None,
                    "order_local_frac": order_stats.get("order_local_frac") if order_stats else None,
                    "order_mean_step": order_stats.get("order_mean_step") if order_stats else None,
                    "order_max_step": order_stats.get("order_max_step") if order_stats else None,
                    "order_mean_circ_dist_norm": order_stats.get("order_mean_circ_dist_norm") if order_stats else None,
                    "seq_neighbor_dist_median": order_stats.get("seq_neighbor_dist_median") if order_stats else None,
                    "seq_neighbor_dist_robust_cv": order_stats.get("seq_neighbor_dist_robust_cv") if order_stats else None,
                })
                continue

            # 椭圆拟合
            fit = self.fit_slice(points)
            if fit is None:
                reason = "Points < Min"
                is_valid = False
            else:
                a, b, rmse = fit["a"], fit["b"], fit["rmse"]
                if rmse > self.max_rmse:
                    reason = f"RMSE > {self.max_rmse}"
                    is_valid = False
                elif a < self.min_axis or b < self.min_axis:
                    reason = "Axis too small"
                    is_valid = False
                elif a > self.max_axis or b > self.max_axis:
                    reason = "Axis too large"
                    is_valid = False
                elif (a / b) > self.max_flattening:
                    reason = "Too flat"
                    is_valid = False
                else:
                    reason = "OK"
                    is_valid = True
                    valid_radii.append((a + b) / 2.0)

            if is_valid:
                valid_count += 1
                valid_count_scored += 1

            results.append({
                "z": float(z),
                "n_points": npts,
                "fit": fit,
                "valid": is_valid,
                "reason": reason,
                "nn_median": nn_stats[0] if nn_stats else None,
                "nn_sigma": nn_stats[1] if nn_stats else None,
                "nn_cv": nn_stats[2] if nn_stats else None,
                "nn_inlier_frac": nn_stats[3] if nn_stats else None,
                "angle_max_gap_deg": ang_stats[0] if ang_stats else None,
                "angle_coverage_deg": ang_stats[1] if ang_stats else None,
                "angle_used_n": ang_stats[2] if ang_stats else None,
                "angle_center_x": ang_stats[3] if ang_stats else None,
                "angle_center_y": ang_stats[4] if ang_stats else None,
                "order_used_n": order_stats.get("order_used_n") if order_stats else None,
                "order_local_frac": order_stats.get("order_local_frac") if order_stats else None,
                "order_mean_step": order_stats.get("order_mean_step") if order_stats else None,
                "order_max_step": order_stats.get("order_max_step") if order_stats else None,
                "order_mean_circ_dist_norm": order_stats.get("order_mean_circ_dist_norm") if order_stats else None,
                "seq_neighbor_dist_median": order_stats.get("seq_neighbor_dist_median") if order_stats else None,
                "seq_neighbor_dist_robust_cv": order_stats.get("seq_neighbor_dist_robust_cv") if order_stats else None,
            })

        score = (valid_count / total_layers) if total_layers else 0.0
        score_adjust = (valid_count_scored / total_scored_layers) if total_scored_layers else 0.0
        avg_radius = float(np.mean(valid_radii)) if valid_radii else 0.0

        return {
            "is_barrel": None,  # 由 main.py 选择 score/score_adjust 后决定
            "score": float(score),
            "score_adjust": float(score_adjust),
            "valid_layers": int(valid_count_scored),
            "total_layers": int(total_layers),
            "total_scored_layers": int(total_scored_layers),
            "avg_radius": float(avg_radius),
            "layer_details": results,
        }
