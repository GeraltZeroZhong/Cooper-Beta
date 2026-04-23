import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


class BarrelAnalyzer:
    """
    Analyze slice data with a general ellipse fit plus score adjustment.

    Optional rules:
      - NN rule: nearest-neighbor spacing uniformity for filtering noisy or
        scattered intersections.
      - Angle rule: angular coverage plus sequence-order vs angle-order
        consistency to suppress jelly-roll / beta-sandwich cases.
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
        General ellipse residuals using the implicit equation minus 1.

        params: ``(xc, yc, a, b, theta)``
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
        Simple robust center estimate based on the mean.

        Slice intersections are usually close to a ring, so the mean is
        sufficient here. If stronger robustness is needed, this can be replaced
        with a geometric median.
        """
        pts = np.asarray(points_xy, dtype=float)
        c = np.mean(pts, axis=0)
        return float(c[0]), float(c[1])

    def _nn_spacing_stats(self, points_xy):
        """
        Robust nearest-neighbor spacing statistics.

        Returns ``(median, robust_sigma, robust_cv, inlier_frac)`` where:
        ``robust_sigma = 1.4826 * MAD`` and
        ``inlier = |d - median| <= 3 * robust_sigma``.
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
        """Compute angular coverage statistics, including the maximum gap.

        1) Compute radius ``r`` and angle ``theta`` around the slice center.
        2) Apply a robust radial filter to remove a few outliers so isolated
           points do not artificially close a gap.
        3) Compute adjacent angular differences, including wrap-around, to get
           ``max_gap_deg``.

        Returns:
          ``(max_gap_deg, coverage_deg, used_n, center_x, center_y)``
        or ``None`` when too few points remain.
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

        # Compute adjacent angular differences, including wrap-around.
        diffs = np.diff(ang)
        wrap = (ang[0] + 2.0 * np.pi) - ang[-1]
        diffs = np.concatenate([diffs, [wrap]])

        max_gap = float(np.max(diffs))  # radians
        max_gap_deg = max_gap * 180.0 / np.pi
        coverage_deg = 360.0 - max_gap_deg

        return max_gap_deg, coverage_deg, int(pts2.shape[0]), cx, cy

    @staticmethod
    def _best_circular_affine_fit_cost(angle_pos_by_seq):
        """Evaluate global consistency between ``seq_order`` and ``angle_order``.

        ``angle_pos_by_seq`` is an integer array of length ``N``. Each element is
        the rank of that ``seq_order`` item inside ``angle_order``, in ``[0, N-1]``.

        Allowed transformations:
          - Circular shift (arbitrary starting point)
          - Direction reversal (clockwise / counter-clockwise)

        Returns ``best_mean_circ_dist_norm`` in ``[0, 1]``:
          - ``0`` means perfectly consistent
          - ``1`` means maximally inconsistent
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

        # Numerical safety: clamp to [0, 1].
        return float(min(1.0, max(0.0, best)))

    def _seq_angle_order_stats(self, points):
        """Compute sequence-order and angle-order consistency on one slice.

        ``points`` is expected to contain ``(x, y, seq_pos)``. Returns ``None``
        if ``seq_pos`` is missing or too few points are available.

        Returns a dict containing:
          - ``order_used_n``
          - ``order_local_frac``
          - ``order_mean_step``
          - ``order_max_step``
          - ``order_mean_circ_dist_norm``
          - ``seq_neighbor_dist_median``
          - ``seq_neighbor_dist_robust_cv``
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 6 or pts.shape[1] < 3:
            return None

        xy = pts[:, :2]
        seq_pos = pts[:, 2]

        # Use the same radial filtering as the angle-gap calculation so a few
        # outliers do not scramble the order statistics.
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

        # Local consistency: circular distance between adjacent seq_order items
        # in angle_order rank space.
        steps = []
        for i in range(n - 1):
            d = abs(int(angle_pos_by_seq[i + 1]) - int(angle_pos_by_seq[i]))
            d = min(d, n - d)
            steps.append(d)
        steps = np.asarray(steps, dtype=float)
        local_frac = float(np.mean(steps <= float(self.angle_order_local_step_max))) if steps.size else 1.0
        mean_step = float(np.mean(steps)) if steps.size else 0.0
        max_step = float(np.max(steps)) if steps.size else 0.0

        # Global consistency with shift and direction reversal allowed.
        mean_circ_dist_norm = float(self._best_circular_affine_fit_cost(angle_pos_by_seq))

        # Additional diagnostic: Euclidean spacing between adjacent seq_order
        # points on the slice.
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
        Fit a rotated general ellipse to the slice intersections.

        Returns a result dict or ``None``.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < self.min_points:
            return None

        # points may include extra columns such as seq_pos; fit only uses x and y.
        pts_xy = pts[:, :2]
        x = pts_xy[:, 0]
        y = pts_xy[:, 1]

        # PCA-based initialization.
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
        Main slice-analysis entry point.

        ``score`` is ``valid_count / total_layers`` across all active slices.
        ``score_adjust`` is ``valid_count_scored / total_scored_layers`` after
        excluding junk slices.

        A slice is considered junk when:
          1) ``n_points < max(min_intersections_for_scoring, min_points)``
          2) The NN rule fails and ``nn_fail_as_junk=True``
          3) The angle rule fails and ``angle_fail_as_junk=True``
        """
        results = []
        valid_count = 0
        valid_count_scored = 0

        sorted_z = sorted(slices_dict.keys())
        active_layers = [z for z in sorted_z if len(slices_dict[z]) > 0]
        total_layers = len(active_layers)

        if total_layers == 0:
            return {"is_barrel": False, "score": 0.0, "score_adjust": 0.0, "msg": "No active slices were found."}

        min_score_pts = max(self.min_intersections_for_scoring, self.min_points)
        total_scored_layers = 0

        valid_radii = []

        for z in active_layers:
            points = slices_dict[z]
            npts = len(points)

            # Too few intersections: exclude this slice from scoring.
            if npts < min_score_pts:
                results.append({
                    "z": float(z),
                    "n_points": npts,
                    "fit": None,
                    "valid": False,
                    "reason": f"JUNK(too few intersections: need >= {min_score_pts})",
                })
                continue

            pts_xy = [(p[0], p[1]) for p in points]

            # Nearest-neighbor spacing rule.
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
                    "reason": "JUNK(irregular nearest-neighbor spacing)",
                    "nn_median": nn_stats[0] if nn_stats else None,
                    "nn_sigma": nn_stats[1] if nn_stats else None,
                    "nn_cv": nn_stats[2] if nn_stats else None,
                    "nn_inlier_frac": nn_stats[3] if nn_stats else None,
                })
                continue

            # Angle rule: geometric coverage plus seq_order vs angle_order consistency.
            ang_stats = None
            order_stats = None
            ang_ok = True
            ang_fail_reason = None
            if self.angle_rule_enabled:
                # 1) Angular gap / geometric coverage.
                ang_stats = self._angular_gap_stats(pts_xy)
                if ang_stats is not None:
                    max_gap_deg, coverage_deg, used_n, cx, cy = ang_stats
                    gap_ok = (max_gap_deg <= self.angle_max_gap_deg)
                else:
                    gap_ok = True

                # 2) Ordering consistency, which requires seq_pos in the points.
                order_ok = True
                if self.angle_order_rule_enabled:
                    order_stats = self._seq_angle_order_stats(points)
                    if order_stats is not None:
                        local_ok = (order_stats.get("order_local_frac", 1.0) >= self.angle_order_min_local_frac)
                        global_ok = (order_stats.get("order_mean_circ_dist_norm", 0.0) <= self.angle_order_max_mean_circ_dist_norm)
                        order_ok = local_ok and global_ok
                        if not local_ok:
                            ang_fail_reason = "Local sequence/angle order mismatch"
                        elif not global_ok:
                            ang_fail_reason = "Global sequence/angle order mismatch"

                if not gap_ok:
                    ang_fail_reason = "Angular gap is too large"

                ang_ok = gap_ok and order_ok

            if (not ang_ok) and self.angle_fail_as_junk:
                results.append({
                    "z": float(z),
                    "n_points": npts,
                    "fit": None,
                    "valid": False,
                    "reason": f"JUNK({ang_fail_reason or 'Angle rule failed'})",
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

            # Include the slice in the scoring denominator. At this point it has
            # enough points and has already passed any rule marked as junk-on-fail.
            total_scored_layers += 1

            # If a rule fails but is not treated as junk, keep the slice in the
            # denominator and mark it invalid.
            if (not nn_ok) and (not self.nn_fail_as_junk):
                results.append({
                    "z": float(z),
                    "n_points": npts,
                    "fit": None,
                    "valid": False,
                    "reason": "Nearest-neighbor spacing is irregular",
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

            # Ellipse fitting.
            fit = self.fit_slice(points)
            if fit is None:
                reason = "Ellipse fit failed"
                is_valid = False
            else:
                a, b, rmse = fit["a"], fit["b"], fit["rmse"]
                if rmse > self.max_rmse:
                    reason = f"Ellipse fit RMSE exceeds {self.max_rmse}"
                    is_valid = False
                elif a < self.min_axis or b < self.min_axis:
                    reason = "Ellipse axis is below the minimum threshold"
                    is_valid = False
                elif a > self.max_axis or b > self.max_axis:
                    reason = "Ellipse axis exceeds the maximum threshold"
                    is_valid = False
                elif (a / b) > self.max_flattening:
                    reason = "Slice is too flattened"
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
            "is_barrel": None,  # Determined later by the pipeline using score or score_adjust.
            "score": float(score),
            "score_adjust": float(score_adjust),
            "valid_layers": int(valid_count_scored),
            "total_layers": int(total_layers),
            "total_scored_layers": int(total_scored_layers),
            "avg_radius": float(avg_radius),
            "layer_details": results,
        }
