import numpy as np
import warnings
from Bio.PDB import PDBParser, DSSP
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet

class BarrelValidator:
    """
    BarrelValidator V14 (Precision Edition)
    
    Changelog V14:
    - Full Precision: Removed all internal rounding. Metrics return raw floats.
    - Status Taxonomy: Distinguishes FAIL_SCORE (calculated but rejected) from FAIL_DSSP/MATH (aborted).
    - Separation of Concerns: _extract_segments now distinguishes execution failure vs empty result.
    - Math Logic: Log-scaled penalty for condition number; Correct denominator for inlier_ratio.
    - Safety: Robust NaN handling in main block.
    """

    def __init__(self, pdb_file, chain_id='A'):
        self.pdb_file = pdb_file
        self.chain_id = chain_id
        self.parser = PDBParser(QUIET=True)
        try:
            self.structure = self.parser.get_structure('protein', pdb_file)
            self.model = self.structure[0]
            self.chain = self.model[chain_id] if chain_id in self.model else None
        except Exception:
            self.chain = None

    def validate(self):
        """
        Returns: {status, is_valid, confidence, issue, metrics}
        """
        metrics = self._get_empty_metrics()
        
        # --- 1. Data Prep ---
        if self.chain is None:
            return self._fail("FAIL_INPUT", "Chain not found", metrics)

        segments, all_coords = self._extract_segments_v14()
        
        # Check for Extraction Failure (None) vs Low Content (Empty List)
        if segments is None:
            return self._fail("FAIL_DSSP_EXEC", "DSSP execution failed", metrics)

        metrics["n_segments"] = len(segments)
        metrics["n_atoms_total"] = len(all_coords)
        
        if len(all_coords) < 30:
            return self._fail("FAIL_LOW_BETA", "Insufficient beta atoms (<30)", metrics)

        # --- 2. PCA & Degeneracy Check ---
        pca = PCA(n_components=3)
        centered_raw = all_coords - np.mean(all_coords, axis=0)
        aligned = pca.fit_transform(centered_raw)
        vars = pca.explained_variance_ 
        
        if vars[1] < 1e-5 or vars[2] < 1e-5: 
            return self._fail("FAIL_DEGENERATE", "Geometry is degenerate (2D/1D)", metrics)

        axis_dominance_ratio = vars[0] / vars[1]
        
        # Selection Strategy
        plane_idx = (1, 2)
        z_idx = 0
        plane_source = "PC1_Axis"
        
        if axis_dominance_ratio < 1.15:
            best_plane, best_cv = self._find_best_plane_fallback(aligned, vars, axis_dominance_ratio)
            if best_plane is None:
                return self._fail("FAIL_AMBIGUOUS", "Shape ambiguous (Fallback failed)", metrics)
            
            plane_idx = best_plane
            z_idx = list(set([0, 1, 2]) - set(plane_idx))[0]
            plane_source = "Fallback_Search"
        
        metrics["plane_source"] = plane_source

        # --- 3. Robust Inlier Selection ---
        pts_2d = aligned[:, plane_idx]
        
        try:
            mcd = MinCovDet(support_fraction=0.75, random_state=42).fit(pts_2d)
            dist_sq = mcd.mahalanobis(pts_2d)
            
            threshold = np.percentile(dist_sq, 80)
            inlier_mask = dist_sq <= threshold
            pts_core = pts_2d[inlier_mask]
            
            # QC Metrics (Fix B: Correct Denominator)
            metrics["n_atoms_core"] = len(pts_core)
            metrics["inlier_ratio"] = len(pts_core) / len(pts_2d) if len(pts_2d) > 0 else 0.0
            
            if len(pts_core) < 15:
                return self._fail("FAIL_UNSTABLE", "Too few core atoms (<15)", metrics)

            center = np.median(pts_core, axis=0)
            is_small_sample = len(pts_core) < 30 
            
            # --- 4. Metric Calculation ---
            
            # A. Stability & Ellipticity
            core_cov = np.cov(pts_core.T)
            evals, _ = np.linalg.eigh(core_cov)
            
            min_eval = max(evals[0], 1e-9)
            cov_eigen_ratio = evals[1] / min_eval
            ellipticity = np.sqrt(cov_eigen_ratio)
            
            # B. Radial Stats
            radii = np.linalg.norm(pts_core - center, axis=1)
            avg_r = np.mean(radii) + 1e-6
            ring_cv = np.std(radii) / avg_r
            
            if is_small_sample:
                rad_kurtosis = 0.0
            else:
                rad_kurtosis = self._calc_kurtosis_safe(radii)
            
            # C. Local Thickness
            z_core = aligned[inlier_mask, z_idx]
            local_thickness = self._calc_local_thickness_spatial(radii, z_core)
            
            # D. Angular Gap
            dx = pts_core[:, 0] - center[0]
            dy = pts_core[:, 1] - center[1]
            angles = np.degrees(np.arctan2(dy, dx)) % 360
            
            n_bins = 18 if is_small_sample else 36
            hist, _ = np.histogram(angles, bins=n_bins, range=(0, 360))
            
            hist_doubled = np.concatenate([hist, hist])
            max_zeros = 0
            current_zeros = 0
            for count in hist_doubled:
                if count == 0:
                    current_zeros += 1
                else:
                    max_zeros = max(max_zeros, current_zeros)
                    current_zeros = 0
            max_zeros = max(max_zeros, current_zeros)
            max_zeros = min(max_zeros, n_bins)
            
            angular_gap = max_zeros * (360.0 / n_bins)
            
            # Update Metrics (No Rounding)
            metrics.update({
                "ellipticity": ellipticity,
                "cov_eigen_ratio": cov_eigen_ratio,
                "ring_cv": ring_cv,
                "rad_kurtosis": rad_kurtosis,
                "local_thickness": local_thickness,
                "angular_gap": angular_gap, # Keep as float
                "n_angular_bins": n_bins
            })

        except Exception as e:
             return self._fail("FAIL_MATH", f"Math Error: {str(e)}", metrics)

        # --- 5. Z-Stability Check ---
        try:
            sort_idx = np.argsort(z_core)
            r_sorted = radii[sort_idx]
            n_chunks = max(2, min(4, len(z_core) // 15))
            chunks = np.array_split(r_sorted, n_chunks)
            chunk_means = [np.mean(c) for c in chunks if len(c) > 0]
            
            if len(chunk_means) > 1:
                z_profile_cv = np.std(chunk_means) / (np.mean(chunk_means) + 1e-6)
            else:
                z_profile_cv = 0.0
            metrics["z_profile_cv"] = z_profile_cv
        except:
            metrics["z_profile_cv"] = 1.0
            z_profile_cv = 1.0

        # --- 6. Explicit Penalty Calculation ---
        
        # 1. Ellipticity
        if ellipticity > 1.8: 
            metrics["pen_ellipticity"] = (ellipticity - 1.8) * 0.6
        
        # 2. Stability (Eigen Ratio) - Log Scale
        if cov_eigen_ratio > 50:
            # Soft log penalty starting at 0.0, capping at 0.3
            metrics["pen_stability"] = min(0.3, 0.05 * np.log10(cov_eigen_ratio / 50))
        
        # 3. Ring CV
        cv_threshold = 0.20 if is_small_sample else 0.25
        cv_penalty_mult = 3.5 if is_small_sample else 2.5
        if ring_cv > cv_threshold: 
            metrics["pen_ringcv"] = (ring_cv - cv_threshold) * cv_penalty_mult
        
        # 4. Thickness
        thick_threshold = 0.25 if is_small_sample else 0.30
        thick_penalty_mult = 3.0 if is_small_sample else 2.5
        if local_thickness > thick_threshold: 
            metrics["pen_thickness"] = (local_thickness - thick_threshold) * thick_penalty_mult
        
        # 5. Kurtosis (Coupled)
        if not is_small_sample:
            is_flat_dist = rad_kurtosis < -1.0
            is_thick = local_thickness > 0.35
            
            if is_thick and is_flat_dist: 
                metrics["pen_kurtosis"] = 0.5 
            elif is_flat_dist: 
                metrics["pen_kurtosis"] = 0.1
        
        # 6. Topology
        if angular_gap > 60: 
            metrics["pen_gap"] = (angular_gap - 60) / 100.0
        
        # 7. Z-Stability
        if z_profile_cv > 0.3: 
            metrics["pen_z"] = 0.3
        
        # 8. Metadata
        if plane_source != "PC1_Axis": 
            metrics["pen_fallback"] = 0.15
        if len(segments) < 3: 
            metrics["pen_segments"] = 0.25

        # Final Score
        total_penalty = sum(v for k, v in metrics.items() if k.startswith("pen_"))
        score = max(0.0, 1.0 - total_penalty)
        is_valid = score > 0.60 

        # Diagnosis
        issue = "None"
        status = "OK"
        
        if not is_valid:
            status = "FAIL_SCORE" # Calculation succeeded, but object is rejected
            
            major_issues = {k: v for k, v in metrics.items() if k.startswith("pen_") and v > 0.15}
            if major_issues:
                max_issue = max(major_issues, key=major_issues.get)
                issue_map = {
                    "pen_ellipticity": "Flat Shape",
                    "pen_stability": "Degenerate Fit",
                    "pen_ringcv": "Irregular Shape",
                    "pen_thickness": "Wall too Thick",
                    "pen_kurtosis": "Bimodal/Sandwich",
                    "pen_gap": "Open Topology",
                    "pen_z": "Unstable Z-Profile",
                    "pen_fallback": "Ambiguous Geometry",
                    "pen_segments": "Too Few Strands"
                }
                issue = issue_map.get(max_issue, "Multiple Issues")
            else:
                issue = "Low Confidence"

        return {
            "status": status,
            "is_valid": is_valid,
            "confidence": score, # Raw float
            "issue": issue,
            "metrics": metrics
        }

    def _get_empty_metrics(self):
        """Schema-compliant NaN initialization"""
        return {
            "n_segments": np.nan,
            "n_atoms_total": np.nan,
            "n_atoms_core": np.nan,
            "inlier_ratio": np.nan,
            "n_angular_bins": np.nan,
            "plane_source": "NA",
            "ellipticity": np.nan,
            "cov_eigen_ratio": np.nan,
            "ring_cv": np.nan,
            "rad_kurtosis": np.nan,
            "local_thickness": np.nan,
            "angular_gap": np.nan,
            "z_profile_cv": np.nan,
            "pen_ellipticity": 0.0,
            "pen_stability": 0.0,
            "pen_ringcv": 0.0,
            "pen_thickness": 0.0,
            "pen_kurtosis": 0.0,
            "pen_gap": 0.0,
            "pen_z": 0.0,
            "pen_fallback": 0.0,
            "pen_segments": 0.0
        }

    def _fail(self, status_code, reason, metrics):
        return {
            "status": status_code,
            "is_valid": False, 
            "confidence": 0.0, 
            "issue": reason, 
            "metrics": metrics
        }

    # ... [Keep helper methods: _calc_local_thickness_spatial, _find_best_plane_fallback, _calc_kurtosis_safe] ...
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
        except:
            return 1.0

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
                mcd = MinCovDet(support_fraction=0.8, random_state=42).fit(pts)
                dist = mcd.mahalanobis(pts)
                thresh = np.percentile(dist, 80)
                mask = dist <= thresh
                if np.sum(mask) < 10: continue
                center = np.median(pts[mask], axis=0) 
                radii = np.linalg.norm(pts[mask] - center, axis=1)
                cv = np.std(radii) / (np.mean(radii) + 1e-6)
                if cv < best_cv:
                    best_cv = cv
                    best_plane = pair
            except:
                continue
        if best_cv > 0.5: return None, None 
        return best_plane, best_cv

    def _calc_kurtosis_safe(self, data):
        if len(data) < 30: return 0.0 
        mean = np.mean(data)
        diff = data - mean
        m4 = np.mean(diff**4)
        m2 = np.mean(diff**2)
        if m2 < 1e-6: return 0.0
        return (m4 / (m2**2)) - 3.0

    def _extract_segments_v14(self):
        """Returns None, None if DSSP execution fails"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                dssp = DSSP(self.model, self.pdb_file)
            except Exception:
                return None, None

        keys = [k for k in dssp.keys() if k[0] == self.chain_id]
        keys.sort(key=lambda k: (k[1][1], k[1][2]))
        segments = []
        current_segment = []
        last_resseq = -999
        JUMP_TOL = 4 
        for k in keys:
            res_id = k[1]
            curr_resseq = res_id[1]
            ss = dssp[k][2]
            if abs(curr_resseq - last_resseq) > JUMP_TOL:
                if len(current_segment) >= 3:
                    segments.append(np.array(current_segment))
                current_segment = []
            if ss == 'E':
                try:
                    atom = self.chain[res_id]['CA']
                    current_segment.append(atom.get_coord())
                    last_resseq = curr_resseq
                except KeyError:
                    pass
        if len(current_segment) >= 3:
            segments.append(np.array(current_segment))
        all_coords = np.concatenate(segments) if segments else np.array([])
        return segments, all_coords

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        v = BarrelValidator(sys.argv[1])
        res = v.validate()
        print(f"Status: {res['status']}")
        print(f"Issue:  {res['issue']}")
        
        # Safe NaN check for Demo
        val = res['metrics']['n_atoms_total']
        if not (isinstance(val, float) and np.isnan(val)):
             print(f"Stats:  N={int(val)}")
             
        # Print non-zero penalties for debug
        print("Penalties:")
        for k, v in res['metrics'].items():
            if k.startswith("pen_") and v > 0:
                print(f"  {k}: {v:.3f}")