import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

class BarrelGeometry:
    """
    BarrelGeometry V15 (Final Production)
    
    Refinements:
    1. Full Audit: 'n_dropped_short' now tracks drops from ALL phases (Init, Split, Re-segment).
    2. Scale Sync: Re-segmentation after DBSCAN uses a relaxed gap threshold (6.0A) 
       to match DBSCAN's eps (6.5A), preventing artificial fragmentation of 
       connected densities.
    3. Architecture: Retains V14's safe 'Split -> Filter -> DBSCAN' pipeline.
    """

    def __init__(self, segments, all_coords, residue_ids=None):
        self.raw_coords = all_coords
        self.residue_ids = residue_ids 
        
        self.audit = {
            'n_splits': 0, 'n_merges': 0, 'n_flips': 0, 
            'n_dropped_short': 0, 'n_linear_splits': 0,
            'n_rejected_helices': 0, 'n_rejected_outliers': 0,
            'status': 'OK', 'parity_method': 'None'
        }
        
        # 1. Standardization (Auto-segment with strict 4.5A physics)
        if not segments and len(all_coords) > 0:
            initial_segments = self._auto_segment_coords(all_coords, gap_threshold=4.5)
        else:
            initial_segments = segments if segments else []
            if not initial_segments and len(all_coords) > 0:
                initial_segments = self._auto_segment_coords(all_coords, gap_threshold=4.5)

        # 2. THE PURGE
        self.clean_segments, self.clean_coords = self._purify_pipeline_v15(initial_segments)
        
        self.params = {
            'n_strands': 0, 'shear_S': 0, 
            'radius': 0.0, 'tilt_angle': 0.0, 'height': 0.0
        }
        
        if len(self.clean_coords) > 15:
            self._align_system_robust()
            self._refine_topology_v15()
            self._calculate_physics()
        else:
            self.audit['status'] = 'Insufficient_Points_After_Purge'
        
        self.params.update(self.audit)

    def _auto_segment_coords(self, coords, gap_threshold=4.5):
        """
        Standard gap-based segmentation.
        Includes built-in audit for dropped short segments.
        """
        segments = []
        if len(coords) == 0: return []
        current_seg = [coords[0]]
        for i in range(1, len(coords)):
            # Use customizable threshold (4.5 for physics, 6.0 for DBSCAN cleanup)
            if np.linalg.norm(coords[i] - coords[i-1]) > gap_threshold:
                if len(current_seg) >= 3: 
                    segments.append(np.array(current_seg))
                else:
                    self.audit['n_dropped_short'] += 1 # Audit here
                current_seg = []
            current_seg.append(coords[i])
        
        # Flush last
        if len(current_seg) >= 3: 
            segments.append(np.array(current_seg))
        else:
            self.audit['n_dropped_short'] += 1 # Audit here
            
        return segments

    def _purify_pipeline_v15(self, initial_segments):
        """
        Pipeline: Split (Kinks) -> Filter (Helices) -> DBSCAN (Noise).
        """
        if not initial_segments: return [], np.array([])

        # --- Phase 1: Pre-Filter Splitting ---
        splitted_segments = []
        for seg in initial_segments:
            sub_segs = self._scan_and_split(seg)
            if len(sub_segs) > 1: self.audit['n_splits'] += (len(sub_segs) - 1)
            splitted_segments.extend(sub_segs)

        # --- Phase 2: Geometry Filter (Helix Removal) ---
        candidates = []
        for seg in splitted_segments:
            # Audit: Catch fragments made too short by splitting
            if len(seg) < 3: 
                self.audit['n_dropped_short'] += 1
                continue
            
            v_start = seg[-1] - seg[0]
            len_direct = np.linalg.norm(v_start)
            len_path = np.sum([np.linalg.norm(seg[k] - seg[k-1]) for k in range(1, len(seg))])
            linearity = len_direct / (len_path + 1e-6)
            
            # Threshold 0.60: Keep curved strands, reject helices (<0.5)
            if linearity > 0.60:
                candidates.append(seg)
            else:
                self.audit['n_rejected_helices'] += 1
        
        if not candidates: return [], np.array([])
        
        # --- Phase 3: Spatial DBSCAN (Segment-Aware) ---
        all_candidate_coords = np.vstack(candidates)
        
        try:
            # eps=6.5 matches the looseness of beta-sheet connectivity
            clustering = DBSCAN(eps=6.5, min_samples=4).fit(all_candidate_coords)
            labels = clustering.labels_
            
            unique_labels = set(labels)
            if -1 in unique_labels: unique_labels.remove(-1)
            if not unique_labels: return [], np.array([])
            
            largest_label = max(unique_labels, key=lambda l: np.sum(labels == l))
            
            final_segments = []
            cursor = 0
            kept_count = 0
            
            for seg in candidates:
                seg_len = len(seg)
                seg_labels = labels[cursor : cursor + seg_len]
                cursor += seg_len
                
                # Filter atoms: Keep only those in the main cluster
                mask = (seg_labels == largest_label)
                
                # Drop segment if mostly noise (<3 atoms left)
                if np.sum(mask) < 3: 
                    self.audit['n_rejected_outliers'] += seg_len # Count atoms lost
                    continue 
                
                clean_seg = seg[mask]
                kept_count += len(clean_seg)
                self.audit['n_rejected_outliers'] += (seg_len - len(clean_seg))
                
                # Re-segmentation: Use relaxed gap (6.0) to match DBSCAN eps (6.5)
                # This prevents splitting strands that DBSCAN thought were connected.
                internal_sub_segs = self._auto_segment_coords(clean_seg, gap_threshold=6.0)
                final_segments.extend(internal_sub_segs)
            
            if final_segments:
                final_coords = np.vstack(final_segments)
            else:
                final_coords = np.array([])
                
            return final_segments, final_coords

        except Exception as e:
            print(f"Warning: DBSCAN failed ({e}), using helix-filtered data.")
            return candidates, all_candidate_coords

    def _align_system_robust(self):
        """Align Barrel Axis to Z."""
        self.centroid = np.mean(self.clean_coords, axis=0)
        centered = self.clean_coords - self.centroid
        
        pca = PCA(n_components=3)
        pca.fit(centered)
        vars = pca.explained_variance_
        basis = pca.components_
        
        eps = 1e-9
        candidates = []
        for i in range(3):
            others = [k for k in range(3) if k != i]
            v1 = vars[others[0]]
            v2 = vars[others[1]]
            metric = abs(np.log((v1 + eps) / (v2 + eps)))
            candidates.append((i, metric, vars[i]))
        
        candidates.sort(key=lambda x: x[1])
        best = candidates[0]
        
        if len(candidates) > 1 and abs(candidates[1][1] - best[1]) < 0.02:
            if candidates[1][2] > best[2]:
                best = candidates[1]
                
        best_axis_idx = best[0]
        z_vec = basis[best_axis_idx]
        others = [k for k in range(3) if k != best_axis_idx]
        x_vec = basis[others[0]]
        
        z_vec /= np.linalg.norm(z_vec)
        x_vec /= np.linalg.norm(x_vec)
        x_vec = x_vec - np.dot(x_vec, z_vec) * z_vec
        x_vec /= np.linalg.norm(x_vec)
        y_vec = np.cross(z_vec, x_vec)
        
        self.rotation_matrix = np.vstack([x_vec, y_vec, z_vec])
        
        self.aligned_segments = []
        for seg in self.clean_segments:
            seg_local = (seg - self.centroid) @ self.rotation_matrix.T 
            self.aligned_segments.append(seg_local)

    def _refine_topology_v15(self):
        """Merge Logic + Decomposition + Parity."""
        merged = []
        pool = [s for s in self.aligned_segments if len(s) >= 3]
        
        if pool:
            merged = [pool[0]]
            for i in range(1, len(pool)):
                prev = merged[-1]
                curr = pool[i]
                
                dist_direct = np.linalg.norm(curr[0] - prev[-1])
                dist_flip = np.linalg.norm(curr[-1] - prev[-1])
                
                target_seg = curr
                dist = dist_direct
                
                # Flip Check
                if (dist_flip < dist_direct - 0.5) and (dist_flip < 6.0):
                    target_seg = curr[::-1]
                    dist = dist_flip
                    self.audit['n_flips'] += 1
                
                v1 = self._get_strand_vector(prev)
                v2 = self._get_strand_vector(target_seg)
                cos_sim = np.dot(v1, v2)
                
                should_merge = False
                if dist < 4.0 and cos_sim > -0.2: should_merge = True
                elif dist < 6.0 and cos_sim > 0.7: should_merge = True
                
                if should_merge:
                    merged[-1] = np.vstack((prev, target_seg))
                    self.audit['n_merges'] += 1
                else:
                    merged.append(target_seg)
        else:
            merged = pool # Fix: Handle single/empty pool case gracefully
        
        self.strands = []
        for m in merged:
            if len(m) >= 3:
                self.strands.append({
                    'coords': m,
                    'vector': self._get_strand_vector(m)
                })

        # Global Decomposition
        self._decompose_complex_strands_safe()
        
        # Parity
        if len(self.strands) % 2 != 0:
            self._fix_parity()

    def _decompose_complex_strands_safe(self):
        max_passes = 3
        min_len = 4
        
        for _ in range(max_passes):
            new_strands = []
            any_split = False
            for s in self.strands:
                coords = s['coords']
                if len(coords) < min_len * 2: 
                    new_strands.append(s)
                    continue

                v_start = coords[-1] - coords[0]
                len_direct = np.linalg.norm(v_start)
                len_path = np.sum([np.linalg.norm(coords[k] - coords[k-1]) for k in range(1, len(coords))])
                linearity = len_direct / (len_path + 1e-6)
                
                if linearity < 0.70:
                    p1, p2 = coords[0], coords[-1]
                    vec_line = p2 - p1
                    len_line = np.linalg.norm(vec_line)
                    
                    if len_line < 0.1: 
                        split_idx = len(coords) // 2
                    else:
                        vec_unit = vec_line / len_line
                        max_dist, split_idx = -1.0, -1
                        for k in range(min_len, len(coords) - min_len):
                            v_point = coords[k] - p1
                            t = np.dot(v_point, vec_unit)
                            t_clamped = np.clip(t, 0, len_line)
                            closest_pt_on_seg = p1 + t_clamped * vec_unit
                            dist = np.linalg.norm(coords[k] - closest_pt_on_seg)
                            
                            if dist > max_dist:
                                max_dist = dist
                                split_idx = k
                    
                    if split_idx != -1:
                        s1_c = coords[:split_idx]
                        s2_c = coords[split_idx:]
                        if len(s1_c) >= min_len and len(s2_c) >= min_len:
                            new_strands.append({'coords': s1_c, 'vector': self._get_strand_vector(s1_c)})
                            new_strands.append({'coords': s2_c, 'vector': self._get_strand_vector(s2_c)})
                            any_split = True
                            self.audit['n_linear_splits'] += 1
                        else:
                            new_strands.append(s)
                    else:
                        new_strands.append(s)
                else:
                    new_strands.append(s)
            self.strands = new_strands
            if not any_split: break

    def _scan_and_split(self, seg):
        min_seg_len = 4
        if len(seg) < min_seg_len * 2: return [seg]
        
        seg_smooth = seg.copy()
        for k in range(3):
            seg_smooth[1:-1, k] = (seg[:-2, k] + seg[1:-1, k] + seg[2:, k]) / 3.0
        
        window_size = 3
        limit = len(seg) - 2 * window_size
        best_split_idx = -1
        min_cos = 1.0
        
        for i in range(0, limit):
            p0 = seg_smooth[i]
            p1 = seg_smooth[i + window_size]
            p2 = seg_smooth[i + 2 * window_size]
            v1, v2 = p1 - p0, p2 - p1
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 0.1 or n2 < 0.1: continue
            
            cos_val = np.dot(v1, v2) / (n1 * n2)
            if cos_val < -0.2:
                if cos_val < min_cos:
                    split_cand = i + window_size
                    if split_cand >= min_seg_len and (len(seg) - split_cand) >= min_seg_len:
                        min_cos = cos_val
                        best_split_idx = split_cand
        
        if best_split_idx != -1:
            part1 = seg[:best_split_idx]
            part2 = seg[best_split_idx:]
            return self._scan_and_split(part1) + self._scan_and_split(part2)
        return [seg]

    def _fix_parity(self):
        scores = []
        for i, s in enumerate(self.strands):
            c = s['coords']
            v_start = c[-1] - c[0]
            len_direct = np.linalg.norm(v_start)
            len_arc = np.sum([np.linalg.norm(c[k] - c[k-1]) for k in range(1, len(c))])
            linearity = len_direct / (len_arc + 1e-6)
            scores.append({'idx': i, 'linearity': linearity, 'len': len(c)})
        
        if not scores: return
        scores.sort(key=lambda x: x['linearity'])
        candidate_split, candidate_drop = scores[0], sorted(scores, key=lambda x: x['len'])[0] 
        
        if candidate_split['linearity'] < 0.85 and candidate_split['len'] > 8:
            idx = candidate_split['idx']
            coords = self.strands[idx]['coords']
            mid = len(coords) // 2
            s1 = {'coords': coords[:mid], 'vector': self._get_strand_vector(coords[:mid])}
            s2 = {'coords': coords[mid:], 'vector': self._get_strand_vector(coords[mid:])}
            self.strands.pop(idx)
            self.strands.insert(idx, s2)
            self.strands.insert(idx, s1)
            self.audit['parity_method'] = f'Split_Strand_{idx}'
        else:
            idx = candidate_drop['idx']
            self.strands.pop(idx)
            self.audit['parity_method'] = f'Drop_Strand_{idx}'

    def _get_strand_vector(self, coords):
        if len(coords) < 3: return np.array([0,0,1])
        c = coords - np.mean(coords, axis=0)
        uu, ss, vh = np.linalg.svd(c)
        vec = vh[0]
        if np.dot(vec, coords[-1]-coords[0]) < 0: vec = -vec
        return vec

    def _calculate_physics(self):
        if not self.strands: return
        n = len(self.strands)
        all_local = np.vstack([s['coords'] for s in self.strands])
        dists = np.linalg.norm(all_local[:, :2], axis=1)
        R = np.median(dists)
        
        tilts = []
        for s in self.strands:
            vec = s['vector']
            cos_theta = abs(vec[2])
            angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
            tilts.append(angle)
        
        avg_tilt = np.clip(np.mean(tilts), 0, 85.0) if tilts else 0.0
        
        a = 4.4; b = 3.3
        circum = 2 * np.pi * R
        width = n * a
        
        delta_sq = circum**2 - width**2
        tol_neg = -15.0
        
        if delta_sq > tol_neg:
            S_calc = np.sqrt(max(0.0, delta_sq)) / b
        else:
            tilt_rad = np.radians(avg_tilt)
            t_val = np.tan(tilt_rad)
            t_val = min(t_val, 5.0) 
            S_calc = (n * a / b) * t_val

        S_int = int(np.floor(S_calc / 2 + 0.5)) * 2
        z_coords = all_local[:, 2]
        height = np.max(z_coords) - np.min(z_coords) if len(z_coords) > 0 else 0.0

        self.params.update({
            "n_strands": n,
            "radius": round(R, 2),
            "tilt_angle": round(avg_tilt, 1),
            "shear_S_raw": round(S_calc, 2),
            "shear_S": S_int,
            "height": round(height, 1)
        })

    def get_summary(self):
        return self.params