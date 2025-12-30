import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

class BarrelGeometry:
    """
    BarrelGeometry V23 (Multi-Extract Purify & Safe Rescue)
    
    Final Polish:
    1. Purify: Now extracts MULTIPLE sub-segments from a single raw segment if they 
       belong to different valid clusters (fixes 'dropped half-barrel').
    2. Rescue: Splitters now default to returning [seg] instead of empty list to 
       prevent silent deletion. Rescue uses min_len=3 for finer granularity.
    3. Audit: Precise outlier counting using boolean masks.
    """

    def __init__(self, segments, all_coords, residue_ids=None):
        self.raw_coords = all_coords
        self.residue_ids = residue_ids 
        
        self.audit = {
            'n_splits': 0, 'n_merges': 0, 'n_flips': 0, 
            'n_dropped_short': 0, 'n_linear_splits': 0,
            'n_rejected_helices': 0, 'n_rejected_outliers': 0,
            'status': 'OK', 'parity_method': 'None',
            'rescue_triggered': False,
            'rescue_success': False,
            'keep_ratio': 0.0,
            'dbscan_clusters_kept': 0
        }

        # Configuration
        self.match_cfg = {
            'MAX_DIST': 4.5,
            'MIN_COS': 0.5,
            'MAX_Z_DIFF': 6.0,
            'MAX_THETA_DIFF': 1.2,
            'FLIP_MARGIN': 0.5,
            'R_GATE': 2.0,       
            'MAX_R_DIFF': 3.0    
        }
        
        # 1. Standardization
        if not segments and len(all_coords) > 0:
            initial_segments = self._auto_segment_coords(all_coords, gap_threshold=4.5)
        else:
            initial_segments = segments if segments else []
            if not initial_segments and len(all_coords) > 0:
                initial_segments = self._auto_segment_coords(all_coords, gap_threshold=4.5)

        # 2. THE PURGE (V23: Multi-Extract)
        self.clean_segments, self.clean_coords = self._purify_pipeline_v23(initial_segments)
        
        # Audit Keep Ratio
        n_raw = len(all_coords)
        n_clean = len(self.clean_coords)
        self.audit['keep_ratio'] = round(n_clean / max(1, n_raw), 3)

        self.params = {
            'n_strands': 0, 'shear_S': 0, 
            'radius': 0.0, 'tilt_angle': 0.0, 'height': 0.0
        }
        
        if len(self.clean_coords) > 15:
            self._align_system_robust()
            
            # 3. Topology Refinement
            self._refine_topology_v22(pool_source=self.aligned_segments)
            
            # 4. Under-count Rescue (V23: min_len=3)
            if 0 < len(self.strands) <= 6:
                self._rescue_under_count_v23()
            
            # 5. Physics
            self._calculate_physics()
        else:
            if self.audit['status'] == 'OK':
                self.audit['status'] = 'Insufficient_Points_After_Purge'
        
        self.params.update(self.audit)

    def _auto_segment_coords(self, coords, gap_threshold=4.5):
        segments = []
        if len(coords) == 0: return []
        current_seg = [coords[0]]
        for i in range(1, len(coords)):
            if np.linalg.norm(coords[i] - coords[i-1]) > gap_threshold:
                if len(current_seg) >= 3: 
                    segments.append(np.array(current_seg))
                else:
                    self.audit['n_dropped_short'] += 1
                current_seg = []
            current_seg.append(coords[i])
        if len(current_seg) >= 3: 
            segments.append(np.array(current_seg))
        else:
            self.audit['n_dropped_short'] += 1
        return segments

    def _purify_pipeline_v23(self, initial_segments):
        """
        V23 Update: Multi-Extraction per segment.
        Extracts ALL valid sub-segments belonging to kept clusters, 
        fixing the issue where a segment spanning two clusters lost half its data.
        """
        if not initial_segments: return [], np.array([])
        
        # 1. Split
        splitted_segments = []
        for seg in initial_segments:
            sub_segs = self._scan_and_split(seg, cos_threshold=-0.2)
            if len(sub_segs) > 1: self.audit['n_splits'] += (len(sub_segs) - 1)
            splitted_segments.extend(sub_segs)

        # 2. Filter Helices
        candidates = []
        for seg in splitted_segments:
            if len(seg) < 3: 
                self.audit['n_dropped_short'] += 1
                continue
            
            v_start = seg[-1] - seg[0]
            len_direct = np.linalg.norm(v_start)
            len_path = np.sum([np.linalg.norm(seg[k] - seg[k-1]) for k in range(1, len(seg))])
            linearity = len_direct / (len_path + 1e-6)
            pca = PCA(n_components=min(3, len(seg)))
            pca.fit(seg)
            residual = np.sqrt(np.sum(pca.explained_variance_[1:])) if len(seg) > 3 else 0.0
            
            keep = False
            if linearity > 0.80: keep = True
            elif linearity > 0.60 and residual < 2.5: keep = True
            
            if keep: candidates.append(seg)
            else: self.audit['n_rejected_helices'] += 1
        
        if not candidates: return [], np.array([])
        
        # 3. DBSCAN (Dual Cluster + Multi-Extract)
        all_candidate_coords = np.vstack(candidates)
        try:
            clustering = DBSCAN(eps=6.5, min_samples=4).fit(all_candidate_coords)
            labels = clustering.labels_
            unique_labels = list(set(labels) - {-1})
            
            if not unique_labels: return [], np.array([])
            
            # Select Clusters
            counts = {l: np.sum(labels == l) for l in unique_labels}
            sorted_labels = sorted(unique_labels, key=lambda l: counts[l], reverse=True)
            
            valid_labels = {sorted_labels[0]}
            self.audit['dbscan_clusters_kept'] = 1
            if len(sorted_labels) > 1:
                l1, l2 = sorted_labels[0], sorted_labels[1]
                if counts[l2] >= 0.4 * counts[l1]:
                    valid_labels.add(l2)
                    self.audit['dbscan_clusters_kept'] = 2

            final_segments = []
            cursor = 0
            for seg in candidates:
                seg_len = len(seg)
                seg_labels = labels[cursor : cursor + seg_len]
                cursor += seg_len
                
                # V23 Fix: Allow extracting multiple sub-segments from different clusters
                kept_mask = np.zeros(seg_len, dtype=bool)
                new_subsegs = []

                for lab in valid_labels:
                    idx = np.where(seg_labels == lab)[0]
                    if len(idx) < 3: continue
                    
                    split_locs = np.where(np.diff(idx) != 1)[0] + 1
                    groups = np.split(idx, split_locs)
                    # We only take the best group *for this label* to keep it simple,
                    # but we do it for EACH valid label.
                    g = max(groups, key=len)
                    
                    if len(g) < 3: continue
                    
                    s, e = int(g[0]), int(g[-1])
                    new_subsegs.append(seg[s:e+1])
                    kept_mask[s:e+1] = True
                
                if not new_subsegs:
                    self.audit['n_rejected_outliers'] += seg_len
                    continue
                
                # Correct outlier counting
                self.audit['n_rejected_outliers'] += int(seg_len - kept_mask.sum())
                final_segments.extend(new_subsegs)
                
            final_coords = np.vstack(final_segments) if final_segments else np.array([])
            return final_segments, final_coords

        except Exception:
            self.audit['status'] = 'DBSCAN_Fallback'
            return candidates, all_candidate_coords

    def _endpoint_anchor(self, seg, which='head', w=4):
        w = min(w, len(seg))
        pts = seg[:w] if which == 'head' else seg[-w:]
        if len(pts) == 0: return seg[0]
        
        r = np.linalg.norm(pts[:, :2], axis=1)
        gate = self.match_cfg.get('R_GATE', 2.0)
        good_indices = np.where(r > gate)[0]
        
        if len(good_indices) > 0:
            return np.mean(pts[good_indices], axis=0)
        return np.mean(pts, axis=0)

    def _align_system_robust(self):
        self.centroid = np.mean(self.clean_coords, axis=0)
        centered = self.clean_coords - self.centroid
        pca = PCA(n_components=3)
        pca.fit(centered)
        basis = pca.components_
        vars = pca.explained_variance_
        candidates = []
        eps = 1e-9
        for i in range(3):
            others = [k for k in range(3) if k != i]
            v1, v2 = vars[others[0]], vars[others[1]]
            metric = abs(np.log((v1 + eps) / (v2 + eps)))
            candidates.append((i, metric))
        candidates.sort(key=lambda x: x[1])
        best_axis_idx = candidates[0][0]
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

    def _refine_topology_v22(self, pool_source, cfg=None):
        active_pool = []
        for i, seg in enumerate(pool_source):
            if len(seg) < 3: continue
            head_anchor = self._endpoint_anchor(seg, 'head', w=4)
            tail_anchor = self._endpoint_anchor(seg, 'tail', w=4)
            active_pool.append({
                'coords': seg,
                'head': head_anchor,
                'tail': tail_anchor,
                'vec': self._get_strand_vector(seg),
                'used': False,
                'id': i
            })

        merged_strands = []
        while True:
            seed = None
            for s in active_pool:
                if not s['used']:
                    seed = s
                    break
            if not seed: break
            seed['used'] = True
            
            curr = seed
            fwd_chain = []
            while True:
                match = self._find_best_match_v20(curr, active_pool, direction='forward', cfg=cfg)
                if match:
                    cand, is_flip = match
                    if is_flip: self._flip_segment(cand)
                    cand['used'] = True
                    fwd_chain.append(cand)
                    curr = cand
                    self.audit['n_merges'] += 1
                else: break
            
            curr = seed
            bwd_chain = []
            while True:
                match = self._find_best_match_v20(curr, active_pool, direction='backward', cfg=cfg)
                if match:
                    cand, is_flip = match
                    if is_flip: self._flip_segment(cand)
                    cand['used'] = True
                    bwd_chain.append(cand)
                    curr = cand
                    self.audit['n_merges'] += 1
                else: break
            
            full_chain = bwd_chain[::-1] + [seed] + fwd_chain
            combined_coords = np.vstack([x['coords'] for x in full_chain])
            merged_strands.append({
                'coords': combined_coords,
                'vector': self._get_strand_vector(combined_coords)
            })

        self.strands = merged_strands
        self._decompose_complex_strands_safe()
        if len(self.strands) % 2 != 0:
            self._fix_parity()

    def _find_best_match_v20(self, curr, pool, direction, cfg=None):
        cfg = self.match_cfg if cfg is None else cfg
        MAX_DIST = cfg['MAX_DIST']
        MIN_COS = cfg['MIN_COS']
        MAX_Z_DIFF = cfg['MAX_Z_DIFF']
        MAX_THETA_DIFF = cfg['MAX_THETA_DIFF']
        FLIP_MARGIN = cfg['FLIP_MARGIN']
        R_GATE = cfg['R_GATE']
        MAX_R_DIFF = cfg['MAX_R_DIFF']

        if direction == 'forward':
            ref_point = curr['tail']
            ref_vec = curr['vec']
        else: 
            ref_point = curr['head']
            ref_vec = curr['vec'] 
            
        best_match = None
        best_cost = 999.0
        ref_r = np.linalg.norm(ref_point[:2])
        use_theta = ref_r > R_GATE
        ref_theta = np.arctan2(ref_point[1], ref_point[0]) if use_theta else 0.0

        for cand in pool:
            if cand['used']: continue
            if direction == 'forward':
                pt_norm, pt_flip = cand['head'], cand['tail']
            else:
                pt_norm, pt_flip = cand['tail'], cand['head']
            
            d_norm = np.linalg.norm(ref_point - pt_norm)
            d_flip = np.linalg.norm(ref_point - pt_flip)
            is_flip = False
            if d_flip < d_norm - FLIP_MARGIN:
                is_flip = True
                d = d_flip
                chosen_pt = pt_flip
                chosen_vec = -cand['vec']
            else:
                d = d_norm
                chosen_pt = pt_norm
                chosen_vec = cand['vec']

            if d > MAX_DIST: continue
            if np.dot(ref_vec, chosen_vec) < MIN_COS: continue 
            if abs(ref_point[2] - chosen_pt[2]) > MAX_Z_DIFF: continue
            cand_r = np.linalg.norm(chosen_pt[:2])
            if abs(ref_r - cand_r) > MAX_R_DIFF: continue

            theta_cost = 0.0
            if use_theta and cand_r > R_GATE:
                cand_theta = np.arctan2(chosen_pt[1], chosen_pt[0])
                diff = abs(ref_theta - cand_theta)
                diff = min(diff, 2*np.pi - diff)
                if diff > MAX_THETA_DIFF: continue
                theta_cost = diff

            cost = d + 5.0 * (1.0 - np.dot(ref_vec, chosen_vec)) + 0.5 * theta_cost
            if cost < best_cost:
                best_cost = cost
                best_match = (cand, is_flip)
        return best_match

    def _flip_segment(self, cand_obj):
        cand_obj['coords'] = cand_obj['coords'][::-1]
        cand_obj['head'], cand_obj['tail'] = cand_obj['tail'], cand_obj['head']
        cand_obj['vec'] = -cand_obj['vec']
        self.audit['n_flips'] += 1

    def _rescue_under_count_v23(self):
        """V23 Rescue: Uses min_len=3 for deep split."""
        self.audit['rescue_triggered'] = True
        n_before = len(self.strands)
        refined_pool = []
        for seg in self.aligned_segments:
            # V23: Pass min_len=3 for granular split
            refined_pool.extend(self._deep_split_under_count(seg, min_len=3))

        strict_cfg = self.match_cfg.copy()
        strict_cfg.update({
            'MAX_DIST': 4.0,       
            'MIN_COS': 0.65,       
            'MAX_Z_DIFF': 5.0,     
            'MAX_THETA_DIFF': 0.8, 
            'FLIP_MARGIN': 0.8,
            'MAX_R_DIFF': 2.0      
        })

        self.strands = []
        self._refine_topology_v22(pool_source=refined_pool, cfg=strict_cfg)
        n_after = len(self.strands)
        is_success = (n_after >= 8) or (n_after >= n_before + 2)
        if is_success:
            self.audit['rescue_success'] = True
            if len(refined_pool) > len(self.aligned_segments):
                self.audit['n_splits'] += (len(refined_pool) - len(self.aligned_segments))

    # --- Split Helpers (Safe V23) ---
    def _split_by_z_turns(self, seg, min_len=4, dz_eps=0.2):
        if len(seg) < 2 * min_len: return [seg]
        z = seg[:, 2].copy()
        z_s = z.copy()
        z_s[1:-1] = (z[:-2] + z[1:-1] + z[2:]) / 3.0
        dz = np.diff(z_s)
        dz[np.abs(dz) < dz_eps] = 0.0
        s = np.sign(dz)
        for i in range(1, len(s)):
            if s[i] == 0: s[i] = s[i-1]
        for i in range(len(s)-2, -1, -1):
            if s[i] == 0: s[i] = s[i+1]
        changes = np.where(s[:-1] * s[1:] < 0)[0] + 1
        cuts = [c for c in changes if c >= min_len and (len(seg) - c) >= min_len]
        if not cuts: return [seg]
        out, start = [], 0
        for c in cuts:
            out.append(seg[start:c])
            start = c
        out.append(seg[start:])
        
        # V23: Fallback safe
        out = [x for x in out if len(x) >= min_len]
        return out if out else [seg]

    def _split_by_theta_span(self, seg, min_len=4, r_min=2.0, span_thr=1.3):
        if len(seg) < 2 * min_len: return [seg]
        r = np.linalg.norm(seg[:, :2], axis=1)
        valid = r > r_min
        if np.sum(valid) < 2 * min_len: return [seg]
        theta = np.arctan2(seg[valid, 1], seg[valid, 0])
        theta = np.unwrap(theta)
        span = float(theta.max() - theta.min())
        if span < span_thr: return [seg]
        dtheta = np.abs(np.diff(theta))
        k = int(np.argmax(dtheta))
        valid_idx = np.where(valid)[0]
        cut = int(valid_idx[k] + 1)
        if cut < min_len or (len(seg) - cut) < min_len: return [seg]
        return [seg[:cut], seg[cut:]]

    def _deep_split_under_count(self, seg, min_len=4):
        # V23: Accepts min_len
        parts = self._scan_and_split(seg, cos_threshold=0.2)
        out = []
        for p in parts:
            for p2 in self._split_by_z_turns(p, min_len=min_len):
                for p3 in self._split_by_theta_span(p2, min_len=min_len):
                    out.append(p3)
        # Safety filter, but logic above usually guarantees seg return if failed
        return [x for x in out if len(x) >= 3]

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
                if linearity < 0.75:
                    p1, p2 = coords[0], coords[-1]
                    vec_line = p2 - p1
                    len_line = np.linalg.norm(vec_line)
                    if len_line < 0.1: split_idx = len(coords) // 2
                    else:
                        vec_unit = vec_line / len_line
                        max_dist, split_idx = -1.0, -1
                        for k in range(min_len, len(coords) - min_len):
                            v_point = coords[k] - p1
                            t = np.dot(v_point, vec_unit)
                            closest = p1 + np.clip(t, 0, len_line) * vec_unit
                            dist = np.linalg.norm(coords[k] - closest)
                            if dist > max_dist: max_dist, split_idx = dist, k
                    if split_idx != -1 and max_dist > 2.0: 
                        s1 = coords[:split_idx]
                        s2 = coords[split_idx:]
                        if len(s1) >= min_len and len(s2) >= min_len:
                            new_strands.append({'coords': s1, 'vector': self._get_strand_vector(s1)})
                            new_strands.append({'coords': s2, 'vector': self._get_strand_vector(s2)})
                            any_split = True
                            self.audit['n_linear_splits'] += 1
                        else: new_strands.append(s)
                    else: new_strands.append(s)
                else: new_strands.append(s)
            self.strands = new_strands
            if not any_split: break

    def _scan_and_split(self, seg, cos_threshold=-0.2):
        min_seg_len = 4
        if len(seg) < min_seg_len * 2: return [seg]
        seg_smooth = seg.copy()
        for k in range(3):
            seg_smooth[1:-1, k] = (seg[:-2, k] + seg[1:-1, k] + seg[2:, k]) / 3.0
        window = 3
        limit = len(seg) - 2 * window
        best_idx, min_cos = -1, 1.0
        for i in range(0, limit):
            p0, p1, p2 = seg_smooth[i], seg_smooth[i+window], seg_smooth[i+2*window]
            v1, v2 = p1-p0, p2-p1
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 0.1 or n2 < 0.1: continue
            cos_val = np.dot(v1, v2) / (n1 * n2)
            if cos_val < cos_threshold:
                if cos_val < min_cos:
                    cand = i + window
                    if cand >= min_seg_len and (len(seg)-cand) >= min_seg_len:
                        min_cos, best_idx = cos_val, cand
        if best_idx != -1:
            return self._scan_and_split(seg[:best_idx], cos_threshold) + \
                   self._scan_and_split(seg[best_idx:], cos_threshold)
        return [seg]

    def _fix_parity(self):
        """V22 Logic: Lock Small N."""
        n = len(self.strands)
        if n % 2 == 0: return

        if n <= 8:
            lens = [len(s['coords']) for s in self.strands]
            idx = int(np.argmax(lens))
            coords = self.strands[idx]['coords']
            min_len = 3 

            if len(coords) >= 2 * min_len:
                p1, p2 = coords[0], coords[-1]
                v = p2 - p1
                L = np.linalg.norm(v)
                split_idx = len(coords) // 2
                
                if L > 1e-6:
                    u = v / L
                    best_k, best_dev = -1, -1.0
                    for k in range(min_len, len(coords) - min_len):
                        t = np.dot(coords[k] - p1, u)
                        closest = p1 + np.clip(t, 0, L) * u
                        dev = np.linalg.norm(coords[k] - closest)
                        if dev > best_dev:
                            best_dev, best_k = dev, k
                    if best_k != -1:
                        split_idx = best_k

                if split_idx >= min_len and (len(coords) - split_idx) >= min_len:
                    s1 = {'coords': coords[:split_idx], 'vector': self._get_strand_vector(coords[:split_idx])}
                    s2 = {'coords': coords[split_idx:], 'vector': self._get_strand_vector(coords[split_idx:])}
                    self.strands.pop(idx)
                    self.strands.insert(idx, s2)
                    self.strands.insert(idx, s1)
                    self.audit['parity_method'] = f'GeoSplit_SmallN_{idx}'
                    return
            
            self.audit['parity_method'] = 'KeepOdd_SmallN'
            return

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
        cand_split = scores[0]
        cand_drop = sorted(scores, key=lambda x: x['len'])[0] 
        
        if cand_split['linearity'] < 0.85 and cand_split['len'] > 8:
            idx = cand_split['idx']
            coords = self.strands[idx]['coords']
            mid = len(coords) // 2
            s1 = {'coords': coords[:mid], 'vector': self._get_strand_vector(coords[:mid])}
            s2 = {'coords': coords[mid:], 'vector': self._get_strand_vector(coords[mid:])}
            self.strands.pop(idx)
            self.strands.insert(idx, s2)
            self.strands.insert(idx, s1)
            self.audit['parity_method'] = f'Split_Strand_{idx}'
        else:
            idx = cand_drop['idx']
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
        a, b = 4.4, 3.3
        delta_sq = (2*np.pi*R)**2 - (n*a)**2
        if delta_sq > -15.0: S_calc = np.sqrt(max(0.0, delta_sq)) / b
        else: S_calc = (n * a / b) * min(np.tan(np.radians(avg_tilt)), 5.0)
        z_coords = all_local[:, 2]
        height = np.max(z_coords) - np.min(z_coords) if len(z_coords) > 0 else 0.0
        self.params.update({
            "n_strands": n,
            "radius": round(R, 2),
            "tilt_angle": round(avg_tilt, 1),
            "shear_S_raw": round(S_calc, 2),
            "shear_S": int(np.floor(S_calc / 2 + 0.5)) * 2,
            "height": round(height, 1)
        })

    def get_summary(self):
        return self.params