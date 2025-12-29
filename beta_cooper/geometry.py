import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class BarrelGeometry:
    """
    BarrelGeometry V5 (Gap Hunter)
    
    Final Logic:
    1. Distance Splitting: Splits segments if CA-CA distance > 4.5A (Physical break).
       This catches hairpins where 'main.py' skipped non-beta residues but kept the segment.
    2. Monotonicity Splitting: Splits if Z-direction reverses (V-shape).
    3. Smart Merge: Re-connects only if aligned and close.
    """

    def __init__(self, segments, all_coords, residue_ids=None):
        self.all_coords = all_coords
        self.residue_ids = residue_ids
        
        # Handle empty/raw input
        if not segments and len(all_coords) > 0:
            self.input_segments = self._auto_segment_coords(all_coords)
            self.cloud_coords = all_coords
        else:
            self.input_segments = segments
            if segments:
                self.cloud_coords = np.vstack(segments)
            else:
                self.cloud_coords = all_coords
        
        self.params = {
            'n_strands': 0, 'shear_S': 0, 'shear_S_raw': 0.0, 
            'tilt_angle': 0.0, 'radius': 0.0, 'height': 0.0
        }

        if len(self.cloud_coords) > 10:
            self._align_system_robust()
            self._refine_topology()
            self._calculate_physics()

    def _auto_segment_coords(self, coords):
        """Raw coordinate segmentation based on distance."""
        segments = []
        if len(coords) == 0: return []
        current_seg = [coords[0]]
        for i in range(1, len(coords)):
            if np.linalg.norm(coords[i] - coords[i-1]) > 4.5:
                if len(current_seg) >= 3: segments.append(np.array(current_seg))
                current_seg = []
            current_seg.append(coords[i])
        if len(current_seg) >= 3: segments.append(np.array(current_seg))
        return segments

    def _align_system_robust(self):
        """Align Barrel Axis to Z."""
        if len(self.cloud_coords) < 10: return
        self.centroid = np.mean(self.cloud_coords, axis=0)
        centered = self.cloud_coords - self.centroid
        
        pca = PCA(n_components=3)
        pca.fit(centered)
        basis = pca.components_ 
        vars = pca.explained_variance_
        
        best_axis_idx = 0
        best_circularity = float('inf')
        for i in range(3):
            others = [k for k in range(3) if k != i]
            ratio = vars[others[0]] / vars[others[1]]
            metric = abs(1.0 - ratio)
            if vars[i] < 0.2 * max(vars): metric += 10.0
            if metric < best_circularity:
                best_circularity = metric
                best_axis_idx = i
        
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
        for seg in self.input_segments:
            seg_local = (seg - self.centroid) @ self.rotation_matrix.T 
            self.aligned_segments.append(seg_local)
            
    def _refine_topology(self):
        """
        1. Gap Split: Physical breaks > 4.5A
        2. Turn Split: Z-direction flip
        3. Merge: Collinear cleanup
        """
        # Phase 1: Hard Split on Gaps & Turns
        split_pool = []
        
        for seg in self.aligned_segments:
            if len(seg) < 3: continue
            
            # Smooth Z for turn detection
            z_vals = seg[:, 2]
            z_smooth = np.convolve(z_vals, np.ones(3)/3, mode='valid')
            z_smooth = np.concatenate([[z_vals[0]], z_smooth, [z_vals[-1]]])
            
            current_poly = [seg[0]]
            current_trend = 0 # 1: Up, -1: Down
            
            for i in range(1, len(seg)):
                # CRITICAL FIX: Physical Distance Check
                # Even if main.py passed it as one segment, if we removed residues, 
                # the distance jump will betray the gap.
                dist = np.linalg.norm(seg[i] - seg[i-1])
                is_gap = dist > 4.5
                
                # Turn Logic
                dz = z_smooth[i] - z_smooth[i-1]
                step_trend = 1 if dz > 0 else -1
                is_turn = False
                
                if current_trend != 0 and step_trend != current_trend:
                    if len(current_poly) >= 4:
                        # Look ahead confirmation
                        future_idx = min(i+2, len(z_smooth)-1)
                        future_dz = z_smooth[future_idx] - z_smooth[i]
                        if (step_trend == 1 and future_dz > 0.5) or (step_trend == -1 and future_dz < -0.5):
                            is_turn = True

                # CUT if Gap OR Turn
                if is_gap or is_turn:
                    if len(current_poly) >= 3:
                        split_pool.append(np.array(current_poly))
                    current_poly = []
                    current_trend = step_trend
                    # If it was a gap, start fresh. If turn, overlap slightly? 
                    # For gaps, no overlap.
                    if is_turn and not is_gap:
                        current_poly = [seg[i-1], seg[i]]
                    else:
                        current_poly = [seg[i]]
                else:
                    current_poly.append(seg[i])
                    if current_trend == 0 and len(current_poly) >= 2:
                        current_trend = step_trend
            
            if len(current_poly) >= 3:
                split_pool.append(np.array(current_poly))

        if not split_pool: split_pool = self.aligned_segments

        # Phase 2: Merge Collinear (Cleanup)
        # Re-connect ONLY if aligned and close
        merged = [split_pool[0]]
        
        for i in range(1, len(split_pool)):
            prev = merged[-1]
            curr = split_pool[i]
            
            if len(prev) < 3 or len(curr) < 3:
                merged.append(curr)
                continue

            dist = np.linalg.norm(curr[0] - prev[-1])
            vec_prev = self._get_strand_vector(prev)
            vec_curr = self._get_strand_vector(curr)
            cos_sim = np.dot(vec_prev, vec_curr)
            
            # Merge Criteria:
            # 1. Very close (broken density)
            # 2. Somewhat close AND aligned (missing loop)
            should_merge = False
            if dist < 4.0: 
                should_merge = True
            elif dist < 6.0 and cos_sim > 0.9: 
                should_merge = True
                
            if should_merge:
                new_coords = np.vstack((prev, curr))
                merged[-1] = new_coords
            else:
                merged.append(curr)
        
        # Final Strand List
        self.strands = []
        for coords in merged:
            if len(coords) < 4: continue
            vec = self._get_strand_vector(coords)
            direction = 1 if vec[2] > 0 else -1
            self.strands.append({
                'coords': coords,
                'vector': vec,
                'direction': direction
            })

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
        avg_tilt = np.mean(tilts) if tilts else 0.0
        
        # Murzin Formula
        # Using 4.4 / 3.3 for Barrel Geometry
        a = 4.4; b = 3.3
        tilt_rad = np.radians(avg_tilt)
        S_float = (n * a / b) * np.tan(tilt_rad)
        S_int = round(S_float / 2) * 2
        
        z_coords = all_local[:, 2]
        height = np.max(z_coords) - np.min(z_coords)

        self.params = {
            "n_strands": n,
            "radius": round(R, 2),
            "tilt_angle": round(avg_tilt, 1),
            "shear_S_raw": round(S_float, 2),
            "shear_S": int(S_int),
            "height": round(height, 1)
        }

    def unroll(self):
        data = []
        for i, s in enumerate(self.strands):
            c = s['coords']
            center = np.mean(c, axis=0)
            theta = np.arctan2(center[1], center[0])
            data.append({"strand_idx": i, "theta": theta, "z_center": center[2]})
        return pd.DataFrame(data)

    def get_summary(self):
        return self.params