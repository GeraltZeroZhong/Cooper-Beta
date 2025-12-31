import numpy as np


def compute_vector_alt_score(strands):
    """矢量交替性 (Vector Alternation Product).

    Score = (1/n) * [sum_{i=1..n-1} v_i·v_{i+1} + v_n·v_1]
    理想反平行桶 ~ -1.0。
    """
    if not strands:
        return np.nan
    n = len(strands)
    if n < 2:
        return np.nan

    vecs = []
    for s in strands:
        v = np.asarray(s.get("vector", [np.nan, np.nan, np.nan]), dtype=float)
        if v.shape != (3,) or not np.all(np.isfinite(v)):
            return np.nan
        vn = np.linalg.norm(v)
        if vn < 1e-9:
            return np.nan
        vecs.append(v / vn)

    dots = []
    for i in range(n - 1):
        dots.append(float(np.dot(vecs[i], vecs[i + 1])))
    dots.append(float(np.dot(vecs[-1], vecs[0])))
    return float(np.mean(dots))


def _map_strands_to_residue_sets(
    strands,
    all_coords,
    all_resids,
    rotation_matrix,
    centroid,
    nn_dist_tol=0.75,
):
    """将 topology 的 aligned strand coords 反变换回原坐标，并映射到 residue id 集合。"""
    if all_coords is None or all_resids is None or len(all_coords) == 0:
        return None, None, None
    if len(all_resids) != len(all_coords):
        return None, None, None
    if rotation_matrix is None or centroid is None:
        return None, None, None

    try:
        from sklearn.neighbors import KDTree
    except Exception:
        return None, None, None

    R = np.asarray(rotation_matrix, dtype=float)
    c = np.asarray(centroid, dtype=float)
    if R.shape != (3, 3) or c.shape != (3,):
        return None, None, None

    kdt = KDTree(np.asarray(all_coords, dtype=float))

    strand_res_sets = []
    mapped_counts = []
    for s in strands:
        aligned = np.asarray(s.get("coords", []), dtype=float)
        if aligned.ndim != 2 or aligned.shape[1] != 3 or len(aligned) == 0:
            strand_res_sets.append(set())
            mapped_counts.append(0)
            continue

        # inverse of: (x - centroid) @ R.T  =>  x = aligned @ R + centroid
        orig = aligned @ R + c
        dist, idx = kdt.query(orig, k=1)
        dist = dist.reshape(-1)
        idx = idx.reshape(-1)
        keep = dist <= nn_dist_tol
        if not np.any(keep):
            strand_res_sets.append(set())
            mapped_counts.append(0)
            continue

        res = set(int(r) for r in np.asarray(all_resids, dtype=int)[idx[keep]].tolist())
        strand_res_sets.append(res)
        mapped_counts.append(int(np.sum(keep)))

    return strand_res_sets, mapped_counts, kdt


def _graph_is_connected(A):
    """无向图连通性（A 为 bool adjacency）。"""
    if A is None:
        return False
    n = A.shape[0]
    if n == 0:
        return False
    seen = set([0])
    stack = [0]
    while stack:
        u = stack.pop()
        nbrs = np.where(A[u])[0].tolist()
        for v in nbrs:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def _radial_peak_count(r, min_rel_height=0.15):
    """对 r 分布做粗略峰数估计（用于截面单环性/单峰性）。
    返回: (n_peaks, unimodal_score)
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 25:
        return np.nan, np.nan

    # histogram bins: sqrt(N) capped
    n_bins = int(np.clip(np.sqrt(len(r)), 12, 32))
    hist, _ = np.histogram(r, bins=n_bins)
    if np.max(hist) <= 0:
        return np.nan, np.nan

    # smooth: simple [1,2,1] kernel
    h = hist.astype(float)
    h = np.convolve(h, np.array([1.0, 2.0, 1.0]) / 4.0, mode="same")

    peaks = []
    hmax = float(np.max(h))
    thr = min_rel_height * hmax
    for i in range(1, len(h) - 1):
        if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] >= thr:
            peaks.append((i, float(h[i])))

    n_peaks = len(peaks)
    unimodal_score = 1.0 if n_peaks == 1 else 0.0
    return float(n_peaks), float(unimodal_score)


def compute_topology_graph_metrics(
    strands,
    all_coords,
    all_resids,
    dssp_hbonds,
    rotation_matrix,
    centroid,
    energy_cutoff=-0.5,
    min_hbonds_per_edge=2,
    nn_dist_tol=0.75,
    radial_min_rel_height=0.15,
):
    """拓扑/氢键/向量/截面联合检查（返回多指标 dict）。

    基于：
      - DSSP 强氢键图（能量 <= energy_cutoff）
      - 当前 topology 给出的 strand 顺序（1..n）
      - cleaner 对齐坐标（aligned space Z 为桶轴）
    """
    out = {
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
    }

    if not strands:
        return out
    n = len(strands)
    if n < 3:
        return out
    if not dssp_hbonds:
        return out

    strand_res_sets, mapped_counts, _ = _map_strands_to_residue_sets(
        strands=strands,
        all_coords=all_coords,
        all_resids=all_resids,
        rotation_matrix=rotation_matrix,
        centroid=centroid,
        nn_dist_tol=nn_dist_tol,
    )
    if strand_res_sets is None:
        return out

    # 若 residue 映射质量太差，直接返回 NaN 避免误判
    if np.sum(mapped_counts) < max(10, 0.5 * len(all_coords)):
        return out

    # residue -> strand
    resid2strand = {}
    for i, sset in enumerate(strand_res_sets):
        for r in sset:
            resid2strand.setdefault(int(r), i)
    if len(resid2strand) < 10:
        return out

    # counts_dir[i,j] = number of strong hbonds directed i->j
    counts_dir = np.zeros((n, n), dtype=int)
    shifts = {}  # (i,j) -> list of offset (partner - resid)
    for r, partners in dssp_hbonds.items():
        si = resid2strand.get(int(r), None)
        if si is None:
            continue
        for p, e in partners:
            if e > energy_cutoff:
                continue
            sj = resid2strand.get(int(p), None)
            if sj is None or sj == si:
                continue
            counts_dir[si, sj] += 1
            shifts.setdefault((si, sj), []).append(int(p) - int(r))

    counts_und = counts_dir + counts_dir.T
    A = counts_und >= int(min_hbonds_per_edge)

    # required cycle edges based on current strand ordering
    edges = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]

    # --- graph_cyclicity ---
    ok = 0
    for i, j in edges:
        if A[i, j]:
            ok += 1
    out["graph_cyclicity"] = float(ok / len(edges))

    # --- connectivity + degree constraint ---
    deg = np.sum(A, axis=1).astype(float)
    out["degree_mean"] = float(np.mean(deg))
    out["degree_std"] = float(np.std(deg))
    out["degree2_fraction"] = float(np.mean(deg == 2.0))
    out["graph_connected"] = float(1.0 if _graph_is_connected(A) else 0.0)

    # --- edge hbond uniformity (cycle edges) ---
    edge_counts = np.array([counts_und[i, j] for (i, j) in edges], dtype=float)
    if np.all(np.isfinite(edge_counts)) and len(edge_counts) > 0:
        out["edge_hbond_mean"] = float(np.mean(edge_counts))
        out["edge_hbond_std"] = float(np.std(edge_counts))
        out["edge_hbond_min"] = float(np.min(edge_counts))
        out["edge_hbond_max"] = float(np.max(edge_counts))
        m = float(np.mean(edge_counts))
        out["edge_hbond_cv"] = float(np.std(edge_counts) / (m + 1e-9)) if m > 1e-9 else np.nan

    # --- registry offset consistency / shear estimate ---
    edge_mean_shifts = []
    edge_abs_medians = []
    for (i, j) in edges:
        vals = shifts.get((i, j), [])
        if len(vals) < 2:
            # fallback to opposite direction
            vals2 = shifts.get((j, i), [])
            if len(vals2) >= 2:
                vals = [-v for v in vals2]
        if len(vals) >= 2:
            v = np.asarray(vals, dtype=float)
            v = v[np.isfinite(v)]
            if len(v) >= 2:
                edge_mean_shifts.append(float(np.mean(v)))
                edge_abs_medians.append(float(np.median(np.abs(v))))

    if len(edge_mean_shifts) >= 2:
        out["registry_shift_edge_mean_std"] = float(np.std(edge_mean_shifts))
        # sign consistency: compare each edge sign with global sum sign
        s = float(np.sum(edge_mean_shifts))
        if abs(s) > 1e-9:
            sign = 1.0 if s > 0 else -1.0
            same = [1.0 if (m * sign) > 0 else 0.0 for m in edge_mean_shifts]
            out["registry_shift_sign_consistency"] = float(np.mean(same))
        out["shear_number_est"] = float(abs(np.sum(edge_mean_shifts)))

    if len(edge_abs_medians) >= 2:
        out["registry_shift_edge_abs_median_std"] = float(np.std(edge_abs_medians))

    # --- strand tilt consistency (aligned space: Z is barrel axis) ---
    angs = []
    for s in strands:
        v = np.asarray(s.get("vector", [np.nan, np.nan, np.nan]), dtype=float)
        if v.shape != (3,) or not np.all(np.isfinite(v)):
            continue
        vn = np.linalg.norm(v)
        if vn < 1e-9:
            continue
        v = v / vn
        # tilt angle relative to axis: arccos(|v·z|) with z=(0,0,1)
        ang = np.degrees(np.arccos(np.clip(abs(v[2]), 0.0, 1.0)))
        angs.append(float(ang))
    if len(angs) >= 3:
        out["tilt_angle_mean_deg"] = float(np.mean(angs))
        out["tilt_angle_std_deg"] = float(np.std(angs))

    # --- cross-section radial unimodality (stack coords in aligned space) ---
    try:
        coords = np.vstack([np.asarray(s.get("coords", []), dtype=float) for s in strands if len(s.get("coords", [])) > 0])
        if coords.ndim == 2 and coords.shape[1] == 3 and len(coords) >= 25:
            r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
            n_peaks, uni = _radial_peak_count(r, min_rel_height=float(radial_min_rel_height))
            out["radial_n_peaks"] = n_peaks
            out["radial_unimodal_score"] = uni
    except Exception:
        pass

    return out


def compute_graph_cyclicity_score(
    strands,
    all_coords,
    all_resids,
    dssp_hbonds,
    rotation_matrix,
    centroid,
    energy_cutoff=-0.5,
    min_hbonds_per_edge=2,
    nn_dist_tol=0.75,
    radial_min_rel_height=0.15,
):
    """向后兼容：仅返回 graph_cyclicity。"""
    m = compute_topology_graph_metrics(
        strands=strands,
        all_coords=all_coords,
        all_resids=all_resids,
        dssp_hbonds=dssp_hbonds,
        rotation_matrix=rotation_matrix,
        centroid=centroid,
        energy_cutoff=energy_cutoff,
        min_hbonds_per_edge=min_hbonds_per_edge,
        nn_dist_tol=nn_dist_tol,
        radial_min_rel_height=radial_min_rel_height,
    )
    return float(m.get("graph_cyclicity", np.nan))
