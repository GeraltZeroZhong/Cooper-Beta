import numpy as np
from collections import defaultdict

class ProteinSlicer:
    """
    沿 Z 轴切片，并提取 beta 段骨架线段与切平面的交点 (x, y)。
    """

    def __init__(self, step_size=1.0):
        self.step_size = float(step_size)

    @staticmethod
    def _fill_short_holes(flags, max_hole_len=1):
        """
        填平长度 <= max_hole_len 的短空洞：True, False, True -> True, True, True
        仅填“夹在两段 True 中间”的空洞。
        """
        flags = np.asarray(flags, dtype=bool).copy()
        n = len(flags)
        i = 0
        while i < n:
            if flags[i]:
                i += 1
                continue
            j = i
            while j < n and (not flags[j]):
                j += 1
            hole_len = j - i
            if hole_len <= max_hole_len and i > 0 and j < n and flags[i - 1] and flags[j]:
                flags[i:j] = True
            i = j
        return flags

    def slice_structure(self, aligned_coords, residues_data):
        """
        Args:
            aligned_coords: (N,3) 已对齐坐标
            residues_data: list[dict]，至少包含 is_sheet

        Returns:
            dict[float, list[tuple]]: z_plane -> [(x, y, seq_pos), ...]

        说明：
            seq_pos 基于相邻残基线段 (i, i+1) 的序列位置，定义为 i + 0.5。
            该值用于后续在同一批交点上同时得到 seq_order 与 angle_order。
        """
        aligned_coords = np.asarray(aligned_coords, dtype=float)
        if aligned_coords.ndim != 2 or aligned_coords.shape[1] != 3:
            raise ValueError("aligned_coords 必须是 (N,3)")

        if len(aligned_coords) != len(residues_data):
            raise ValueError("aligned_coords 与 residues_data 长度不一致")

        n = len(residues_data)
        if n < 2:
            return {}

        # 1) beta 标记预处理：填平 ≤1 的短空洞
        sheet_flags = [bool(r.get("is_sheet", False)) for r in residues_data]
        sheet_flags = self._fill_short_holes(sheet_flags, max_hole_len=1)

        # 2) 切片索引范围（用整数 k 避免浮点累计误差）
        z_coords = aligned_coords[:, 2]
        min_z = float(np.min(z_coords))
        max_z = float(np.max(z_coords))

        step = self.step_size
        if step <= 0:
            raise ValueError("step_size 必须 > 0")

        # 允许在 [min_z, max_z] 覆盖所有平面：k_start..k_end
        k_start = int(np.ceil(min_z / step))
        k_end = int(np.floor(max_z / step))
        if k_start > k_end:
            return {}

        slices = defaultdict(list)
        eps = 1e-12

        # 3) 遍历相邻残基线段，求与各 z_plane 的交点
        for i in range(n - 1):
            # 改动：至少一个端点属于 beta
            if not (sheet_flags[i] or sheet_flags[i + 1]):
                continue

            p1 = aligned_coords[i]
            p2 = aligned_coords[i + 1]

            z1 = float(p1[2])
            z2 = float(p2[2])
            dz = z2 - z1
            if abs(dz) < eps:
                continue

            seg_min_z = min(z1, z2)
            seg_max_z = max(z1, z2)

            k_min = int(np.ceil(seg_min_z / step))
            k_max = int(np.floor(seg_max_z / step))
            if k_min > k_max:
                continue

            for k in range(k_min, k_max + 1):
                z_plane = k * step
                t = (z_plane - z1) / dz
                # 数值容差：t 应落在 [0,1]
                if t < -1e-9 or t > 1 + 1e-9:
                    continue
                x = float(p1[0] + t * (p2[0] - p1[0]))
                y = float(p1[1] + t * (p2[1] - p1[1]))
                # 交点对应的序列位置：线段 (i, i+1) -> i + 0.5
                seq_pos = float(i) + 0.5
                slices[z_plane].append((x, y, seq_pos))

        return dict(sorted(slices.items()))
