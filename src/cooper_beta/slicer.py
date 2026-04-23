import numpy as np
from collections import defaultdict

class ProteinSlicer:
    """
    Slice along the Z axis and collect segment-plane intersections as ``(x, y)``.
    """

    def __init__(self, step_size=1.0):
        self.step_size = float(step_size)

    @staticmethod
    def _fill_short_holes(flags, max_hole_len=1):
        """
        Fill short False runs such as ``True, False, True``.

        Only gaps sandwiched between True regions are filled, and only when the
        gap length is at most ``max_hole_len``.
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
            aligned_coords: Aligned coordinates with shape ``(N, 3)``.
            residues_data: ``list[dict]`` containing at least ``is_sheet``.

        Returns:
            dict[float, list[tuple]]: ``z_plane -> [(x, y, seq_pos), ...]``

        Notes:
            ``seq_pos`` is derived from the residue segment ``(i, i + 1)`` and is
            defined as ``i + 0.5``. It is used later to compare ``seq_order`` and
            ``angle_order`` on the same set of intersections.
        """
        aligned_coords = np.asarray(aligned_coords, dtype=float)
        if aligned_coords.ndim != 2 or aligned_coords.shape[1] != 3:
            raise ValueError("`aligned_coords` must have shape `(N, 3)`.")

        if len(aligned_coords) != len(residues_data):
            raise ValueError("`aligned_coords` and `residues_data` must have the same length.")

        n = len(residues_data)
        if n < 2:
            return {}

        # 1) Preprocess beta-sheet flags by filling one-residue holes.
        sheet_flags = [bool(r.get("is_sheet", False)) for r in residues_data]
        sheet_flags = self._fill_short_holes(sheet_flags, max_hole_len=1)

        # 2) Determine the slice index range. Use integer k to avoid accumulating
        # floating-point error.
        z_coords = aligned_coords[:, 2]
        min_z = float(np.min(z_coords))
        max_z = float(np.max(z_coords))

        step = self.step_size
        if step <= 0:
            raise ValueError("`step_size` must be greater than 0.")

        # Cover all slice planes in [min_z, max_z] with indices k_start..k_end.
        k_start = int(np.ceil(min_z / step))
        k_end = int(np.floor(max_z / step))
        if k_start > k_end:
            return {}

        slices = defaultdict(list)
        eps = 1e-12

        # 3) Walk residue segments and compute intersections with each z-plane.
        for i in range(n - 1):
            # Keep the segment if either endpoint belongs to a beta-sheet region.
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
                # Numerical tolerance: t should stay within [0, 1].
                if t < -1e-9 or t > 1 + 1e-9:
                    continue
                x = float(p1[0] + t * (p2[0] - p1[0]))
                y = float(p1[1] + t * (p2[1] - p1[1]))
                # Sequence position for intersection from segment (i, i+1).
                seq_pos = float(i) + 0.5
                slices[z_plane].append((x, y, seq_pos))

        return dict(sorted(slices.items()))
