from collections import defaultdict

import numpy as np

from .constants import (
    DEFAULT_FILL_SHEET_HOLE_LENGTH,
    DEFAULT_SLICE_STEP_SIZE,
    EPSILON,
    TOLERANCE,
)


class ProteinSlicer:
    """
    Slice along the Z axis and collect segment-plane intersections as ``(x, y)``.
    """

    def __init__(
        self,
        step_size=DEFAULT_SLICE_STEP_SIZE,
        fill_sheet_hole_length=DEFAULT_FILL_SHEET_HOLE_LENGTH,
    ):
        self.step_size = float(step_size)
        self.fill_sheet_hole_length = int(fill_sheet_hole_length)

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

    @staticmethod
    def _assign_sheet_runs(sheet_flags):
        """Return per-residue run IDs plus the sequence midpoint of each run."""
        flags = np.asarray(sheet_flags, dtype=bool)
        run_ids = np.full(flags.shape[0], -1, dtype=int)
        run_seq_pos: list[float] = []

        run_id = 0
        index = 0
        while index < flags.shape[0]:
            if not flags[index]:
                index += 1
                continue

            start = index
            while index < flags.shape[0] and flags[index]:
                index += 1
            end = index - 1

            run_ids[start:index] = run_id
            run_seq_pos.append((float(start) + float(end)) / 2.0)
            run_id += 1

        return run_ids, np.asarray(run_seq_pos, dtype=float)

    def slice_structure(self, aligned_coords, residues_data):
        """
        Args:
            aligned_coords: Aligned coordinates with shape ``(N, 3)``.
            residues_data: ``list[dict]`` containing at least ``is_sheet``.

        Returns:
            dict[float, list[tuple]]: ``z_plane -> [(x, y, seq_pos, strand_id), ...]``

        Notes:
            ``seq_pos`` is derived from the midpoint of one contiguous beta-sheet run.
            ``strand_id`` identifies that run. Downstream order checks compare
            circular order at the strand level instead of the raw segment level.
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
        sheet_flags = self._fill_short_holes(
            sheet_flags,
            max_hole_len=self.fill_sheet_hole_length,
        )
        sheet_run_ids, sheet_run_seq_pos = self._assign_sheet_runs(sheet_flags)

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
        # 3) Walk residue segments and compute intersections with each z-plane.
        for i in range(n - 1):
            # Keep only segments that stay inside one contiguous beta-sheet run.
            if not (sheet_flags[i] and sheet_flags[i + 1]):
                continue

            strand_id = int(sheet_run_ids[i])
            if strand_id < 0 or strand_id != int(sheet_run_ids[i + 1]):
                continue

            p1 = aligned_coords[i]
            p2 = aligned_coords[i + 1]

            z1 = float(p1[2])
            z2 = float(p2[2])
            dz = z2 - z1
            if abs(dz) < EPSILON:
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
                if t < -TOLERANCE or t > 1 + TOLERANCE:
                    continue
                x = float(p1[0] + t * (p2[0] - p1[0]))
                y = float(p1[1] + t * (p2[1] - p1[1]))
                seq_pos = float(sheet_run_seq_pos[strand_id])
                slices[z_plane].append((x, y, seq_pos, float(strand_id)))

        return dict(sorted(slices.items()))
