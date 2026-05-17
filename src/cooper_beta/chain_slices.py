from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .alignment import PCAAligner
from .analyzer import BarrelAnalyzer
from .config import AppConfig, build_config, validate_config
from .loader import ProteinLoader
from .models import AnalysisReport, ChainSliceBundle, LayerDiagnostic
from .pipeline_workers import select_alignment_slices
from .slicer import ProteinSlicer


def _resolve_config(
    *,
    config: AppConfig | None = None,
    cfg: AppConfig | None = None,
    overrides: dict[str, object] | list[str] | None = None,
) -> AppConfig:
    if config is not None and cfg is not None:
        raise TypeError("Pass only one of `config` or `cfg`.")
    if overrides is not None and (config is not None or cfg is not None):
        raise TypeError("Pass `overrides` only when Cooper-Beta builds the config for you.")

    resolved_cfg = config or cfg or build_config(overrides)
    validate_config(resolved_cfg)
    return resolved_cfg


def _slice_points_to_tuples(points: list[tuple[float, ...]] | np.ndarray) -> list[tuple[float, ...]]:
    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2:
        return []
    return [tuple(float(value) for value in row) for row in point_array]


def _slice_dict_to_tuples(
    slices: dict[float, list[tuple[float, ...]]],
) -> dict[float, list[tuple[float, ...]]]:
    return {
        float(z_value): _slice_points_to_tuples(points)
        for z_value, points in sorted(slices.items())
    }


def _select_core_slices(
    raw_slices: dict[float, list[tuple[float, ...]]],
    analyzer: BarrelAnalyzer,
) -> dict[float, list[tuple[float, ...]]]:
    core_slices: dict[float, list[tuple[float, ...]]] = {}
    for z_value in sorted(raw_slices):
        points = raw_slices[z_value]
        if not points:
            continue
        selected_points, _, _ = analyzer.select_sequence_core(points)
        core_slices[float(z_value)] = _slice_points_to_tuples(selected_points)
    return core_slices


def _coordinate_array(residues_data: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([residue["coord"] for residue in residues_data], dtype=float)


def _sheet_coordinate_array(residues_data: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [residue["coord"] for residue in residues_data if residue.get("is_sheet", False)],
        dtype=float,
    )


def extract_chain_slices(
    pdb_path: str | Path,
    *,
    chain: str,
    include_raw_axis_slices: bool = False,
    include_core_slices: bool = False,
    include_layer_diagnostics: bool = False,
    config: AppConfig | None = None,
    cfg: AppConfig | None = None,
    overrides: dict[str, object] | list[str] | None = None,
    model_id: int = 0,
    strict_chain: bool = True,
) -> ChainSliceBundle:
    """
    Extract aligned beta-sheet slice data for one chain.

    ``raw_axis_slices`` are the intersections produced after the same PCA axis
    selection used by detection. ``core_slices`` are the analyzer-selected
    sequence cores after same-strand collapse and optional terminal trimming.
    """
    resolved_cfg = _resolve_config(config=config, cfg=cfg, overrides=overrides)
    structure_path = str(Path(pdb_path).expanduser())

    loader = ProteinLoader(
        structure_path,
        model_id=model_id,
        dssp_bin=resolved_cfg.runtime.dssp_bin_path,
        fail_on_dssp_error=resolved_cfg.runtime.fail_on_dssp_error,
        strict_chain=strict_chain,
    )
    residues_data = loader.get_chain_data(chain, strict_chain=strict_chain)

    chain_residue_count = len(residues_data)
    sheet_coordinates = _sheet_coordinate_array(residues_data)
    sheet_residue_count = int(sheet_coordinates.shape[0]) if sheet_coordinates.ndim == 2 else 0

    raw_slices: dict[float, list[tuple[float, ...]]] = {}
    if sheet_residue_count >= 3:
        aligner = PCAAligner()
        aligner.fit(sheet_coordinates)
        slicer = ProteinSlicer(
            step_size=resolved_cfg.slicer.step_size,
            fill_sheet_hole_length=resolved_cfg.slicer.fill_sheet_hole_length,
        )
        raw_slices = _slice_dict_to_tuples(
            select_alignment_slices(
                aligner,
                _coordinate_array(residues_data),
                residues_data,
                slicer,
                resolved_cfg,
            )
        )

    analyzer = None
    core_slices = None
    if include_core_slices:
        analyzer = BarrelAnalyzer(resolved_cfg.analyzer)
        core_slices = _select_core_slices(raw_slices, analyzer)

    analysis_report = None
    layer_diagnostics = None
    if include_layer_diagnostics:
        analyzer = analyzer or BarrelAnalyzer(resolved_cfg.analyzer)
        report = analyzer.analyze(raw_slices)
        analysis_report = AnalysisReport.from_mapping(report)
        layer_diagnostics = [
            LayerDiagnostic.from_mapping(layer)
            for layer in list(report.get("layer_details", []) or [])
        ]

    return ChainSliceBundle(
        structure_path=structure_path,
        filename=Path(structure_path).name,
        chain=chain,
        chain_residues=chain_residue_count,
        sheet_residues=sheet_residue_count,
        informative_slices=len(raw_slices),
        raw_axis_slices=raw_slices if include_raw_axis_slices else None,
        core_slices=core_slices,
        layer_diagnostics=layer_diagnostics,
        analysis_report=analysis_report,
    )
