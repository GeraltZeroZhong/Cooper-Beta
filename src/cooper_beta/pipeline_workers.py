from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .alignment import PCAAligner
from .analyzer import BarrelAnalyzer
from .config import AppConfig
from .loader import ProteinLoader
from .prepare_cache import load_prepare_payloads, store_prepare_payloads
from .slicer import ProteinSlicer

try:
    from tqdm.auto import tqdm

    tqdm_write = tqdm.write
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

    def tqdm_write(message: str) -> None:
        print(message)


def analyze_chain_payload(payload: dict[str, object], cfg: AppConfig) -> dict[str, object]:
    """
    Analyze one chain payload and return the final row written to the results table.
    """
    filename = str(payload.get("filename", ""))
    chain_id = str(payload.get("chain", ""))
    residues_data = list(payload.get("residues_data", []) or [])
    min_chain_residues = cfg.input.min_chain_residues
    min_sheet_residues = cfg.input.min_sheet_residues
    min_informative_slices = cfg.input.min_informative_slices

    def build_row(
        result: str,
        reason: str,
        report: dict[str, object] | None = None,
    ) -> dict[str, object]:
        report = report or {}
        return {
            "filename": filename,
            "chain": chain_id,
            "result": result,
            "score_adjust": float(report.get("score_adjust", 0.0)),
            "valid_layers": int(report.get("valid_layers", 0)),
            "all_adjusted_layers": int(report.get("total_scored_layers", 0)),
            "all_layers": int(report.get("total_layers", 0)),
            "reason": reason,
        }

    if len(residues_data) < min_chain_residues:
        return build_row("SKIP", f"Chain has fewer than {min_chain_residues} residues")

    all_coordinates = np.array([residue["coord"] for residue in residues_data], dtype=float)
    sheet_coordinates = np.array(
        [residue["coord"] for residue in residues_data if residue.get("is_sheet", False)],
        dtype=float,
    )
    if sheet_coordinates.shape[0] < min_sheet_residues:
        return build_row("SKIP", "Not enough beta-sheet residues")

    try:
        aligner = PCAAligner()
        aligner.fit(sheet_coordinates)
        aligned_coordinates = aligner.transform(all_coordinates)
    except Exception:
        return build_row("ERROR", "Alignment failed")

    slicer = ProteinSlicer(
        step_size=cfg.slicer.step_size,
        fill_sheet_hole_length=cfg.slicer.fill_sheet_hole_length,
    )
    slices = slicer.slice_structure(aligned_coordinates, residues_data)
    if len(slices) < min_informative_slices:
        return build_row("SKIP", "Too few informative slices")

    analyzer = BarrelAnalyzer(cfg.analyzer)
    try:
        report = analyzer.analyze(slices)
    except Exception:
        fallback = {"total_layers": len([z_value for z_value in slices if slices[z_value]])}
        return build_row("ERROR", "Slice analyzer crashed unexpectedly", fallback)

    score_raw = float(report.get("score", 0.0))
    score_adjust = float(report.get("score_adjust", 0.0))
    decision_cfg = cfg.analyzer.decision
    final_score = score_adjust if decision_cfg.use_adjusted_score else score_raw

    total_layers = int(report.get("total_layers", 0))
    total_scored_layers = int(report.get("total_scored_layers", 0))
    enough_scored_layers = True
    if decision_cfg.use_adjusted_score:
        enough_scored_layers = (total_layers > 0) and (
            total_scored_layers > total_layers * float(decision_cfg.min_scored_layer_frac)
        )

    is_barrel = enough_scored_layers and (final_score >= decision_cfg.barrel_valid_ratio)
    layer_details = list(report.get("layer_details", []) or [])

    invalid_reasons: list[str] = []
    junk_reasons: list[str] = []
    for layer in layer_details:
        reason = str(layer.get("reason", "") or "")
        if not reason or reason == "OK":
            continue
        if reason.startswith("JUNK"):
            junk_reasons.append(reason)
        else:
            invalid_reasons.append(reason)

    if invalid_reasons:
        main_reason = Counter(invalid_reasons).most_common(1)[0][0]
    elif junk_reasons:
        main_reason = Counter(junk_reasons).most_common(1)[0][0]
    else:
        main_reason = "Too few valid slices"

    if is_barrel:
        reason = "OK"
    elif not enough_scored_layers:
        reason = "Too few scored slices for a stable decision"
    else:
        reason = main_reason

    return build_row("BARREL" if is_barrel else "NON_BARREL", reason, report)


def prepare_one_file(file_path: str, cfg: AppConfig) -> list[dict[str, object]] | dict[str, str]:
    """
    Parse a structure once, run DSSP once, and produce per-chain payloads.
    """
    filename = os.path.basename(file_path)
    try:
        cached_payloads = load_prepare_payloads(file_path, cfg)
        if cached_payloads is not None:
            return cached_payloads

        loader = ProteinLoader(
            file_path,
            dssp_bin=cfg.runtime.dssp_bin_path,
            fail_on_dssp_error=cfg.runtime.fail_on_dssp_error,
        )
    except Exception as exc:
        return {"_error": f"{filename}: {exc}"}

    payloads: list[dict[str, object]] = []
    for chain in loader.model:
        chain_id = chain.id
        try:
            residues_data = loader.get_chain_data(chain_id)
        except Exception as exc:
            return {"_error": f"{filename}: {exc}"}

        if len(residues_data) < cfg.input.min_chain_residues:
            continue
        payloads.append(
            {
                "filename": filename,
                "chain": chain_id,
                "residues_data": residues_data,
            }
        )

    store_prepare_payloads(file_path, cfg, payloads)
    return payloads


def collect_payloads(
    files: Iterable[str],
    cfg: AppConfig,
    prepare_workers: int,
) -> list[dict[str, object]]:
    """
    Run the preparation phase and collect chain-level payloads.
    """
    file_list = list(files)
    payloads: list[dict[str, object]] = []

    if prepare_workers <= 1:
        for file_path in tqdm(file_list, desc="Preparing", unit="file"):
            result = prepare_one_file(file_path, cfg)
            if isinstance(result, dict) and result.get("_error"):
                tqdm_write(f"  [X] Load failed: {result['_error']}")
                continue
            payloads.extend(result)
        return payloads

    with ProcessPoolExecutor(max_workers=prepare_workers) as executor:
        futures = [executor.submit(prepare_one_file, file_path, cfg) for file_path in file_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preparing", unit="file"):
            try:
                result = future.result()
            except Exception as exc:
                tqdm_write(f"  [X] Prepare worker failed: {exc}")
                continue

            if isinstance(result, dict) and result.get("_error"):
                tqdm_write(f"  [X] Load failed: {result['_error']}")
                continue
            payloads.extend(result)

    return payloads


def run_analysis(
    payloads: list[dict[str, object]],
    cfg: AppConfig,
    workers: int,
) -> list[dict[str, object]]:
    """
    Run the CPU-heavy chain analysis stage.
    """
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(analyze_chain_payload, payload, cfg) for payload in payloads]
        with tqdm(total=len(futures), desc="Analyzing", unit="chain") as progress_bar:
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        {
                            "filename": "",
                            "chain": "",
                            "result": "ERROR",
                            "reason": f"Worker crashed: {exc}",
                        }
                    )
                finally:
                    progress_bar.update(1)
    return results
