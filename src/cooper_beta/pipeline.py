# -*- coding: utf-8 -*-
"""
Beta-barrel detection pipeline.

This module is derived from the original project-level `main.py` and refactored for
a package (src/) layout. The main entry point remains `main()` for backward
compatibility.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
import os
import glob
import csv
import numpy as np

# 进度条（可选依赖）。如果未安装 tqdm，会自动退化为普通迭代。
try:
    from tqdm.auto import tqdm
    _tqdm_write = tqdm.write
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

    def _tqdm_write(msg):
        print(msg)

try:
    import pandas as pd
except Exception:
    pd = None


def _write_results_csv(rows, path):
    """Write results to CSV without requiring pandas."""
    if not rows:
        return
    # Stable, human-friendly column order
    preferred = [
        "filename", "chain", "result",
        "score_adjust", "valid_layers", "all_adjusted_layers", "all_layers",
        "reason",
    ]
    keys = []
    seen = set()
    for k in preferred:
        if any(k in r for r in rows):
            keys.append(k); seen.add(k)
    # Add any remaining keys in sorted order
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import Config
from .loader import ProteinLoader
from .alignment import PCAAligner
from .slicer import ProteinSlicer
from .analyzer import BarrelAnalyzer


def _analyze_chain_payload(payload):
    """
    多进程 worker：分析单条链（payload 仅包含可序列化数据）。
    返回结果字典（用于汇总/写 CSV）。

    结果字段（保持干净）：
      - score_adjust: 忽略 junk slices 的有效层占比
      - valid_layers: score_adjust 分子（有效计分层数量）
      - all_adjusted_layers: score_adjust 分母（计分层总数，已排除 junk）
      - all_layers: 总切片层数（active layers）
    """
    t0 = time.perf_counter()

    filename = payload.get('filename', '')
    chain_id = payload.get('chain', '')
    residues_data = payload.get('residues_data', []) or []
    num_ca = len(residues_data)

    def _ret(result, reason, report=None):
        report = report or {}
        score_adjust = float(report.get('score_adjust', 0.0))
        valid_layers = int(report.get('valid_layers', 0))
        all_adjusted_layers = int(report.get('total_scored_layers', 0))
        all_layers = int(report.get('total_layers', 0))

        return {
            'filename': filename,
            'chain': chain_id,
            'result': result,
            'score_adjust': score_adjust,
            'valid_layers': valid_layers,
            'all_adjusted_layers': all_adjusted_layers,
            'all_layers': all_layers,
            'reason': reason,
        }

    # 基本过滤
    if (not residues_data) or num_ca < 20:
        return _ret('SKIP', 'Chain too short')

    all_coords = np.array([r['coord'] for r in residues_data], dtype=float)
    sheet_coords = np.array([r['coord'] for r in residues_data if r.get('is_sheet', False)], dtype=float)

    if sheet_coords.shape[0] < 10:
        return _ret('SKIP', 'Not enough beta-sheets')

    # 对齐
    try:
        aligner = PCAAligner()
        aligner.fit(sheet_coords)
        aligned_coords = aligner.transform(all_coords)
    except Exception:
        return _ret('ERROR', 'Alignment failed')

    # 切片
    slicer = ProteinSlicer(step_size=Config.SLICE_STEP_SIZE)
    slices = slicer.slice_structure(aligned_coords, residues_data)

    if len(slices) < 5:
        return _ret('SKIP', 'Too few slices')

    # 分析
    analyzer = BarrelAnalyzer(
        min_points=Config.MIN_POINTS_PER_SLICE,
        max_rmse=Config.MAX_FIT_RMSE,
        min_axis=Config.MIN_AXIS,
        max_axis=Config.MAX_AXIS,
        max_flattening=Config.MAX_FLATTENING,
        valid_ratio=Config.BARREL_VALID_RATIO,
        lsq_method=Config.LSQ_METHOD,
        loss=Config.LSQ_LOSS,
        f_scale=Config.LSQ_F_SCALE,
        min_intersections_for_scoring=Config.MIN_INTERSECTIONS_FOR_SCORING,
        nn_rule_enabled=Config.NN_RULE_ENABLED,
        nn_max_robust_cv=Config.NN_MAX_ROBUST_CV,
        nn_min_inlier_frac=Config.NN_MIN_INLIER_FRAC,
        nn_fail_as_junk=Config.NN_FAIL_AS_JUNK,
        angle_rule_enabled=Config.ANGLE_RULE_ENABLED,
        angle_max_gap_deg=Config.ANGLE_MAX_GAP_DEG,
        angle_order_rule_enabled=Config.ANGLE_ORDER_RULE_ENABLED,
        angle_order_local_step_max=Config.ANGLE_ORDER_LOCAL_STEP_MAX,
        angle_order_min_local_frac=Config.ANGLE_ORDER_MIN_LOCAL_FRAC,
        angle_order_max_mean_circ_dist_norm=Config.ANGLE_ORDER_MAX_MEAN_CIRC_DIST_NORM,
        angle_fail_as_junk=Config.ANGLE_FAIL_AS_JUNK,
    )

    try:
        report = analyzer.analyze(slices)
    except Exception:
        # 失败时，至少给出 all_layers（active layers）用于定位
        fallback = {'total_layers': len([z for z in slices.keys() if len(slices[z]) > 0])}
        return _ret('ERROR', 'Analyzer crashed', fallback)

    # 判定分数（内部用，不输出）
    score_raw = float(report.get('score', 0.0))
    score_adjust = float(report.get('score_adjust', 0.0))
    final_score = score_adjust if Config.USE_ADJUSTED_SCORE else score_raw
    
    # 额外控制条件：计分层数量必须足够（避免 all_adjusted_layers 太小导致误判）
    total_layers = int(report.get('total_layers', 0))
    total_scored_layers = int(report.get('total_scored_layers', 0))

    # 仅当使用 score_adjust 时，才需要对“计分层数量”施加稳定性约束；
    # baseline（USE_ADJUSTED_SCORE=False）时，分母采用 total_layers，因此不应被该门控影响。
    if Config.USE_ADJUSTED_SCORE:
        enough_scored_layers = (total_layers > 0) and (
            total_scored_layers > total_layers * float(Config.MIN_SCORED_LAYER_FRAC)
        )
    else:
        enough_scored_layers = True

    is_barrel = enough_scored_layers and (final_score >= Config.BARREL_VALID_RATIO)

    # 聚合主失败原因（层级）
    layer_details = report.get('layer_details', []) or []
    invalid_reasons = []
    junk_reasons = []
    for d in layer_details:
        r = str(d.get('reason', '') or '')
        if not r or r == 'OK':
            continue
        if r.startswith('JUNK'):
            junk_reasons.append(r)
        else:
            invalid_reasons.append(r)

    if invalid_reasons:
        main_reason = Counter(invalid_reasons).most_common(1)[0][0]
    elif junk_reasons:
        main_reason = Counter(junk_reasons).most_common(1)[0][0]
    else:
        main_reason = 'Low score'

    reason = 'OK' if is_barrel else ('Too few scored layers' if (not enough_scored_layers) else main_reason)

    return _ret('BARREL' if is_barrel else 'NON_BARREL', reason, report)

def _prepare_one_file(file_path, dssp_bin_path=None):
    """
    worker：解析单个文件并运行一次 DSSP，返回该文件的链 payload 列表。
    """
    filename = os.path.basename(file_path)
    try:
        loader = ProteinLoader(file_path, dssp_bin=dssp_bin_path)
    except Exception as e:
        return {'_error': f"{filename}: {e}"}

    out = []
    for chain in loader.model:
        chain_id = chain.id
        if len(chain) < 20:
            continue
        residues_data = loader.get_chain_data(chain_id)
        out.append({'filename': filename, 'chain': chain_id, 'residues_data': residues_data})
    return out


def _collect_payloads(files, prepare_workers=1):
    """
    准备阶段：对每个文件只运行一次 DSSP/解析结构，提取每条链的 residues_data，生成 payload 列表。

    - prepare_workers=1：单进程（最省资源，但慢）
    - prepare_workers>1：多进程并行“按文件”跑 DSSP（更快，但更吃 CPU/IO）
    """
    payloads = []

    # 单进程
    if prepare_workers is None or prepare_workers <= 1:
        for file_path in tqdm(files, desc="Preparing", unit="file"):
            filename = os.path.basename(file_path)
            try:
                loader = ProteinLoader(file_path, dssp_bin=Config.DSSP_BIN_PATH)
            except Exception as e:
                _tqdm_write(f"  [X] Load failed: {filename}: {e}")
                continue

            for chain in loader.model:
                chain_id = chain.id
                if len(chain) < 20:
                    continue
                residues_data = loader.get_chain_data(chain_id)
                payloads.append({'filename': filename, 'chain': chain_id, 'residues_data': residues_data})

        return payloads

    # 多进程：按文件并行（每个文件仍只跑一次 DSSP）
    prepare_workers = int(max(1, prepare_workers))
    with ProcessPoolExecutor(max_workers=prepare_workers) as ex:
        futs = [ex.submit(_prepare_one_file, fp, Config.DSSP_BIN_PATH) for fp in files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Preparing", unit="file"):
            try:
                res = fut.result()
            except Exception as e:
                _tqdm_write(f"  [X] Prepare worker failed: {e}")
                continue

            # 错误回传
            if isinstance(res, dict) and res.get('_error'):
                _tqdm_write(f"  [X] Load failed: {res['_error']}")
                continue

            # 正常 payload list
            if res:
                payloads.extend(res)

    return payloads
def main(input_path, workers=None, prepare_workers=None, out_csv='cooper_beta_results.csv'):
    out_csv = str(out_csv)
    if os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*.pdb")) + glob.glob(os.path.join(input_path, "*.cif")) + glob.glob(os.path.join(input_path, "*.mmcif"))
        if not files:
            print("未找到 .pdb/.cif/.mmcif 文件")
            return
    else:
        files = [input_path]

    payloads = _collect_payloads(files, prepare_workers=prepare_workers)
    if not payloads:
        print("无可用任务")
        return

    if workers is None:
        cpu = os.cpu_count() or 1
        workers = max(1, cpu - 1)

    print(f"\nRunning analysis with {workers} worker(s) 等")

    results = []
    # 进程池：多核加速（适用于 CPU-heavy 的拟合）
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_analyze_chain_payload, p) for p in payloads]
        with tqdm(total=len(futs), desc="Analyzing", unit="chain") as pbar:
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({'filename': '', 'chain': '', 'result': 'ERROR', 'reason': f'Worker crashed: {e}'})
                finally:
                    pbar.update(1)

    # 输出汇总
    if pd is not None:
        df = pd.DataFrame(results)
        cols = [c for c in ['filename','chain','result','score_adjust','valid_layers','all_adjusted_layers','all_layers','reason'] if c in df.columns]
        print("\n=== Summary ===")
        print(df[cols].to_string(index=False))

        df.to_csv(out_csv, index=False)
        print(f"\n结果已保存: {out_csv}")
    else:
        header = f"{'Filename':<20} | {'Chain':<5} | {'Result':<10} | {'ScoreAdj':<8} | {'Valid':<9} | {'Radius':<7} | {'Reason':<25}"
        print("\n=== Summary ===")
        print(header)
        print("-" * len(header))
        for r in results:
            print(f"{r.get('filename',''):<20} | {r.get('chain',''):<5} | {r.get('result',''):<10} | "
                  f"{r.get('score_adjust',0.0):<8.2f} | {r.get('valid_layers',''):<9} | {str(r.get('avg_radius','')):<7} | "
                  f"{r.get('reason',''):<25}")

        _write_results_csv(results, out_csv)
        print(f"\n结果已保存: {out_csv}")
