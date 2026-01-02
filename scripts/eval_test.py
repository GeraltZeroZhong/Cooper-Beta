#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate Cooper-Beta detector on two folders:
  - data-true/  : positives (beta-barrel-like)
  - data-false/ : negatives (non beta-barrel; e.g. jelly-roll, etc.)

Metrics policy:
  - EXCLUDE ERROR samples from metrics (dropped)
  - INCLUDE SKIP in metrics, treated as predicted negative

This script:
  1) Invokes cooper_beta.pipeline.main() on both folders
  2) Prints metrics at chain-level and/or file-level
  3) Saves raw outputs for future ROC/PR curves:
       - Chain-level combined CSV (y_true, score_adjust, result, flags)
       - File-level aggregated CSV (score_adjust_max per file, etc.)

File-level definition:
  - Drop ERROR chains (but keep file if it still has at least one non-ERROR chain)
  - Continuous score for ROC/PR: score_adjust_max (max over chains per file)
  - Discrete prediction: pred_barrel_any (any chain result == BARREL)

Usage:
  python scripts/eval_test.py
  python scripts/eval_test.py --workers 8 --prepare 4 --save-dir eval_outputs
  python scripts/eval_test.py --metric-level file
  python scripts/eval_test.py --ablation
"""

import argparse
import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path as _Path
from pathlib import Path

# Allow running from a source checkout without installation (adds ../src to sys.path)
_ROOT = _Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import pandas as pd
except Exception:
    pd = None


def _import_main_module():
    # Preferred: installed/packaged layout
    try:
        from cooper_beta import pipeline as m  # type: ignore
        return m
    except Exception:
        pass

    # Fallbacks (legacy)
    try:
        import main as m  # type: ignore
        if hasattr(m, "main"):
            return m
    except Exception:
        pass

    try:
        from cooper_parallel import main as m  # type: ignore
        return m
    except Exception as e:
        raise RuntimeError(
            "Cannot import project main module. "
            "Run this script from the project root, or install the package so `cooper_beta` is importable."
        ) from e


def _run_detector(main_mod, folder: Path, workers: int | None, prepare_workers: int | None) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required because pipeline writes cooper_beta_results.csv via pandas.")

    folder = folder.resolve()
    if not folder.exists():
        raise FileNotFoundError(str(folder))

    tmpdir = tempfile.mkdtemp(prefix="cooper_beta_eval_")
    old_cwd = os.getcwd()

    buf_out = io.StringIO()
    buf_err = io.StringIO()

    try:
        os.chdir(tmpdir)
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            try:
                main_mod.main(str(folder), workers=workers, prepare_workers=prepare_workers)
            except TypeError:
                # legacy signature main(path, workers)
                main_mod.main(str(folder), workers)

        out_csv = Path(tmpdir) / "cooper_beta_results.csv"
        if not out_csv.exists():
            err = buf_err.getvalue().strip()
            raise RuntimeError(f"Expected CSV not found: {out_csv}\nCaptured stderr:\n{err[:2000]}")
        df = pd.read_csv(out_csv)
        return df
    finally:
        os.chdir(old_cwd)
        try:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def _ensure_cols(df: "pd.DataFrame") -> "pd.DataFrame":
    df = df.copy()
    for c in [
        "filename",
        "chain",
        "result",
        "score_adjust",
        "valid_layers",
        "all_adjusted_layers",
        "all_layers",
        "reason",
    ]:
        if c not in df.columns:
            df[c] = None
    return df


def _drop_error_chain_level(df: "pd.DataFrame") -> tuple["pd.DataFrame", int]:
    r = df["result"].astype(str).str.upper()
    mask = r != "ERROR"
    dropped = int((~mask).sum())
    return df.loc[mask].copy(), dropped


def _metrics(TP, FP, TN, FN):
    def safe_div(a, b):
        return (a / b) if b else 0.0

    recall = safe_div(TP, TP + FN)
    precision = safe_div(TP, TP + FP)
    specificity = safe_div(TN, TN + FP)
    accuracy = safe_div(TP + TN, TP + TN + FP + FN)
    f1 = safe_div(2 * precision * recall, precision + recall)
    bal_acc = 0.5 * (recall + specificity)
    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": bal_acc,
    }


def _print_metrics(title: str, met: dict):
    if title:
        print(title)
    print("Confusion matrix:")
    print(f"  TP={met['TP']}  FP={met['FP']}")
    print(f"  FN={met['FN']}  TN={met['TN']}\n")

    print("Metrics:")
    print(f"  Recall      : {met['recall']:.4f}")
    print(f"  Precision   : {met['precision']:.4f}")
    print(f"  F1          : {met['f1']:.4f}")
    print(f"  Specificity : {met['specificity']:.4f}")
    print(f"  Accuracy    : {met['accuracy']:.4f}")
    print(f"  Bal. Acc.   : {met['balanced_accuracy']:.4f}\n")


def _save_outputs(df_true_raw: "pd.DataFrame", df_false_raw: "pd.DataFrame", save_dir: Path, tag: str):
    save_dir.mkdir(parents=True, exist_ok=True)

    def decorate(df: "pd.DataFrame", y_true: int, split: str) -> "pd.DataFrame":
        d = _ensure_cols(df)
        d["y_true"] = y_true
        d["split"] = split
        r = d["result"].astype(str).str.upper()
        d["is_error"] = r.eq("ERROR")
        d["is_skip"] = r.eq("SKIP")
        d["pred_barrel"] = r.eq("BARREL")
        d["use_for_metrics"] = ~d["is_error"]  # drop ERROR; keep SKIP
        d["sample_id"] = d["filename"].astype(str) + ":" + d["chain"].astype(str)
        d["score_adjust"] = pd.to_numeric(d["score_adjust"], errors="coerce").fillna(0.0)
        return d

    d_true = decorate(df_true_raw, 1, "true")
    d_false = decorate(df_false_raw, 0, "false")
    df_all = pd.concat([d_true, d_false], ignore_index=True)

    chain_out = save_dir / f"eval_chain_results_{tag}.csv"
    df_all.to_csv(chain_out, index=False)

    df_noerr = df_all[df_all["use_for_metrics"]].copy()

    agg = df_noerr.groupby(["split", "y_true", "filename"], as_index=False).agg(
        score_adjust_max=("score_adjust", "max"),
        pred_barrel_any=("pred_barrel", "max"),
        any_skip=("is_skip", "max"),
        chains_n=("sample_id", "count"),
    )
    file_out = save_dir / f"eval_file_results_{tag}.csv"
    agg.to_csv(file_out, index=False)

    return str(chain_out), str(file_out), agg


@contextlib.contextmanager
def _override_config(overrides: dict | None):
    """
    Temporarily override cooper_beta.config.Config class attributes.
    Lightweight monkey-patch to support ablation studies.
    """
    if not overrides:
        yield
        return

    try:
        from cooper_beta.config import Config  # type: ignore
    except Exception:
        # If the package isn't importable, just run without overrides.
        yield
        return

    old = {}
    for k, v in overrides.items():
        if not hasattr(Config, k):
            raise KeyError(f"Unknown Config field: {k}")
        old[k] = getattr(Config, k)
        setattr(Config, k, v)

    try:
        yield
    finally:
        for k, v in old.items():
            setattr(Config, k, v)


def _compute_chain_metrics(df_true_raw: "pd.DataFrame", df_false_raw: "pd.DataFrame"):
    df_true2, dropped_true = _drop_error_chain_level(df_true_raw)
    df_false2, dropped_false = _drop_error_chain_level(df_false_raw)

    def is_barrel(series):
        return series.astype(str).str.upper().eq("BARREL")

    tp_mask = is_barrel(df_true2["result"])
    fp_mask = is_barrel(df_false2["result"])

    TP = int(tp_mask.sum())
    FN = int((~tp_mask).sum())
    FP = int(fp_mask.sum())
    TN = int((~fp_mask).sum())

    met = _metrics(TP, FP, TN, FN)
    extra = {
        "dropped_true_error": int(dropped_true),
        "dropped_false_error": int(dropped_false),
        "n_true_used": int(len(df_true2)),
        "n_false_used": int(len(df_false2)),
    }
    return met, extra


def _compute_file_metrics(agg: "pd.DataFrame"):
    t = agg[agg["split"] == "true"].copy()
    f = agg[agg["split"] == "false"].copy()

    TP = int(t["pred_barrel_any"].astype(int).sum()) if len(t) else 0
    FN = int(len(t) - TP) if len(t) else 0
    FP = int(f["pred_barrel_any"].astype(int).sum()) if len(f) else 0
    TN = int(len(f) - FP) if len(f) else 0

    met = _metrics(TP, FP, TN, FN)

    true_any_skip = int(t["any_skip"].astype(int).sum()) if len(t) else 0
    false_any_skip = int(f["any_skip"].astype(int).sum()) if len(f) else 0

    extra = {
        "n_true_files": int(len(t)),
        "n_false_files": int(len(f)),
        "true_any_skip": int(true_any_skip),
        "false_any_skip": int(false_any_skip),
    }
    return met, extra


def evaluate(
    main_mod,
    true_dir: Path,
    false_dir: Path,
    workers: int | None,
    prepare_workers: int | None,
    save_dir: Path,
    metric_level: str,
    tag: str,
    overrides: dict | None = None,
    print_metrics: bool = True,
):
    """
    Run detector on (true_dir, false_dir) and compute metrics.
    Returns a dict suitable for tabular ablation summaries.
    """
    if pd is None:
        raise RuntimeError("pandas is required (pip install -e '.[full]').")

    with _override_config(overrides):
        df_true = _run_detector(main_mod, true_dir, workers, prepare_workers)
        df_false = _run_detector(main_mod, false_dir, workers, prepare_workers)

    chain_csv, file_csv, agg = _save_outputs(df_true, df_false, save_dir, tag)

    row = {
        "tag": tag,
        "chain_csv": chain_csv,
        "file_csv": file_csv,
    }
    if overrides:
        for k, v in overrides.items():
            row[k] = v

    if metric_level in ("chain", "both"):
        met_c, extra_c = _compute_chain_metrics(df_true, df_false)
        row.update({f"chain_{k}": v for k, v in met_c.items()})
        row.update({f"chain_{k}": v for k, v in extra_c.items()})
        if print_metrics:
            _print_metrics("=== Chain-level ===", met_c)
            print(f"Dropped ERROR chains: true={extra_c['dropped_true_error']} false={extra_c['dropped_false_error']}\n")

    if metric_level in ("file", "both"):
        met_f, extra_f = _compute_file_metrics(agg)
        row.update({f"file_{k}": v for k, v in met_f.items()})
        row.update({f"file_{k}": v for k, v in extra_f.items()})
        if print_metrics:
            _print_metrics("=== File-level ===", met_f)
            print("Breakdown (file-level):")
            print(f"  true:  any_SKIP={extra_f['true_any_skip']}")
            print(f"  false: any_SKIP={extra_f['false_any_skip']}\n")
            print("Note:")
            print("  File-level metric uses discrete prediction pred_barrel_any (any chain == BARREL).")
            print("  For ROC/PR at file-level, use 'score_adjust_max' in the saved file CSV.\n")

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true", default="data-true", help="正例文件夹（beta-barrel-like）")
    ap.add_argument("--false", default="data-false", help="反例文件夹（non barrel）")
    ap.add_argument("--workers", type=int, default=None, help="分析阶段进程数（默认：pipeline 内部默认）")
    ap.add_argument("--prepare", type=int, default=None, help="Preparing 阶段进程数（默认：跟随 workers）")
    ap.add_argument("--save-dir", default="eval_outputs", help="保存评估结果的目录（CSV，用于 ROC/PR）")
    ap.add_argument(
        "--metric-level",
        choices=["chain", "file", "both"],
        default="both",
        help="打印指标层级：chain/file/both（默认 both）",
    )
    ap.add_argument(
        "--ablation",
        action="store_true",
        help="运行主效应与交互（7 组）消融套件，并输出 summary CSV",
    )
    args = ap.parse_args()

    main_mod = _import_main_module()
    true_dir = Path(args.true)
    false_dir = Path(args.false)

    import datetime as _dt

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.ablation:
        tag = ts
        row = evaluate(
            main_mod=main_mod,
            true_dir=true_dir,
            false_dir=false_dir,
            workers=args.workers,
            prepare_workers=args.prepare,
            save_dir=Path(args.save_dir),
            metric_level=args.metric_level,
            tag=tag,
            overrides=None,
            print_metrics=True,
        )
        print("\nSaved for ROC/PR:")
        print(f"  chain-level: {row['chain_csv']}")
        print(f"  file-level : {row['file_csv']}\n")
        return

    # --- Ablation suite: main effects & interactions (7 experiments) ---
    # Factors:
    #   S: USE_ADJUSTED_SCORE
    #   N: NN_RULE_ENABLED (+ NN_FAIL_AS_JUNK)
    #   A: ANGLE_RULE_ENABLED + ANGLE_ORDER_RULE_ENABLED
    base = {
        "USE_ADJUSTED_SCORE": False,
        "NN_RULE_ENABLED": False,
        "NN_FAIL_AS_JUNK": False,
        "ANGLE_RULE_ENABLED": False,
        "ANGLE_ORDER_RULE_ENABLED": False,
    }

    # Main effects
    nn_only = {**base, "NN_RULE_ENABLED": True, "NN_FAIL_AS_JUNK": True}
    adjusted_only = {**base, "USE_ADJUSTED_SCORE": True}
    angle_only = {**base, "ANGLE_RULE_ENABLED": True, "ANGLE_ORDER_RULE_ENABLED": True}

    # Interactions / combos
    adjusted_nn = {**base, "USE_ADJUSTED_SCORE": True, "NN_RULE_ENABLED": True, "NN_FAIL_AS_JUNK": True}
    adjusted_angle = {**base, "USE_ADJUSTED_SCORE": True, "ANGLE_RULE_ENABLED": True, "ANGLE_ORDER_RULE_ENABLED": True}
    full = {
        **base,
        "USE_ADJUSTED_SCORE": True,
        "NN_RULE_ENABLED": True,
        "NN_FAIL_AS_JUNK": True,
        "ANGLE_RULE_ENABLED": True,
        "ANGLE_ORDER_RULE_ENABLED": True,
    }

    suite = [
        ("A0_baseline_ellipse", base),
        ("A1_nn_only", nn_only),
        ("A2_adjusted_only", adjusted_only),
        ("A3_angle_only", angle_only),
        ("A4_adjusted_plus_nn", adjusted_nn),
        ("A5_adjusted_plus_angle", adjusted_angle),
        ("A6_full", full),
    ]

    ab_dir = Path(args.save_dir) / f"ablation_{ts}"
    ab_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print("\n=== Ablation study (7 exps: main effects & interactions) ===")
    print(f"Output dir: {ab_dir}\n")

    for name, overrides in suite:
        print(f"[{name}] overrides={overrides}")
        tag = f"{ts}_{name}"
        row = evaluate(
            main_mod=main_mod,
            true_dir=true_dir,
            false_dir=false_dir,
            workers=args.workers,
            prepare_workers=args.prepare,
            save_dir=ab_dir,
            metric_level=args.metric_level,
            tag=tag,
            overrides=overrides,
            print_metrics=False,
        )
        row["exp"] = name
        rows.append(row)

        c_f1 = row.get("chain_f1", None)
        f_f1 = row.get("file_f1", None)
        c_bal = row.get("chain_balanced_accuracy", None)
        f_bal = row.get("file_balanced_accuracy", None)

        c_tp = row.get("chain_TP", None)
        c_fp = row.get("chain_FP", None)
        c_tn = row.get("chain_TN", None)
        c_fn = row.get("chain_FN", None)

        f_tp = row.get("file_TP", None)
        f_fp = row.get("file_FP", None)
        f_tn = row.get("file_TN", None)
        f_fn = row.get("file_FN", None)

        parts = []
        if c_f1 is not None and c_bal is not None and c_tp is not None:
            parts.append(
                f"chain: F1={c_f1:.4f}  BalAcc={c_bal:.4f}  (TP={c_tp} FP={c_fp} TN={c_tn} FN={c_fn})"
            )
        if f_f1 is not None and f_bal is not None and f_tp is not None:
            parts.append(
                f"file: F1={f_f1:.4f}  BalAcc={f_bal:.4f}  (TP={f_tp} FP={f_fp} TN={f_tn} FN={f_fn})"
            )
        print("  " + " | ".join(parts) + "\n")

    if pd is None:
        return

    df = pd.DataFrame(rows)

    keep = [
        "exp",
        "chain_recall",
        "chain_precision",
        "chain_f1",
        "chain_specificity",
        "chain_accuracy",
        "chain_balanced_accuracy",
        "file_recall",
        "file_precision",
        "file_f1",
        "file_specificity",
        "file_accuracy",
        "file_balanced_accuracy",
        "chain_TP",
        "chain_FP",
        "chain_TN",
        "chain_FN",
        "file_TP",
        "file_FP",
        "file_TN",
        "file_FN",
        "chain_csv",
        "file_csv",
    ]
    cols = [c for c in keep if c in df.columns] + [c for c in df.columns if c not in keep]
    df = df[cols]

    out = ab_dir / f"ablation_summary_{ts}.csv"
    df.to_csv(out, index=False)

    print("=== Ablation summary ===")
    try:
        show_cols = [c for c in ["exp", "chain_f1", "chain_balanced_accuracy", "file_f1", "file_balanced_accuracy"] if c in df.columns]
        print(df[show_cols].to_string(index=False))
    except Exception:
        pass
    print(f"\nSaved: {out}\n")


if __name__ == "__main__":
    main()
