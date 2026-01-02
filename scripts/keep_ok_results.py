#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
keep_ok_results.py

Filter cooper_beta_results.csv to keep only "OK" predictions:
  - result == BARREL
  - reason == OK

Default output: cooper_beta_results_OK.csv

Usage:
  python keep_ok_results.py
  python keep_ok_results.py --input cooper_beta_results.csv --output ok.csv
  python keep_ok_results.py --inplace
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="cooper_beta_results.csv",
        help="输入 CSV（默认: cooper_beta_results.csv）",
    )
    ap.add_argument(
        "--output",
        default="cooper_beta_results_OK.csv",
        help="输出 CSV（默认: cooper_beta_results_OK.csv）",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="直接覆盖输入文件（忽略 --output）",
    )
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        print(f"[X] 输入文件不存在: {in_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(in_path)
    except Exception as e:
        print(f"[X] 读取失败: {in_path}\n    {e}", file=sys.stderr)
        sys.exit(1)

    # 检查必要列
    for col in ("result", "reason"):
        if col not in df.columns:
            print(f"[X] 缺少列: {col}（现有列: {list(df.columns)}）", file=sys.stderr)
            sys.exit(1)

    # 过滤：BARREL + OK
    result_upper = df["result"].astype(str).str.upper()
    reason_str = df["reason"].astype(str)

    df_ok = df[(result_upper == "BARREL") & (reason_str == "OK")].copy()

    out_path = in_path if args.inplace else Path(args.output).resolve()
    try:
        df_ok.to_csv(out_path, index=False)
    except Exception as e:
        print(f"[X] 写出失败: {out_path}\n    {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] 输入: {in_path}")
    print(f"[OK] 输出: {out_path}")
    print(f"[OK] 保留行数: {len(df_ok)} / {len(df)}")


if __name__ == "__main__":
    main()
