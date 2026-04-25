#!/usr/bin/env python3

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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description='Filter a Cooper-Beta results CSV to rows where `result == "BARREL"` and `reason == "OK"`.'
    )
    ap.add_argument(
        "--input",
        default="cooper_beta_results.csv",
        help="Input CSV path (default: cooper_beta_results.csv).",
    )
    ap.add_argument(
        "--output",
        default="cooper_beta_results_OK.csv",
        help="Output CSV path (default: cooper_beta_results_OK.csv).",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input CSV in place and ignore --output.",
    )
    args = ap.parse_args(argv)

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        print(f"[ERROR] Input CSV does not exist: {in_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(in_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {in_path}\n    {e}", file=sys.stderr)
        sys.exit(1)

    # Validate required columns.
    for col in ("result", "reason"):
        if col not in df.columns:
            print(f"[ERROR] Missing required column: {col}. Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

    # Keep rows classified as BARREL with reason OK.
    result_upper = df["result"].astype(str).str.upper()
    reason_str = df["reason"].astype(str)

    df_ok = df[(result_upper == "BARREL") & (reason_str == "OK")].copy()

    out_path = in_path if args.inplace else Path(args.output).resolve()
    try:
        df_ok.to_csv(out_path, index=False)
    except Exception as e:
        print(f"[ERROR] Failed to write CSV: {out_path}\n    {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] Input : {in_path}")
    print(f"[OK] Output: {out_path}")
    print(f"[OK] Rows kept: {len(df_ok)} / {len(df)}")


if __name__ == "__main__":
    main()
