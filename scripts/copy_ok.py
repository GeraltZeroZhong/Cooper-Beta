from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def find_in_source(source_dir: Path, relative_or_name: str, recursive: bool) -> Path | None:
    candidate = Path(relative_or_name)

    direct_path = source_dir / candidate
    if direct_path.exists():
        return direct_path

    basename_path = source_dir / candidate.name
    if basename_path.exists():
        return basename_path

    if recursive:
        matches = list(source_dir.rglob(candidate.name))
        if matches:
            return matches[0]
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy or move structure files for OK Cooper-Beta detections into a target folder. "
            "By default this keeps rows where result == BARREL and reason == OK."
        )
    )
    parser.add_argument(
        "--csv",
        default="cooper_beta_results.csv",
        help="Path to the source CSV.",
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Source directory containing structure files.",
    )
    parser.add_argument(
        "--dst",
        default="selected-structures",
        help="Destination directory.",
    )
    parser.add_argument(
        "--no-recursive-search",
        action="store_true",
        help="Disable recursive basename search when direct lookup fails.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing destination files instead of overwriting them.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Copy every unique filename in the CSV instead of filtering to OK detections.",
    )
    parser.add_argument(
        "--result",
        default="BARREL",
        help='Result value to keep when filtering (default: "BARREL").',
    )
    parser.add_argument(
        "--reason",
        default="OK",
        help='Reason value to keep when filtering (default: "OK").',
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    source_dir = Path(args.src)
    destination_dir = Path(args.dst)
    recursive_search = not args.no_recursive_search
    overwrite = not args.no_overwrite

    dataframe = pd.read_csv(csv_path)
    if "filename" not in dataframe.columns:
        raise ValueError('CSV is missing the required column: "filename"')
    if not args.all:
        required_columns = {"result", "reason"}
        missing_columns = sorted(required_columns - set(dataframe.columns))
        if missing_columns:
            joined = ", ".join(missing_columns)
            raise ValueError(f"CSV is missing required column(s) for OK filtering: {joined}")
        result_matches = dataframe["result"].astype(str).str.upper() == str(args.result).upper()
        reason_matches = dataframe["reason"].astype(str) == str(args.reason)
        dataframe = dataframe[result_matches & reason_matches].copy()

    destination_dir.mkdir(parents=True, exist_ok=True)

    filenames = (
        dataframe["filename"]
        .dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda series: series.ne("")]
        .unique()
        .tolist()
    )

    copied = 0
    skipped = 0
    missing: list[str] = []

    for filename in filenames:
        source_path = find_in_source(source_dir, filename, recursive_search)
        if source_path is None:
            missing.append(filename)
            continue

        destination_path = destination_dir / source_path.name
        if destination_path.exists() and not overwrite:
            skipped += 1
            continue

        if args.move:
            shutil.move(str(source_path), str(destination_path))
        else:
            shutil.copy2(str(source_path), str(destination_path))
        copied += 1

    print(f"Rows selected          : {len(dataframe)}")
    print(f"Total unique filenames : {len(filenames)}")
    print(f"Copied or moved        : {copied}")
    print(f"Skipped (kept existing): {skipped}")
    print(f"Missing                : {len(missing)}")

    if missing:
        missing_path = destination_dir / "missing_files.txt"
        missing_path.write_text("\n".join(missing), encoding="utf-8")
        print(f"Missing-file list written to: {missing_path}")


if __name__ == "__main__":
    main()
