from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd


def _safe_lookup_path(root: Path, value: str) -> Path:
    root = root.expanduser().resolve()
    candidate = Path(value.strip()).expanduser()
    if not str(candidate):
        raise ValueError("Empty filename is not a valid structure path.")
    if candidate.is_absolute():
        resolved = candidate.resolve()
        try:
            return resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Structure path escapes --src: {value!r}") from exc
    if ".." in candidate.parts:
        raise ValueError(f"Unsafe structure filename in CSV: {value!r}")
    return candidate


def _resolve_under(root: Path, relative_path: Path) -> Path | None:
    root = root.expanduser().resolve()
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def find_in_source(source_dir: Path, relative_or_name: str, recursive: bool) -> Path | None:
    source_root = source_dir.expanduser().resolve()
    candidate = _safe_lookup_path(source_root, relative_or_name)

    direct_path = _resolve_under(source_root, candidate)
    if direct_path is not None and direct_path.is_file():
        return direct_path

    basename_path = _resolve_under(source_root, Path(candidate.name))
    if basename_path is not None and basename_path.is_file():
        return basename_path

    if recursive:
        matches = [
            match.resolve()
            for match in source_root.rglob(candidate.name)
            if match.is_file()
        ]
        if len(matches) > 1:
            joined = ", ".join(str(match) for match in matches[:5])
            suffix = " ..." if len(matches) > 5 else ""
            raise ValueError(
                f"Ambiguous recursive match for {candidate.name!r}: {joined}{suffix}"
            )
        if len(matches) == 1:
            return matches[0]
    return None


def _destination_for_source(
    destination_dir: Path,
    source_path: Path,
    seen_basenames: dict[str, str],
) -> Path:
    basename = source_path.name
    resolved_source = str(source_path.resolve())
    previous_source = seen_basenames.setdefault(basename, resolved_source)
    if previous_source == resolved_source:
        return destination_dir / basename
    digest = hashlib.sha256(resolved_source.encode("utf-8")).hexdigest()[:8]
    return destination_dir / f"{source_path.stem}__{digest}{source_path.suffix}"


def main(argv: list[str] | None = None) -> int:
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination files.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit successfully even if some selected files are not found.",
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

    csv_path = Path(args.csv).expanduser()
    source_dir = Path(args.src).expanduser().resolve()
    destination_dir = Path(args.dst).expanduser()
    recursive_search = not args.no_recursive_search
    overwrite = bool(args.overwrite)

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")

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

    lookup_values = dataframe["filename"].fillna("").astype(str).map(str.strip)
    if "source_path" in dataframe.columns:
        source_paths = dataframe["source_path"].fillna("").astype(str).map(str.strip)
        lookup_values = source_paths.where(source_paths.ne(""), lookup_values)
    filenames = lookup_values.loc[lambda series: series.ne("")].unique().tolist()

    copied = 0
    skipped = 0
    missing: list[str] = []
    seen_basenames: dict[str, str] = {}

    for filename in filenames:
        source_path = find_in_source(source_dir, filename, recursive_search)
        if source_path is None:
            missing.append(filename)
            continue

        destination_path = _destination_for_source(destination_dir, source_path, seen_basenames)
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
        if not args.allow_missing:
            return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
