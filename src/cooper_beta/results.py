from __future__ import annotations

import csv
from collections.abc import Iterable

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from .constants import DEFAULT_RESULT_COLUMNS, SUMMARY_COLUMN_WIDTHS


def write_results_csv(rows: list[dict[str, object]], output_path: str) -> None:
    """Write result rows without requiring pandas."""
    if not rows:
        return

    ordered_keys: list[str] = []
    seen_keys: set[str] = set()
    for key in DEFAULT_RESULT_COLUMNS:
        if any(key in row for row in rows):
            ordered_keys.append(key)
            seen_keys.add(key)

    for row in rows:
        for key in row:
            if key not in seen_keys:
                seen_keys.add(key)
                ordered_keys.append(key)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ordered_keys})


def _summary_columns(results: Iterable[dict[str, object]]) -> list[str]:
    present_columns = {key for row in results for key in row.keys()}
    return [key for key in DEFAULT_RESULT_COLUMNS if key in present_columns]


def print_results_summary(results: list[dict[str, object]], output_path: str) -> None:
    """Print a human-readable summary and persist the CSV."""
    if pd is not None:
        dataframe = pd.DataFrame(results)
        columns = _summary_columns(results)
        print("\n=== Summary ===")
        print(dataframe[columns].to_string(index=False))
        dataframe.to_csv(output_path, index=False)
        print(f"\nResults written to: {output_path}")
        return

    filename_width = SUMMARY_COLUMN_WIDTHS["filename"]
    chain_width = SUMMARY_COLUMN_WIDTHS["chain"]
    result_width = SUMMARY_COLUMN_WIDTHS["result"]
    score_width = SUMMARY_COLUMN_WIDTHS["score_adjust"]
    valid_width = SUMMARY_COLUMN_WIDTHS["valid_layers"]
    reason_width = SUMMARY_COLUMN_WIDTHS["reason"]
    header = (
        f"{'Filename':<{filename_width}} | "
        f"{'Chain':<{chain_width}} | "
        f"{'Result':<{result_width}} | "
        f"{'ScoreAdj':<{score_width}} | "
        f"{'Valid':<{valid_width}} | "
        f"{'Reason':<{reason_width}}"
    )
    print("\n=== Summary ===")
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{str(row.get('filename', '')):<{filename_width}} | "
            f"{str(row.get('chain', '')):<{chain_width}} | "
            f"{str(row.get('result', '')):<{result_width}} | "
            f"{float(row.get('score_adjust', 0.0)):<{score_width}.2f} | "
            f"{str(row.get('valid_layers', '')):<{valid_width}} | "
            f"{str(row.get('reason', '')):<{reason_width}}"
        )

    write_results_csv(results, output_path)
    print(f"\nResults written to: {output_path}")
