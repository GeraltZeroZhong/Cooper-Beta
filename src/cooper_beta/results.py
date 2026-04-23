from __future__ import annotations

import csv
from collections.abc import Iterable

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from .constants import DEFAULT_RESULT_COLUMNS, DEFAULT_SUMMARY_COLUMNS, SUMMARY_COLUMN_WIDTHS

SUMMARY_DISPLAY_NAMES = {
    "filename": "Filename",
    "chain": "Chain",
    "result": "Result",
    "result_stage": "Stage",
    "decision_score": "Score",
    "decision_basis": "Basis",
    "layer_counts": "V/S/T",
    "reason": "Reason",
}


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


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _layer_counts(row: dict[str, object]) -> str:
    valid_layers = _safe_int(row.get("valid_layers", 0))
    scored_layers = _safe_int(row.get("scored_layers", row.get("all_adjusted_layers", 0)))
    total_layers = _safe_int(row.get("total_layers", row.get("all_layers", 0)))
    return f"{valid_layers}/{scored_layers}/{total_layers}"


def _summary_rows(results: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in results:
        result_stage = str(row.get("result_stage", ""))
        decision_basis = str(row.get("decision_basis", "")) if result_stage == "decision" else ""
        decision_score = (
            f"{_safe_float(row.get('decision_score', row.get('score_adjust', 0.0))):.2f}"
            if result_stage == "decision"
            else ""
        )
        rows.append(
            {
                "filename": str(row.get("filename", "")),
                "chain": str(row.get("chain", "")),
                "result": str(row.get("result", "")),
                "result_stage": result_stage,
                "decision_score": decision_score,
                "decision_basis": decision_basis,
                "layer_counts": _layer_counts(row),
                "reason": str(row.get("reason", "")),
            }
        )
    return rows


def print_results_summary(results: list[dict[str, object]], output_path: str) -> None:
    """Print a human-readable summary and persist the CSV."""
    summary_rows = _summary_rows(results)
    if pd is not None:
        dataframe = pd.DataFrame(summary_rows)
        display_frame = dataframe[list(DEFAULT_SUMMARY_COLUMNS)].rename(columns=SUMMARY_DISPLAY_NAMES)
        print("\n=== Summary ===")
        print(
            display_frame.to_string(index=False)
        )
        write_results_csv(results, output_path)
        print(f"\nResults written to: {output_path}")
        return

    filename_width = SUMMARY_COLUMN_WIDTHS["filename"]
    chain_width = SUMMARY_COLUMN_WIDTHS["chain"]
    result_width = SUMMARY_COLUMN_WIDTHS["result"]
    stage_width = SUMMARY_COLUMN_WIDTHS["result_stage"]
    score_width = SUMMARY_COLUMN_WIDTHS["decision_score"]
    basis_width = SUMMARY_COLUMN_WIDTHS["decision_basis"]
    layer_width = SUMMARY_COLUMN_WIDTHS["layer_counts"]
    reason_width = SUMMARY_COLUMN_WIDTHS["reason"]
    header = (
        f"{'Filename':<{filename_width}} | "
        f"{'Chain':<{chain_width}} | "
        f"{'Result':<{result_width}} | "
        f"{'Stage':<{stage_width}} | "
        f"{'Score':<{score_width}} | "
        f"{'Basis':<{basis_width}} | "
        f"{'V/S/T':<{layer_width}} | "
        f"{'Reason':<{reason_width}}"
    )
    print("\n=== Summary ===")
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{str(row.get('filename', '')):<{filename_width}} | "
            f"{str(row.get('chain', '')):<{chain_width}} | "
            f"{str(row.get('result', '')):<{result_width}} | "
            f"{str(row.get('result_stage', '')):<{stage_width}} | "
            f"{str(row.get('decision_score', '')):<{score_width}} | "
            f"{str(row.get('decision_basis', '')):<{basis_width}} | "
            f"{str(row.get('layer_counts', '')):<{layer_width}} | "
            f"{str(row.get('reason', '')):<{reason_width}}"
        )

    write_results_csv(results, output_path)
    print(f"\nResults written to: {output_path}")
