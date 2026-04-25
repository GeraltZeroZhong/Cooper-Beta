from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

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
    "layer_counts": "Valid/Scored/Total",
    "reason": "Reason",
}


def _result_fieldnames(rows: list[dict[str, object]]) -> list[str]:
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
    return ordered_keys or list(DEFAULT_RESULT_COLUMNS)


def _row_for_fieldnames(row: dict[str, object], fieldnames: list[str]) -> dict[str, object]:
    return {key: row.get(key, "") for key in fieldnames}


def _ensure_output_parent(output_path: str) -> None:
    parent = Path(output_path).expanduser().parent
    if str(parent) and parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)


class ResultCsvWriter:
    """Incrementally write result rows using the stable Cooper-Beta schema."""

    def __init__(self, output_path: str, fieldnames: Iterable[str] | None = None):
        self.output_path = output_path
        self.fieldnames = list(fieldnames or DEFAULT_RESULT_COLUMNS)
        self._handle = None
        self._writer = None

    def __enter__(self) -> ResultCsvWriter:
        _ensure_output_parent(self.output_path)
        self._handle = open(self.output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        self._writer.writeheader()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()

    def write_rows(self, rows: Iterable[dict[str, object]]) -> None:
        if self._writer is None:
            raise RuntimeError("ResultCsvWriter must be opened before writing rows.")
        for row in rows:
            self._writer.writerow(_row_for_fieldnames(row, self.fieldnames))

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None


def write_results_csv(rows: list[dict[str, object]], output_path: str) -> None:
    """Write result rows without requiring pandas."""
    ordered_keys = _result_fieldnames(rows)
    _ensure_output_parent(output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_for_fieldnames(row, ordered_keys))


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


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counter.most_common())


def _summary_limit_value(summary_limit: int | None) -> int | None:
    if summary_limit is None:
        return 50
    try:
        value = int(summary_limit)
    except (TypeError, ValueError):
        return 50
    if value < 0:
        return None
    return value


def _limited_summary_rows(
    results: list[dict[str, object]],
    summary_limit: int | None,
) -> tuple[list[dict[str, object]], int | None]:
    limit = _summary_limit_value(summary_limit)
    if limit is None:
        return _summary_rows(results), None
    return _summary_rows(results[:limit]), limit


def print_results_summary(
    results: list[dict[str, object]],
    output_path: str,
    *,
    summary_limit: int | None = 50,
    write_csv: bool = True,
    output_written: bool | None = None,
) -> None:
    """Print a human-readable summary and persist the CSV."""
    if output_written is None:
        output_written = write_csv
    summary_rows, resolved_limit = _limited_summary_rows(results, summary_limit)
    result_counts = Counter(str(row.get("result", "") or "<blank>") for row in results)
    stage_counts = Counter(str(row.get("result_stage", "") or "<blank>") for row in results)

    print("\n=== Summary ===")
    print(f"Rows: {len(results)}")
    print(f"Results: {_format_counter(result_counts)}")
    print(f"Stages: {_format_counter(stage_counts)}")

    if pd is not None:
        dataframe = pd.DataFrame(summary_rows)
        if not dataframe.empty:
            display_frame = dataframe[list(DEFAULT_SUMMARY_COLUMNS)].rename(
                columns=SUMMARY_DISPLAY_NAMES
            )
            print()
            print(display_frame.to_string(index=False))
        if resolved_limit is not None and len(results) > resolved_limit:
            omitted = len(results) - resolved_limit
            print(f"\n... omitted {omitted} row(s) from console summary.")
        if write_csv:
            write_results_csv(results, output_path)
        if output_written or write_csv:
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
        f"{'Valid/Scored/Total':<{layer_width}} | "
        f"{'Reason':<{reason_width}}"
    )
    if summary_rows:
        print()
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

    if resolved_limit is not None and len(results) > resolved_limit:
        omitted = len(results) - resolved_limit
        print(f"\n... omitted {omitted} row(s) from console summary.")
    if write_csv:
        write_results_csv(results, output_path)
    if output_written or write_csv:
        print(f"\nResults written to: {output_path}")
