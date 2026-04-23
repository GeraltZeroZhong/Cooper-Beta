from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None

from cooper_beta import pipeline
from cooper_beta.constants import LEGACY_RESULT_SKIP, RESULT_FILTERED_OUT

from .metrics import (
    compute_chain_metrics,
    compute_file_metrics,
    ensure_columns,
    print_metrics,
)


def _is_filtered_result(result_series: pd.Series) -> pd.Series:
    return result_series.isin({RESULT_FILTERED_OUT, LEGACY_RESULT_SKIP})


def run_detector(
    folder: Path,
    workers: int | None,
    prepare_workers: int | None,
    detector_overrides: dict[str, object] | list[str] | None = None,
) -> pd.DataFrame:
    if pd is None:
        raise RuntimeError("pandas is required to run evaluation.")

    folder = folder.resolve()
    if not folder.exists():
        raise FileNotFoundError(str(folder))

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="cooper_beta_eval_") as temp_dir:
        output_csv = Path(temp_dir) / "cooper_beta_results.csv"
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            pipeline.main(
                str(folder),
                workers=workers,
                prepare_workers=prepare_workers,
                out_csv=str(output_csv),
                overrides=detector_overrides,
            )

        if not output_csv.exists():
            captured_error = stderr_buffer.getvalue().strip()
            raise RuntimeError(
                f"Expected CSV not found: {output_csv}\nCaptured stderr:\n{captured_error[:2000]}"
            )
        return pd.read_csv(output_csv)


def save_outputs(
    positive_raw: pd.DataFrame,
    negative_raw: pd.DataFrame,
    save_dir: Path,
    tag: str,
) -> tuple[str, str, pd.DataFrame]:
    save_dir.mkdir(parents=True, exist_ok=True)

    def decorate(dataframe: pd.DataFrame, y_true: int, split: str) -> pd.DataFrame:
        decorated = ensure_columns(dataframe)
        decorated["y_true"] = y_true
        decorated["split"] = split
        result_series = decorated["result"].astype(str).str.upper()
        decorated["is_error"] = result_series.eq("ERROR")
        decorated["is_filtered_out"] = _is_filtered_result(result_series)
        decorated["is_skip"] = decorated["is_filtered_out"]
        decorated["pred_barrel"] = result_series.eq("BARREL")
        decorated["use_for_metrics"] = ~decorated["is_error"]
        decorated["sample_id"] = (
            decorated["filename"].astype(str) + ":" + decorated["chain"].astype(str)
        )
        decorated["decision_score"] = (
            pd.to_numeric(decorated["decision_score"], errors="coerce")
            .fillna(pd.to_numeric(decorated["score_adjust"], errors="coerce").fillna(0.0))
        )
        decorated["score_adjust"] = (
            pd.to_numeric(decorated["score_adjust"], errors="coerce").fillna(0.0)
        )
        return decorated

    positive_decorated = decorate(positive_raw, 1, "true")
    negative_decorated = decorate(negative_raw, 0, "false")
    combined = pd.concat([positive_decorated, negative_decorated], ignore_index=True)

    chain_output = save_dir / f"eval_chain_results_{tag}.csv"
    combined.to_csv(chain_output, index=False)

    no_error = combined[combined["use_for_metrics"]].copy()
    aggregated = no_error.groupby(["split", "y_true", "filename"], as_index=False).agg(
        decision_score_max=("decision_score", "max"),
        score_adjust_max=("score_adjust", "max"),
        pred_barrel_any=("pred_barrel", "max"),
        any_filtered_out=("is_filtered_out", "max"),
        any_skip=("is_skip", "max"),
        chains_n=("sample_id", "count"),
    )
    file_output = save_dir / f"eval_file_results_{tag}.csv"
    aggregated.to_csv(file_output, index=False)

    return str(chain_output), str(file_output), aggregated


def evaluate(
    true_dir: Path,
    false_dir: Path,
    workers: int | None,
    prepare_workers: int | None,
    save_dir: Path,
    metric_level: str,
    tag: str,
    detector_overrides: dict[str, object] | list[str] | None = None,
    print_metric_tables: bool = True,
) -> dict[str, object]:
    if pd is None:
        raise RuntimeError("pandas is required (pip install -e '.[full]').")

    positive_dataframe = run_detector(true_dir, workers, prepare_workers, detector_overrides)
    negative_dataframe = run_detector(false_dir, workers, prepare_workers, detector_overrides)

    chain_csv, file_csv, aggregated = save_outputs(
        positive_dataframe,
        negative_dataframe,
        save_dir,
        tag,
    )

    row: dict[str, object] = {
        "tag": tag,
        "chain_csv": chain_csv,
        "file_csv": file_csv,
    }
    if detector_overrides and isinstance(detector_overrides, dict):
        row.update(detector_overrides)

    if metric_level in ("chain", "both"):
        chain_metrics, chain_extra = compute_chain_metrics(positive_dataframe, negative_dataframe)
        row.update({f"chain_{key}": value for key, value in chain_metrics.items()})
        row.update({f"chain_{key}": value for key, value in chain_extra.items()})
        if print_metric_tables:
            print_metrics("=== Chain-level ===", chain_metrics)
            print(
                "Dropped ERROR chains: "
                f"true={chain_extra['dropped_true_error']} "
                f"false={chain_extra['dropped_false_error']}\n"
            )

    if metric_level in ("file", "both"):
        file_metrics, file_extra = compute_file_metrics(aggregated)
        row.update({f"file_{key}": value for key, value in file_metrics.items()})
        row.update({f"file_{key}": value for key, value in file_extra.items()})
        if print_metric_tables:
            print_metrics("=== File-level ===", file_metrics)
            print("Breakdown (file-level):")
            print(f"  true:  any_FILTERED_OUT={file_extra['true_any_filtered_out']}")
            print(f"  false: any_FILTERED_OUT={file_extra['false_any_filtered_out']}\n")
            print("Note:")
            print("  File-level metric uses discrete prediction pred_barrel_any (any chain == BARREL).")
            print("  For ROC/PR at file-level, use 'decision_score_max' in the saved file CSV.\n")

    return row
