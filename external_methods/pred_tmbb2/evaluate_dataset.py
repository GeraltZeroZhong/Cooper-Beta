from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from external_methods.pred_tmbb2.runner import (
    DEFAULT_MIN_TM_STRANDS,
    DEFAULT_PREDICTION_FIELD,
    PredTmbb2Result,
    run_baseline,
)
from external_methods.pred_tmbb2.sequences import (
    DEFAULT_MIN_RESIDUES,
    GeneratedFastaSet,
    GeneratedSequence,
    generate_structure_fasta,
)

BASELINE_NAME = "pred_tmbb2_single_juchmme"
RAW_CHAIN_FIELDS = [
    "filename",
    "chain",
    "result",
    "result_stage",
    "decision_score",
    "decision_basis",
    "decision_threshold",
    "score_raw",
    "score_adjust",
    "valid_layers",
    "scored_layers",
    "total_layers",
    "valid_layer_frac",
    "scored_layer_frac",
    "junk_layers",
    "invalid_layers",
    "avg_radius",
    "chain_residues",
    "sheet_residues",
    "informative_slices",
    "reason",
    "all_adjusted_layers",
    "all_layers",
    "y_true",
    "split",
    "is_error",
    "is_filtered_out",
    "is_skip",
    "pred_barrel",
    "use_for_metrics",
    "sample_id",
    "baseline",
    "source_file",
    "pdb_id",
    "tm_strands",
    "prediction_field",
    "reliability",
    "algorithm_score",
    "length",
    "logodds",
    "max_prob",
    "neg_logprob_per_length",
    "topology",
    "original_split",
]
MANUAL_EXTRA_FIELDS = [
    "final_split",
    "include_for_metrics",
    "policy",
    "manual_label_reason",
]
FILE_FIELDS = [
    "split",
    "y_true",
    "filename",
    "decision_score_max",
    "score_adjust_max",
    "pred_barrel_any",
    "any_filtered_out",
    "any_skip",
    "chains_n",
]
SUMMARY_FIELDS = [
    "scope",
    "level",
    "n_used",
    "TP",
    "FP",
    "TN",
    "FN",
    "recall",
    "precision",
    "f1",
    "specificity",
    "accuracy",
    "balanced_accuracy",
    "mcc",
]


@dataclass(frozen=True)
class SplitRun:
    split_name: str
    y_true: int
    generated: GeneratedFastaSet
    results: list[PredTmbb2Result]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pdb_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    return stem.split("_", 1)[0].upper()


def _metadata_by_sample(generated: GeneratedFastaSet) -> dict[str, GeneratedSequence]:
    return {record.sample_id: record for record in generated.records}


def _chain_rows_for_split(run: SplitRun) -> list[dict[str, object]]:
    metadata = _metadata_by_sample(run.generated)
    rows: list[dict[str, object]] = []
    result_by_sample = {result.sample_id: result for result in run.results}
    for sample_id, record in metadata.items():
        result = result_by_sample[sample_id]
        filename = Path(record.source_path).name
        pred_barrel = result.result == "BARREL"
        rows.append(
            {
                "filename": filename,
                "chain": record.chain_id,
                "result": result.result,
                "result_stage": BASELINE_NAME,
                "decision_score": result.score,
                "decision_basis": result.decision_rule,
                "decision_threshold": DEFAULT_MIN_TM_STRANDS,
                "score_raw": result.score,
                "score_adjust": result.score,
                "valid_layers": 0,
                "scored_layers": result.tm_strands,
                "total_layers": 0,
                "valid_layer_frac": 0.0,
                "scored_layer_frac": 0.0,
                "junk_layers": 0,
                "invalid_layers": 0,
                "avg_radius": 0.0,
                "chain_residues": record.n_residues,
                "sheet_residues": 0,
                "informative_slices": result.tm_strands,
                "reason": result.decision_rule,
                "all_adjusted_layers": "",
                "all_layers": "",
                "y_true": run.y_true,
                "split": "true" if run.y_true == 1 else "false",
                "is_error": False,
                "is_filtered_out": False,
                "is_skip": False,
                "pred_barrel": pred_barrel,
                "use_for_metrics": True,
                "sample_id": result.sample_id,
                "baseline": BASELINE_NAME,
                "source_file": record.source_path,
                "pdb_id": _pdb_id_from_filename(filename),
                "tm_strands": result.tm_strands,
                "prediction_field": result.prediction_field,
                "reliability": result.reliability,
                "algorithm_score": result.algorithm_score,
                "length": result.length,
                "logodds": result.logodds,
                "max_prob": result.max_prob,
                "neg_logprob_per_length": result.neg_logprob_per_length,
                "topology": result.topology,
                "original_split": run.split_name,
            }
        )
    return rows


def _file_rows_from_chain_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("use_for_metrics") is False:
            continue
        grouped[str(row["filename"])].append(row)

    file_rows: list[dict[str, object]] = []
    for filename in sorted(grouped):
        group = grouped[filename]
        decision_scores = [float(row["decision_score"]) for row in group]
        score_adjusts = [float(row["score_adjust"]) for row in group]
        pred_any = any(bool(row["pred_barrel"]) for row in group)
        y_true = int(group[0]["y_true"])
        file_rows.append(
            {
                "split": "true" if y_true == 1 else "false",
                "y_true": y_true,
                "filename": filename,
                "decision_score_max": max(decision_scores) if decision_scores else 0.0,
                "score_adjust_max": max(score_adjusts) if score_adjusts else 0.0,
                "pred_barrel_any": pred_any,
                "any_filtered_out": any(bool(row["is_filtered_out"]) for row in group),
                "any_skip": any(bool(row["is_skip"]) for row in group),
                "chains_n": len(group),
            }
        )
    return file_rows


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _manual_annotations(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    return {row["filename"]: row for row in _read_csv(path)}


def _apply_manual_annotations(
    chain_rows: Sequence[dict[str, object]],
    annotations: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    reviewed: list[dict[str, object]] = []
    for row in chain_rows:
        filename = str(row["filename"])
        annotation = annotations.get(filename)
        if annotation is None:
            final_split = str(row["original_split"])
            include = True
            policy = "keep_original_label"
            reason = "no explicit correction in manual notes"
        else:
            include = annotation["include_for_metrics"] == "True"
            policy = annotation["policy"]
            reason = annotation["reason"]
            final_split = annotation["final_split"]

        updated = dict(row)
        updated["final_split"] = final_split
        updated["include_for_metrics"] = include
        updated["policy"] = policy
        updated["manual_label_reason"] = reason
        updated["use_for_metrics"] = include
        if final_split == "positive":
            updated["y_true"] = 1
            updated["split"] = "true"
        elif final_split == "negative":
            updated["y_true"] = 0
            updated["split"] = "false"
        else:
            updated["y_true"] = ""
            updated["split"] = final_split
        if include:
            reviewed.append(updated)
    return reviewed


def _with_manual_file_columns(
    file_rows: Sequence[dict[str, object]],
    annotations: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    reviewed: list[dict[str, object]] = []
    for row in file_rows:
        filename = str(row["filename"])
        annotation = annotations.get(filename)
        if annotation is None:
            include = True
            final_split = "positive" if int(row["y_true"]) == 1 else "negative"
            policy = "keep_original_label"
            reason = "no explicit correction in manual notes"
        else:
            include = annotation["include_for_metrics"] == "True"
            final_split = annotation["final_split"]
            policy = annotation["policy"]
            reason = annotation["reason"]
        if not include:
            continue
        updated = dict(row)
        updated["final_split"] = final_split
        updated["include_for_metrics"] = include
        updated["policy"] = policy
        updated["manual_label_reason"] = reason
        if final_split == "positive":
            updated["y_true"] = 1
            updated["split"] = "true"
        elif final_split == "negative":
            updated["y_true"] = 0
            updated["split"] = "false"
        reviewed.append(updated)
    return reviewed


def _metrics(rows: Sequence[dict[str, object]], *, file_level: bool) -> dict[str, object]:
    tp = fp = tn = fn = 0
    for row in rows:
        y_true = int(row["y_true"])
        pred = _boolish(row["pred_barrel_any"] if file_level else row["pred_barrel"])
        if y_true == 1 and pred:
            tp += 1
        elif y_true == 0 and pred:
            fp += 1
        elif y_true == 0 and not pred:
            tn += 1
        elif y_true == 1 and not pred:
            fn += 1

    recall = tp / (tp + fn) if tp + fn else math.nan
    precision = tp / (tp + fp) if tp + fp else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    accuracy = (tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
    balanced_accuracy = (
        (recall + specificity) / 2
        if not math.isnan(recall) and not math.isnan(specificity)
        else math.nan
    )
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else math.nan

    return {
        "n_used": tp + fp + tn + fn,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }


def _summary_rows(
    raw_chain_rows: Sequence[dict[str, object]],
    raw_file_rows: Sequence[dict[str, object]],
    reviewed_chain_rows: Sequence[dict[str, object]],
    reviewed_file_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    cases = [
        ("raw", "chain", raw_chain_rows, False),
        ("raw", "file", raw_file_rows, True),
        ("manual_reviewed", "chain", reviewed_chain_rows, False),
        ("manual_reviewed", "file", reviewed_file_rows, True),
    ]
    output = []
    for scope, level, rows, file_level in cases:
        row = {"scope": scope, "level": level}
        row.update(_metrics(rows, file_level=file_level))
        output.append(row)
    return output


def _write_summary_md(path: Path, summary_rows: Sequence[dict[str, object]]) -> None:
    headers = SUMMARY_FIELDS
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in summary_rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset(
    positive_dir: Path,
    negative_dir: Path,
    output_dir: Path,
    *,
    juchmme_dir: Path,
    manual_manifest: Path | None,
    min_residues: int,
    prediction_field: str,
    min_tm_strands: int,
    java_executable: str,
    timeout: float | None,
    tag: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_runs: list[SplitRun] = []
    for split_name, y_true, input_dir in [
        ("positive", 1, positive_dir),
        ("negative", 0, negative_dir),
    ]:
        split_output = output_dir / split_name
        generated = generate_structure_fasta(
            input_dir,
            split_output,
            min_residues=min_residues,
        )
        results = run_baseline(
            generated.fasta_path,
            juchmme_dir=juchmme_dir,
            work_dir=split_output / "juchmme_work",
            output_path=split_output / "normalized.csv",
            prediction_field=prediction_field,
            min_tm_strands=min_tm_strands,
            java_executable=java_executable,
            timeout=timeout,
        )
        split_runs.append(SplitRun(split_name, y_true, generated, results))

    raw_chain_rows = [row for split_run in split_runs for row in _chain_rows_for_split(split_run)]
    raw_file_rows = _file_rows_from_chain_rows(raw_chain_rows)
    annotations = _manual_annotations(manual_manifest)
    reviewed_chain_rows = _apply_manual_annotations(raw_chain_rows, annotations)
    reviewed_file_rows = _with_manual_file_columns(
        _file_rows_from_chain_rows(reviewed_chain_rows),
        annotations,
    )
    summary_rows = _summary_rows(
        raw_chain_rows,
        raw_file_rows,
        reviewed_chain_rows,
        reviewed_file_rows,
    )

    _write_csv(
        output_dir / f"eval_chain_results_{tag}_raw.csv",
        RAW_CHAIN_FIELDS,
        raw_chain_rows,
    )
    _write_csv(
        output_dir / f"eval_file_results_{tag}_raw.csv",
        FILE_FIELDS,
        raw_file_rows,
    )
    _write_csv(
        output_dir / f"eval_chain_results_{tag}_manual_reviewed.csv",
        [*RAW_CHAIN_FIELDS, *MANUAL_EXTRA_FIELDS],
        reviewed_chain_rows,
    )
    _write_csv(
        output_dir / f"eval_file_results_{tag}_manual_reviewed.csv",
        [*FILE_FIELDS, *MANUAL_EXTRA_FIELDS],
        reviewed_file_rows,
    )
    _write_csv(
        output_dir / f"{BASELINE_NAME}_summary_{tag}.csv",
        SUMMARY_FIELDS,
        summary_rows,
    )
    _write_summary_md(output_dir / f"{BASELINE_NAME}_summary_{tag}.md", summary_rows)

    print(f"Raw chain rows: {len(raw_chain_rows)}")
    print(f"Raw file rows: {len(raw_file_rows)}")
    print(f"Manual-reviewed chain rows: {len(reviewed_chain_rows)}")
    print(f"Manual-reviewed file rows: {len(reviewed_file_rows)}")
    print("Raw chain result counts:", Counter(row["result"] for row in raw_chain_rows))
    print(
        "Manual file final split counts:",
        Counter(row["final_split"] for row in reviewed_file_rows),
    )
    print(f"Output directory: {output_dir.resolve()}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PRED-TMBB2 single-sequence JUCHMME on Cooper-Beta data."
    )
    parser.add_argument("--positive-dir", default="data/positive")
    parser.add_argument("--negative-dir", default="data/negative")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--juchmme-dir", required=True)
    parser.add_argument(
        "--manual-manifest",
        default="eval_outputs/notes_aware_manifest_20260425_021152/notes_aware_file_manifest.csv",
        help="Manual-review manifest with filename/final_split/include_for_metrics/policy/reason.",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--min-residues", type=int, default=DEFAULT_MIN_RESIDUES)
    parser.add_argument(
        "--prediction-field",
        default=DEFAULT_PREDICTION_FIELD,
        choices=["LP", "VP", "lp", "vp"],
    )
    parser.add_argument("--min-tm-strands", type=int, default=DEFAULT_MIN_TM_STRANDS)
    parser.add_argument("--java", default="java")
    parser.add_argument("--timeout", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manual_manifest = Path(args.manual_manifest).expanduser()
    run_dataset(
        Path(args.positive_dir).expanduser(),
        Path(args.negative_dir).expanduser(),
        Path(args.out_dir).expanduser(),
        juchmme_dir=Path(args.juchmme_dir).expanduser(),
        manual_manifest=manual_manifest if str(args.manual_manifest).strip() else None,
        min_residues=args.min_residues,
        prediction_field=args.prediction_field.upper(),
        min_tm_strands=args.min_tm_strands,
        java_executable=args.java,
        timeout=args.timeout,
        tag=args.tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
