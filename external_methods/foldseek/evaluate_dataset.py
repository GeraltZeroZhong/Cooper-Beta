from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from external_methods.foldseek.runner import (
    BASELINE_NAME,
    DEFAULT_ALIGNMENT_TYPE,
    DEFAULT_EVALUE,
    DEFAULT_MAX_SEQS,
    DEFAULT_MIN_QUERY_COVERAGE,
    DEFAULT_MIN_TARGET_COVERAGE,
    DEFAULT_SCORE_MODE,
    DEFAULT_SCORE_THRESHOLD,
    SUPPORTED_SCORE_MODES,
    FoldseekResult,
    run_baseline,
)
from external_methods.foldseek.structures import (
    DEFAULT_MIN_RESIDUES,
    GeneratedStructureChain,
    GeneratedStructureSet,
    foldseek_query_aliases,
    generate_structure_chains,
)

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
    "original_split",
    "alignment_type",
    "score_mode",
    "score_threshold",
    "min_query_coverage",
    "min_target_coverage",
    "reference_policy",
    "hit_count",
    "eligible_hit_count",
    "ignored_target_hit_count",
    "best_target",
    "best_target_source_file",
    "best_target_pdb_id",
    "qlen",
    "tlen",
    "alnlen",
    "qcov",
    "tcov",
    "qtmscore",
    "ttmscore",
    "alntmscore",
    "evalue",
    "bits",
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
DEFAULT_MANUAL_MANIFEST = (
    "eval_outputs/notes_aware_manifest_20260425_021152/notes_aware_file_manifest.csv"
)
DEFAULT_REFERENCE_POLICY = "positive_reference_same_pdb_targets_excluded"


@dataclass(frozen=True)
class SplitRun:
    split_name: str
    y_true: int
    generated: GeneratedStructureSet
    results: list[FoldseekResult]


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


def _metadata_by_sample(
    generated: GeneratedStructureSet,
) -> dict[str, GeneratedStructureChain]:
    return {record.sample_id: record for record in generated.records}


def _target_ids_by_pdb(
    records: Sequence[GeneratedStructureChain],
) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        grouped[_pdb_id_from_filename(Path(record.source_path).name)].add(record.sample_id)
    return grouped


def _ignore_map_for_queries(
    query_records: Sequence[GeneratedStructureChain],
    reference_ids_by_pdb: dict[str, set[str]],
) -> dict[str, set[str]]:
    ignored: dict[str, set[str]] = {}
    for record in query_records:
        pdb_id = _pdb_id_from_filename(Path(record.source_path).name)
        target_ids = reference_ids_by_pdb.get(pdb_id, set())
        if target_ids:
            ignored[record.sample_id] = set(target_ids)
    return ignored


def _chain_rows_for_split(
    run: SplitRun,
    *,
    reference_metadata: dict[str, GeneratedStructureChain],
    alignment_type: int,
    reference_policy: str,
) -> list[dict[str, object]]:
    metadata = _metadata_by_sample(run.generated)
    rows: list[dict[str, object]] = []
    result_by_sample = {result.sample_id: result for result in run.results}
    for sample_id, record in metadata.items():
        result = result_by_sample[sample_id]
        filename = Path(record.source_path).name
        pred_barrel = result.result == "BARREL"
        best_target_record = reference_metadata.get(result.best_target or "")
        best_target_source_file = (
            best_target_record.source_path if best_target_record is not None else ""
        )
        best_target_pdb_id = (
            _pdb_id_from_filename(Path(best_target_record.source_path).name)
            if best_target_record is not None
            else ""
        )
        rows.append(
            {
                "filename": filename,
                "chain": record.chain_id,
                "result": result.result,
                "result_stage": BASELINE_NAME,
                "decision_score": result.score,
                "decision_basis": result.decision_rule,
                "decision_threshold": result.score_threshold,
                "score_raw": result.score,
                "score_adjust": result.score,
                "valid_layers": 0,
                "scored_layers": 0,
                "total_layers": 0,
                "valid_layer_frac": 0.0,
                "scored_layer_frac": 0.0,
                "junk_layers": 0,
                "invalid_layers": 0,
                "avg_radius": 0.0,
                "chain_residues": record.n_residues,
                "sheet_residues": 0,
                "informative_slices": result.eligible_hit_count,
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
                "original_split": run.split_name,
                "alignment_type": alignment_type,
                "score_mode": result.score_mode,
                "score_threshold": result.score_threshold,
                "min_query_coverage": result.min_query_coverage,
                "min_target_coverage": result.min_target_coverage,
                "reference_policy": reference_policy,
                "hit_count": result.hit_count,
                "eligible_hit_count": result.eligible_hit_count,
                "ignored_target_hit_count": result.ignored_target_hit_count,
                "best_target": result.best_target,
                "best_target_source_file": best_target_source_file,
                "best_target_pdb_id": best_target_pdb_id,
                "qlen": result.qlen,
                "tlen": result.tlen,
                "alnlen": result.alnlen,
                "qcov": result.qcov,
                "tcov": result.tcov,
                "qtmscore": result.qtmscore,
                "ttmscore": result.ttmscore,
                "alntmscore": result.alntmscore,
                "evalue": result.evalue,
                "bits": result.bits,
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


def _default_foldseek_executable() -> str:
    local_binary = Path("tools/foldseek/bin/foldseek")
    if local_binary.exists():
        return str(local_binary.resolve())
    return os.environ.get("FOLDSEEK_BIN", "foldseek")


def run_dataset(
    positive_dir: Path,
    negative_dir: Path,
    output_dir: Path,
    *,
    reference_dir: Path,
    foldseek_executable: str | Path | None,
    manual_manifest: Path | None,
    min_residues: int,
    create_index: bool,
    alignment_type: int,
    score_mode: str,
    score_threshold: float,
    min_query_coverage: float,
    min_target_coverage: float,
    evalue: float,
    max_seqs: int,
    extra_args: Sequence[str] | None,
    timeout: float | None,
    tag: str,
    reference_policy: str = DEFAULT_REFERENCE_POLICY,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_generated = generate_structure_chains(
        reference_dir,
        output_dir / "reference_chains",
        min_residues=min_residues,
    )
    reference_metadata = _metadata_by_sample(reference_generated)
    reference_ids_by_pdb = _target_ids_by_pdb(reference_generated.records)
    target_aliases = foldseek_query_aliases(reference_generated.records)

    split_runs: list[SplitRun] = []
    for split_name, y_true, input_dir in [
        ("positive", 1, positive_dir),
        ("negative", 0, negative_dir),
    ]:
        split_output = output_dir / split_name
        generated = (
            reference_generated
            if input_dir.resolve() == reference_dir.resolve()
            else generate_structure_chains(
                input_dir,
                split_output / "query_chains",
                min_residues=min_residues,
            )
        )
        results = run_baseline(
            generated.chain_dir,
            reference_generated.chain_dir,
            foldseek_executable=foldseek_executable,
            work_dir=split_output / "foldseek_work",
            output_path=split_output / "normalized.csv",
            query_ids=[record.sample_id for record in generated.records],
            query_aliases=foldseek_query_aliases(generated.records),
            target_aliases=target_aliases,
            ignore_target_ids_by_query=_ignore_map_for_queries(
                generated.records,
                reference_ids_by_pdb,
            ),
            build_target_db=True,
            create_index=create_index,
            alignment_type=alignment_type,
            score_mode=score_mode,
            score_threshold=score_threshold,
            min_query_coverage=min_query_coverage,
            min_target_coverage=min_target_coverage,
            evalue=evalue,
            max_seqs=max_seqs,
            extra_args=extra_args,
            timeout=timeout,
        )
        split_runs.append(SplitRun(split_name, y_true, generated, results))

    raw_chain_rows = [
        row
        for split_run in split_runs
        for row in _chain_rows_for_split(
            split_run,
            reference_metadata=reference_metadata,
            alignment_type=alignment_type,
            reference_policy=reference_policy,
        )
    ]
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

    print(f"Reference chains: {len(reference_generated.records)}")
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
        description="Evaluate Foldseek global-TMalign structure search on Cooper-Beta data."
    )
    parser.add_argument("--positive-dir", default="data/positive")
    parser.add_argument("--negative-dir", default="data/negative")
    parser.add_argument(
        "--reference-dir",
        help="Reference barrel structures. Defaults to --positive-dir.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--foldseek",
        default=_default_foldseek_executable(),
        help="Foldseek executable. Defaults to tools/foldseek/bin/foldseek, FOLDSEEK_BIN, or foldseek.",
    )
    parser.add_argument(
        "--manual-manifest",
        default=DEFAULT_MANUAL_MANIFEST,
        help="Manual-review manifest with filename/final_split/include_for_metrics/policy/reason.",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--min-residues", type=int, default=DEFAULT_MIN_RESIDUES)
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Run foldseek createindex for the generated positive-reference DB.",
    )
    parser.add_argument(
        "--alignment-type",
        type=int,
        default=DEFAULT_ALIGNMENT_TYPE,
        help=f"Foldseek alignment type. Default: {DEFAULT_ALIGNMENT_TYPE} for TMalign.",
    )
    parser.add_argument(
        "--score-mode",
        default=DEFAULT_SCORE_MODE,
        choices=sorted(SUPPORTED_SCORE_MODES),
        help=f"Per-hit score used for BARREL decisions. Default: {DEFAULT_SCORE_MODE}.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=f"Minimum score for BARREL. Default: {DEFAULT_SCORE_THRESHOLD}.",
    )
    parser.add_argument(
        "--min-query-coverage",
        type=float,
        default=DEFAULT_MIN_QUERY_COVERAGE,
        help=f"Minimum query coverage for eligible hits. Default: {DEFAULT_MIN_QUERY_COVERAGE}.",
    )
    parser.add_argument(
        "--min-target-coverage",
        type=float,
        default=DEFAULT_MIN_TARGET_COVERAGE,
        help=f"Minimum target coverage for eligible hits. Default: {DEFAULT_MIN_TARGET_COVERAGE}.",
    )
    parser.add_argument("--evalue", type=float, default=DEFAULT_EVALUE, help="Foldseek -e value.")
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=DEFAULT_MAX_SEQS,
        help="Foldseek --max-seqs value.",
    )
    parser.add_argument("--timeout", type=float, help="Optional subprocess timeout in seconds.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args, extra_args = parser.parse_known_args(argv)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    positive_dir = Path(args.positive_dir).expanduser()
    manual_manifest = Path(args.manual_manifest).expanduser()
    run_dataset(
        positive_dir,
        Path(args.negative_dir).expanduser(),
        Path(args.out_dir).expanduser(),
        reference_dir=(
            Path(args.reference_dir).expanduser() if args.reference_dir else positive_dir
        ),
        foldseek_executable=args.foldseek,
        manual_manifest=manual_manifest if str(args.manual_manifest).strip() else None,
        min_residues=args.min_residues,
        create_index=args.create_index,
        alignment_type=args.alignment_type,
        score_mode=args.score_mode,
        score_threshold=args.score_threshold,
        min_query_coverage=args.min_query_coverage,
        min_target_coverage=args.min_target_coverage,
        evalue=args.evalue,
        max_seqs=args.max_seqs,
        extra_args=extra_args,
        timeout=args.timeout,
        tag=args.tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
