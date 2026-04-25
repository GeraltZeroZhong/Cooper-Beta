from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

BASELINE_NAME = "foldseek_tmalign_structure_search"
DEFAULT_ALIGNMENT_TYPE = 1
DEFAULT_SCORE_MODE = "min_qtmscore_ttmscore"
DEFAULT_SCORE_THRESHOLD = 0.50
DEFAULT_MIN_QUERY_COVERAGE = 0.60
DEFAULT_MIN_TARGET_COVERAGE = 0.60
DEFAULT_EVALUE = 10.0
DEFAULT_MAX_SEQS = 1000
DEFAULT_FORMAT_FIELDS = [
    "query",
    "target",
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
SUPPORTED_SCORE_MODES = {
    "min_qtmscore_ttmscore",
    "qtmscore",
    "ttmscore",
    "alntmscore",
}


@dataclass(frozen=True)
class FoldseekHit:
    query: str
    target: str
    qlen: int | None
    tlen: int | None
    alnlen: int | None
    qcov: float | None
    tcov: float | None
    qtmscore: float | None
    ttmscore: float | None
    alntmscore: float | None
    evalue: float | None
    bits: float | None


@dataclass(frozen=True)
class FoldseekResult:
    sample_id: str
    result: str
    score: float
    decision_rule: str
    score_mode: str
    score_threshold: float
    min_query_coverage: float
    min_target_coverage: float
    hit_count: int
    eligible_hit_count: int
    ignored_target_hit_count: int = 0
    best_target: str | None = None
    qlen: int | None = None
    tlen: int | None = None
    alnlen: int | None = None
    qcov: float | None = None
    tcov: float | None = None
    qtmscore: float | None = None
    ttmscore: float | None = None
    alntmscore: float | None = None
    evalue: float | None = None
    bits: float | None = None

    def as_row(self) -> dict[str, object]:
        return {
            "baseline": BASELINE_NAME,
            "sample_id": self.sample_id,
            "result": self.result,
            "score": self.score,
            "decision_rule": self.decision_rule,
            "score_mode": self.score_mode,
            "score_threshold": self.score_threshold,
            "min_query_coverage": self.min_query_coverage,
            "min_target_coverage": self.min_target_coverage,
            "hit_count": self.hit_count,
            "eligible_hit_count": self.eligible_hit_count,
            "ignored_target_hit_count": self.ignored_target_hit_count,
            "best_target": self.best_target,
            "qlen": self.qlen,
            "tlen": self.tlen,
            "alnlen": self.alnlen,
            "qcov": self.qcov,
            "tcov": self.tcov,
            "qtmscore": self.qtmscore,
            "ttmscore": self.ttmscore,
            "alntmscore": self.alntmscore,
            "evalue": self.evalue,
            "bits": self.bits,
        }


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _require_path(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _foldseek_prefix(
    foldseek_executable: str | Path | None,
    command_prefix: Sequence[str] | None,
) -> list[str]:
    if command_prefix is not None:
        return list(command_prefix)
    if foldseek_executable is None:
        foldseek_executable = os.environ.get("FOLDSEEK_BIN", "foldseek")
    executable_path = Path(foldseek_executable).expanduser()
    if executable_path.exists():
        return [str(executable_path.resolve())]
    return [str(foldseek_executable)]


def _run_foldseek_command(
    args: Sequence[str],
    *,
    work_dir: Path,
    foldseek_executable: str | Path | None,
    command_prefix: Sequence[str] | None,
    timeout: float | None,
) -> None:
    command = [
        *_foldseek_prefix(foldseek_executable, command_prefix),
        *[str(arg) for arg in args],
    ]
    completed = subprocess.run(
        command,
        cwd=work_dir,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        raise RuntimeError(
            "Foldseek baseline failed with exit code "
            f"{completed.returncode} while running: {' '.join(command)}\n"
            f"stdout:\n{stdout[-2000:]}\nstderr:\n{stderr[-2000:]}"
        )


def _row_to_hit(row: Mapping[str, str]) -> FoldseekHit:
    return FoldseekHit(
        query=(row.get("query") or "").strip(),
        target=(row.get("target") or "").strip(),
        qlen=_parse_int(row.get("qlen")),
        tlen=_parse_int(row.get("tlen")),
        alnlen=_parse_int(row.get("alnlen")),
        qcov=_parse_float(row.get("qcov")),
        tcov=_parse_float(row.get("tcov")),
        qtmscore=_parse_float(row.get("qtmscore")),
        ttmscore=_parse_float(row.get("ttmscore")),
        alntmscore=_parse_float(row.get("alntmscore")),
        evalue=_parse_float(row.get("evalue")),
        bits=_parse_float(row.get("bits")),
    )


def load_hits_tsv(
    hits_path: str | Path,
    *,
    fields: Sequence[str] = DEFAULT_FORMAT_FIELDS,
) -> list[FoldseekHit]:
    hits_file = _require_path(hits_path, "Foldseek hits TSV")
    hits: list[FoldseekHit] = []

    with hits_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, values in enumerate(reader, start=1):
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) != len(fields):
                raise ValueError(
                    f"Foldseek hits row {line_number} has {len(values)} columns; "
                    f"expected {len(fields)} for fields {','.join(fields)!r}."
                )
            row = dict(zip(fields, values, strict=True))
            hit = _row_to_hit(row)
            if not hit.query:
                raise ValueError(f"Foldseek hits row {line_number} is missing query id.")
            hits.append(hit)

    return hits


def _score_hit(hit: FoldseekHit, score_mode: str) -> float:
    if score_mode not in SUPPORTED_SCORE_MODES:
        raise ValueError(
            "score_mode must be one of: " + ", ".join(sorted(SUPPORTED_SCORE_MODES))
        )
    if score_mode == "qtmscore":
        return float(hit.qtmscore or 0.0)
    if score_mode == "ttmscore":
        return float(hit.ttmscore or 0.0)
    if score_mode == "alntmscore":
        return float(hit.alntmscore or 0.0)
    return min(float(hit.qtmscore or 0.0), float(hit.ttmscore or 0.0))


def _passes_coverage(
    hit: FoldseekHit,
    *,
    min_query_coverage: float,
    min_target_coverage: float,
) -> bool:
    return (
        float(hit.qcov or 0.0) >= min_query_coverage
        and float(hit.tcov or 0.0) >= min_target_coverage
    )


def _decision_rule(
    *,
    score_mode: str,
    score_threshold: float,
    min_query_coverage: float,
    min_target_coverage: float,
) -> str:
    return (
        f"{score_mode}>={score_threshold:g};"
        f"qcov>={min_query_coverage:g};"
        f"tcov>={min_target_coverage:g}"
    )


def _result_from_best_hit(
    sample_id: str,
    hits: Sequence[FoldseekHit],
    *,
    score_mode: str,
    score_threshold: float,
    min_query_coverage: float,
    min_target_coverage: float,
    ignored_target_hit_count: int = 0,
) -> FoldseekResult:
    rule = _decision_rule(
        score_mode=score_mode,
        score_threshold=score_threshold,
        min_query_coverage=min_query_coverage,
        min_target_coverage=min_target_coverage,
    )
    eligible_hits = [
        hit
        for hit in hits
        if _passes_coverage(
            hit,
            min_query_coverage=min_query_coverage,
            min_target_coverage=min_target_coverage,
        )
    ]
    best_pool = eligible_hits or list(hits)
    best_hit = max(best_pool, key=lambda hit: _score_hit(hit, score_mode), default=None)
    if best_hit is None:
        return FoldseekResult(
            sample_id=sample_id,
            result="NON_BARREL",
            score=0.0,
            decision_rule=rule,
            score_mode=score_mode,
            score_threshold=score_threshold,
            min_query_coverage=min_query_coverage,
            min_target_coverage=min_target_coverage,
            hit_count=0,
            eligible_hit_count=0,
            ignored_target_hit_count=ignored_target_hit_count,
        )

    score = _score_hit(best_hit, score_mode)
    result = "BARREL" if best_hit in eligible_hits and score >= score_threshold else "NON_BARREL"
    return FoldseekResult(
        sample_id=sample_id,
        result=result,
        score=score,
        decision_rule=rule,
        score_mode=score_mode,
        score_threshold=score_threshold,
        min_query_coverage=min_query_coverage,
        min_target_coverage=min_target_coverage,
        hit_count=len(hits),
        eligible_hit_count=len(eligible_hits),
        ignored_target_hit_count=ignored_target_hit_count,
        best_target=best_hit.target,
        qlen=best_hit.qlen,
        tlen=best_hit.tlen,
        alnlen=best_hit.alnlen,
        qcov=best_hit.qcov,
        tcov=best_hit.tcov,
        qtmscore=best_hit.qtmscore,
        ttmscore=best_hit.ttmscore,
        alntmscore=best_hit.alntmscore,
        evalue=best_hit.evalue,
        bits=best_hit.bits,
    )


def summarize_hits(
    hits: Sequence[FoldseekHit],
    *,
    query_ids: Sequence[str] | None = None,
    query_aliases: Mapping[str, str] | None = None,
    target_aliases: Mapping[str, str] | None = None,
    ignore_target_ids_by_query: Mapping[str, set[str]] | None = None,
    score_mode: str = DEFAULT_SCORE_MODE,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    min_query_coverage: float = DEFAULT_MIN_QUERY_COVERAGE,
    min_target_coverage: float = DEFAULT_MIN_TARGET_COVERAGE,
) -> list[FoldseekResult]:
    if score_mode not in SUPPORTED_SCORE_MODES:
        raise ValueError(
            "score_mode must be one of: " + ", ".join(sorted(SUPPORTED_SCORE_MODES))
        )

    aliases = dict(query_aliases or {})
    target_alias_map = dict(target_aliases or {})
    ignored_targets = {
        sample_id: set(target_ids)
        for sample_id, target_ids in (ignore_target_ids_by_query or {}).items()
    }
    grouped: dict[str, list[FoldseekHit]] = defaultdict(list)
    ignored_counts: Counter[str] = Counter()
    for hit in hits:
        sample_id = aliases.get(hit.query, hit.query)
        target_id = target_alias_map.get(hit.target, hit.target)
        if target_id in ignored_targets.get(sample_id, set()):
            ignored_counts[sample_id] += 1
            continue
        grouped[sample_id].append(hit if target_id == hit.target else replace(hit, target=target_id))

    if query_ids is None:
        ordered_ids = sorted(grouped)
    else:
        ordered_ids = list(query_ids)

    return [
        _result_from_best_hit(
            sample_id,
            grouped.get(sample_id, []),
            score_mode=score_mode,
            score_threshold=score_threshold,
            min_query_coverage=min_query_coverage,
            min_target_coverage=min_target_coverage,
            ignored_target_hit_count=ignored_counts[sample_id],
        )
        for sample_id in ordered_ids
    ]


def write_results_csv(results: Iterable[FoldseekResult], output_path: str | Path) -> Path:
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "baseline",
        "sample_id",
        "result",
        "score",
        "decision_rule",
        "score_mode",
        "score_threshold",
        "min_query_coverage",
        "min_target_coverage",
        "hit_count",
        "eligible_hit_count",
        "ignored_target_hit_count",
        "best_target",
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
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.as_row())

    return output_file


def _prepare_target(
    target: str | Path,
    *,
    run_dir: Path,
    build_target_db: bool,
    create_index: bool,
    foldseek_executable: str | Path | None,
    command_prefix: Sequence[str] | None,
    timeout: float | None,
) -> Path:
    if not build_target_db:
        return _require_path(target, "Foldseek target DB or structures")

    target_path = _require_path(target, "Foldseek reference structures")
    target_db = run_dir / "targetDB"
    _run_foldseek_command(
        ["createdb", str(target_path), str(target_db)],
        work_dir=run_dir,
        foldseek_executable=foldseek_executable,
        command_prefix=command_prefix,
        timeout=timeout,
    )
    if create_index:
        index_tmp = run_dir / "target_index_tmp"
        _run_foldseek_command(
            ["createindex", str(target_db), str(index_tmp)],
            work_dir=run_dir,
            foldseek_executable=foldseek_executable,
            command_prefix=command_prefix,
            timeout=timeout,
        )
    return target_db


def _run_in_directory(
    query_structures: Path,
    target: Path,
    *,
    run_dir: Path,
    foldseek_executable: str | Path | None,
    command_prefix: Sequence[str] | None,
    alignment_type: int,
    evalue: float,
    max_seqs: int,
    extra_args: Sequence[str] | None,
    timeout: float | None,
) -> Path:
    hits_path = run_dir / "foldseek_hits.tsv"
    tmp_dir = run_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "easy-search",
        str(query_structures),
        str(target),
        str(hits_path),
        str(tmp_dir),
        "--alignment-type",
        str(alignment_type),
        "--format-output",
        ",".join(DEFAULT_FORMAT_FIELDS),
        "-e",
        str(evalue),
        "--max-seqs",
        str(max_seqs),
        *(extra_args or []),
    ]
    _run_foldseek_command(
        command,
        work_dir=run_dir,
        foldseek_executable=foldseek_executable,
        command_prefix=command_prefix,
        timeout=timeout,
    )
    return hits_path


def run_baseline(
    query_structures: str | Path,
    target: str | Path,
    *,
    foldseek_executable: str | Path | None = None,
    work_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    query_ids: Sequence[str] | None = None,
    query_aliases: Mapping[str, str] | None = None,
    target_aliases: Mapping[str, str] | None = None,
    ignore_target_ids_by_query: Mapping[str, set[str]] | None = None,
    build_target_db: bool = False,
    create_index: bool = False,
    alignment_type: int = DEFAULT_ALIGNMENT_TYPE,
    score_mode: str = DEFAULT_SCORE_MODE,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    min_query_coverage: float = DEFAULT_MIN_QUERY_COVERAGE,
    min_target_coverage: float = DEFAULT_MIN_TARGET_COVERAGE,
    evalue: float = DEFAULT_EVALUE,
    max_seqs: int = DEFAULT_MAX_SEQS,
    extra_args: Sequence[str] | None = None,
    command_prefix: Sequence[str] | None = None,
    timeout: float | None = None,
) -> list[FoldseekResult]:
    query_path = _require_path(query_structures, "Foldseek query structures")

    def run_in(run_dir: Path) -> list[FoldseekResult]:
        target_input = _prepare_target(
            target,
            run_dir=run_dir,
            build_target_db=build_target_db,
            create_index=create_index,
            foldseek_executable=foldseek_executable,
            command_prefix=command_prefix,
            timeout=timeout,
        )
        hits_path = _run_in_directory(
            query_path,
            target_input,
            run_dir=run_dir,
            foldseek_executable=foldseek_executable,
            command_prefix=command_prefix,
            alignment_type=alignment_type,
            evalue=evalue,
            max_seqs=max_seqs,
            extra_args=extra_args,
            timeout=timeout,
        )
        hits = load_hits_tsv(hits_path)
        return summarize_hits(
            hits,
            query_ids=query_ids,
            query_aliases=query_aliases,
            target_aliases=target_aliases,
            ignore_target_ids_by_query=ignore_target_ids_by_query,
            score_mode=score_mode,
            score_threshold=score_threshold,
            min_query_coverage=min_query_coverage,
            min_target_coverage=min_target_coverage,
        )

    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="cooper_beta_foldseek_") as temp_dir:
            results = run_in(Path(temp_dir))
    else:
        run_dir = Path(work_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        results = run_in(run_dir)

    if output_path is not None:
        write_results_csv(results, output_path)
    return results


def _print_summary(results: Sequence[FoldseekResult], output_path: str | Path | None) -> None:
    counts = Counter(result.result for result in results)
    print(f"Rows: {len(results)}")
    print(f"BARREL: {counts.get('BARREL', 0)}")
    print(f"NON_BARREL: {counts.get('NON_BARREL', 0)}")
    if output_path is not None:
        print(f"Output: {Path(output_path).expanduser().resolve()}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and normalize the external Foldseek global-TMalign baseline."
    )
    parser.add_argument("query_structures", help="PDB/CIF/mmCIF query file, directory, or DB.")
    parser.add_argument("target", help="Foldseek target DB or reference-structure path.")
    parser.add_argument("--foldseek", help="Foldseek executable. Defaults to FOLDSEEK_BIN or foldseek.")
    parser.add_argument("--work-dir", help="Working directory for Foldseek raw outputs.")
    parser.add_argument("--out", help="Optional normalized CSV output path.")
    parser.add_argument(
        "--build-target-db",
        action="store_true",
        help="Run foldseek createdb on the target path before searching.",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Run foldseek createindex after createdb. Used only with --build-target-db.",
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

    results = run_baseline(
        args.query_structures,
        args.target,
        foldseek_executable=args.foldseek,
        work_dir=args.work_dir,
        output_path=args.out,
        build_target_db=args.build_target_db,
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
    )
    _print_summary(results, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
