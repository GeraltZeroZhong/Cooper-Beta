#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

BLAST_FIELDS = [
    "qseqid",
    "saccver",
    "pident",
    "length",
    "qcovs",
    "evalue",
    "bitscore",
    "sscinames",
    "sskingdoms",
    "stitle",
]

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",
    "GLX": "Z",
    "SEC": "U",
    "PYL": "O",
    "MSE": "M",
    "UNK": "X",
}

LOW_INFORMATION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bhypothetical protein\b",
        r"\buncharacteri[sz]ed protein\b",
        r"\bputative uncharacteri[sz]ed protein\b",
        r"\bprotein of unknown function\b",
        r"\bdomain of unknown function\b",
        r"\bunknown protein\b",
        r"\bunknown function\b",
        r"\bunnamed protein product\b",
        r"\bpredicted protein\b",
        r"\bconserved hypothetical protein\b",
        r"\bUPF\d+\b",
        r"\bDUF\d+\b",
    ]
]


@dataclass
class Candidate:
    query_id: str
    filename: str
    chain: str
    result: str
    reason: str
    decision_score: str
    source_path: str
    sequence_length: int
    sequence_status: str


@dataclass
class BlastHit:
    qseqid: str
    saccver: str
    pident: float
    length: int
    qcovs: float
    evalue: float
    bitscore: float
    sscinames: str
    sskingdoms: str
    stitle: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate Cooper-Beta BFVD candidate proteins with blastp and produce "
            "candidate-level annotation tables plus concrete summary counts."
        )
    )
    parser.add_argument(
        "--results",
        default="eval_outputs/bfvd_full_20260425_040946/results.csv",
        help="Cooper-Beta results CSV.",
    )
    parser.add_argument(
        "--structures",
        default="BFVD",
        help="Directory containing BFVD PDB files.",
    )
    parser.add_argument(
        "--out-dir",
        default="eval_outputs/bfvd_blastp_annotation",
        help="Output directory for FASTA, BLAST TSV, annotation CSV, and summaries.",
    )
    parser.add_argument(
        "--result",
        default="BARREL",
        help='Cooper-Beta result value to keep (default: "BARREL").',
    )
    parser.add_argument(
        "--reason",
        default="OK",
        help='Cooper-Beta reason value to keep (default: "OK"). Use "" to disable.',
    )
    parser.add_argument(
        "--min-query-length",
        type=int,
        default=20,
        help="Minimum extracted protein length to write to FASTA.",
    )
    parser.add_argument(
        "--no-recursive-search",
        action="store_true",
        help="Disable recursive basename search when a structure is not found directly.",
    )
    parser.add_argument(
        "--blastp",
        default="blastp",
        help="Path to blastp executable.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Local BLAST protein database name/path. Required unless --remote or --blast-tsv is used.",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Run NCBI remote blastp instead of a local database. Slower; omit --threads.",
    )
    parser.add_argument(
        "--blast-tsv",
        default=None,
        help="Existing BLAST outfmt 6 TSV to parse instead of running blastp.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="blastp worker threads for local searches.",
    )
    parser.add_argument(
        "--max-target-seqs",
        type=int,
        default=10,
        help="Maximum reported BLAST hits per query.",
    )
    parser.add_argument(
        "--search-evalue",
        default="1e-5",
        help="blastp -evalue used while searching.",
    )
    parser.add_argument(
        "--hit-evalue",
        type=float,
        default=1e-5,
        help="Maximum e-value for a hit to count as an annotation.",
    )
    parser.add_argument(
        "--min-pident",
        type=float,
        default=25.0,
        help="Minimum percent identity for a hit to count as an annotation.",
    )
    parser.add_argument(
        "--min-qcov",
        type=float,
        default=50.0,
        help="Minimum query coverage percentage for a hit to count as an annotation.",
    )
    parser.add_argument(
        "--entrez-query",
        default=None,
        help='Optional BLAST Entrez query, e.g. "Viruses[Organism]".',
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing BLAST TSV by rerunning blastp.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stop after candidate filtering and FASTA extraction; do not call blastp.",
    )
    return parser.parse_args(argv)


def sanitize_query_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip())
    return cleaned.strip("_") or "query"


def percent(part: int, whole: int) -> str:
    if whole == 0:
        return "0.00%"
    return f"{part / whole * 100:.2f}%"


def read_candidate_rows(results_path: Path, result: str, reason: str) -> tuple[list[dict[str, str]], int]:
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV does not exist: {results_path}")

    rows: list[dict[str, str]] = []
    with results_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "chain", "result", "reason"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"Results CSV is missing required column(s): {joined}")

        total_rows = 0
        for row in reader:
            total_rows += 1
            if str(row.get("result", "")).upper() != result.upper():
                continue
            if reason and str(row.get("reason", "")) != reason:
                continue
            rows.append({key: value or "" for key, value in row.items()})

    return rows, total_rows


def resolve_structure_path(
    structures_dir: Path,
    filename: str,
    recursive: bool,
    cache: dict[str, Path | None],
) -> Path | None:
    direct = structures_dir / filename
    if direct.exists():
        return direct

    basename = Path(filename).name
    fallback = structures_dir / basename
    if fallback.exists():
        return fallback

    if not recursive:
        return None

    if basename not in cache:
        matches = list(structures_dir.rglob(basename))
        cache[basename] = matches[0] if matches else None
    return cache[basename]


def extract_pdb_chain_sequence(path: Path, chain: str) -> str:
    residues: list[str] = []
    seen_residues: set[tuple[str, str, str]] = set()
    target_chain = chain.strip()

    with path.open(errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "ENDMDL":
                break
            if record != "ATOM":
                continue
            if len(line) < 27:
                continue

            line_chain = line[21].strip()
            if target_chain and line_chain != target_chain:
                continue

            residue_key = (line_chain, line[22:26].strip(), line[26].strip())
            if residue_key in seen_residues:
                continue
            seen_residues.add(residue_key)

            residue_name = line[17:20].strip().upper()
            residues.append(AA3_TO_1.get(residue_name, "X"))

    return "".join(residues)


def build_candidates(
    selected_rows: list[dict[str, str]],
    structures_dir: Path,
    min_query_length: int,
    recursive: bool,
) -> tuple[list[Candidate], dict[str, str], int]:
    path_cache: dict[str, Path | None] = {}
    query_id_counts: Counter[str] = Counter()
    sequences: dict[str, str] = {}
    candidates: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    duplicates = 0

    for row in selected_rows:
        filename = row["filename"].strip()
        chain = row.get("chain", "").strip()
        key = (filename, chain)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)

        base_query_id = sanitize_query_id(f"{Path(filename).stem}__chain_{chain or 'blank'}")
        query_id_counts[base_query_id] += 1
        query_id = base_query_id
        if query_id_counts[base_query_id] > 1:
            query_id = f"{base_query_id}__{query_id_counts[base_query_id]}"

        source_path = resolve_structure_path(structures_dir, filename, recursive, path_cache)
        sequence = ""
        status = "missing_structure"
        if source_path is not None:
            sequence = extract_pdb_chain_sequence(source_path, chain)
            if len(sequence) < min_query_length:
                status = "too_short"
            else:
                status = "ok"
                sequences[query_id] = sequence

        candidates.append(
            Candidate(
                query_id=query_id,
                filename=filename,
                chain=chain,
                result=row.get("result", ""),
                reason=row.get("reason", ""),
                decision_score=row.get("decision_score", ""),
                source_path=str(source_path or ""),
                sequence_length=len(sequence),
                sequence_status=status,
            )
        )

    return candidates, sequences, duplicates


def write_fasta(sequences: dict[str, str], path: Path) -> None:
    with path.open("w") as handle:
        for query_id, sequence in sequences.items():
            handle.write(f">{query_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def write_candidate_manifest(candidates: list[Candidate], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(candidates[0]).keys()) if candidates else [])
        if candidates:
            writer.writeheader()
            for candidate in candidates:
                writer.writerow(asdict(candidate))


def run_blastp(args: argparse.Namespace, fasta_path: Path, blast_tsv: Path) -> list[str]:
    blastp_path = shutil.which(args.blastp) or args.blastp
    if shutil.which(blastp_path) is None and not Path(blastp_path).exists():
        raise FileNotFoundError(
            f"blastp executable was not found: {args.blastp}. Install BLAST+ or pass --blastp."
        )

    command = [
        blastp_path,
        "-query",
        str(fasta_path),
        "-out",
        str(blast_tsv),
        "-outfmt",
        "6 " + " ".join(BLAST_FIELDS),
        "-evalue",
        str(args.search_evalue),
        "-max_target_seqs",
        str(args.max_target_seqs),
    ]
    if args.remote:
        command.append("-remote")
    else:
        if not args.db:
            raise ValueError("Pass --db for a local BLAST search, or use --remote/--blast-tsv.")
        command.extend(["-db", str(args.db)])
        if args.threads > 0:
            command.extend(["-num_threads", str(args.threads)])

    if args.entrez_query:
        command.extend(["-entrez_query", args.entrez_query])

    env = os.environ.copy()
    if args.db and not args.remote:
        db_parent = str(Path(args.db).expanduser().resolve().parent)
        existing_blastdb = env.get("BLASTDB")
        env["BLASTDB"] = (
            db_parent if not existing_blastdb else f"{db_parent}{os.pathsep}{existing_blastdb}"
        )

    subprocess.run(command, check=True, env=env)
    return command


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except ValueError:
        return default


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except ValueError:
        return default


def read_blast_hits(path: Path) -> dict[str, list[BlastHit]]:
    hits: dict[str, list[BlastHit]] = defaultdict(list)
    if not path.exists():
        raise FileNotFoundError(f"BLAST TSV does not exist: {path}")

    with path.open(errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(BLAST_FIELDS):
                parts.extend([""] * (len(BLAST_FIELDS) - len(parts)))
            row = dict(zip(BLAST_FIELDS, parts[: len(BLAST_FIELDS)], strict=False))
            hits[row["qseqid"]].append(
                BlastHit(
                    qseqid=row["qseqid"],
                    saccver=row["saccver"],
                    pident=parse_float(row["pident"]),
                    length=parse_int(row["length"]),
                    qcovs=parse_float(row["qcovs"]),
                    evalue=parse_float(row["evalue"], default=float("inf")),
                    bitscore=parse_float(row["bitscore"]),
                    sscinames=row["sscinames"],
                    sskingdoms=row["sskingdoms"],
                    stitle=row["stitle"],
                )
            )

    for query_hits in hits.values():
        query_hits.sort(key=lambda hit: (-hit.bitscore, hit.evalue, -hit.pident, -hit.qcovs))
    return hits


def hit_passes(hit: BlastHit, max_evalue: float, min_pident: float, min_qcov: float) -> bool:
    return hit.evalue <= max_evalue and hit.pident >= min_pident and hit.qcovs >= min_qcov


def is_low_information_title(title: str) -> bool:
    return any(pattern.search(title) for pattern in LOW_INFORMATION_PATTERNS)


def clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"^(?:sp|tr|ref|gb|emb|dbj|pir|prf)\|[^ ]+\s+", "", title)
    return title


def annotate_candidates(
    candidates: list[Candidate],
    hits_by_query: dict[str, list[BlastHit]],
    args: argparse.Namespace,
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    kingdom_counts: Counter[str] = Counter()
    species_counts: Counter[str] = Counter()
    title_counts: Counter[str] = Counter()
    queries_with_any_hit = 0
    queries_with_passing_hit = 0

    for candidate in candidates:
        hits = hits_by_query.get(candidate.query_id, [])
        if hits:
            queries_with_any_hit += 1

        passing_hits = [
            hit
            for hit in hits
            if hit_passes(hit, args.hit_evalue, args.min_pident, args.min_qcov)
        ]
        top_hit = passing_hits[0] if passing_hits else (hits[0] if hits else None)
        hit_status = "no_blast_hit"
        annotation_label = ""
        low_information = ""

        if candidate.sequence_status != "ok":
            hit_status = candidate.sequence_status
        elif passing_hits:
            queries_with_passing_hit += 1
            annotation_label = clean_title(top_hit.stitle)
            low_information = str(is_low_information_title(top_hit.stitle))
            hit_status = "low_information_hit" if low_information == "True" else "informative_hit"
            if top_hit.sskingdoms:
                kingdom_counts[top_hit.sskingdoms] += 1
            if top_hit.sscinames:
                species_counts[top_hit.sscinames] += 1
            if annotation_label:
                title_counts[annotation_label] += 1
        elif hits:
            hit_status = "no_passing_hit"

        status_counts[hit_status] += 1
        rows.append(
            {
                "query_id": candidate.query_id,
                "filename": candidate.filename,
                "chain": candidate.chain,
                "sequence_length": candidate.sequence_length,
                "sequence_status": candidate.sequence_status,
                "cooper_beta_score": candidate.decision_score,
                "annotation_status": hit_status,
                "annotation_label": annotation_label,
                "low_information_title": low_information,
                "top_saccver": top_hit.saccver if top_hit else "",
                "top_pident": top_hit.pident if top_hit else "",
                "top_qcovs": top_hit.qcovs if top_hit else "",
                "top_evalue": top_hit.evalue if top_hit else "",
                "top_bitscore": top_hit.bitscore if top_hit else "",
                "top_species": top_hit.sscinames if top_hit else "",
                "top_kingdom": top_hit.sskingdoms if top_hit else "",
                "top_title": top_hit.stitle if top_hit else "",
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "annotation_status_counts": dict(status_counts),
        "queries_with_any_blast_hit": queries_with_any_hit,
        "queries_with_passing_hit": queries_with_passing_hit,
        "top_kingdoms": kingdom_counts.most_common(20),
        "top_species": species_counts.most_common(20),
        "top_annotation_labels": title_counts.most_common(30),
    }
    return rows, summary


def sequence_length_stats(candidates: list[Candidate]) -> dict[str, Any]:
    lengths = [candidate.sequence_length for candidate in candidates if candidate.sequence_status == "ok"]
    if not lengths:
        return {"count": 0}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(mean(lengths), 2),
        "median": round(median(lengths), 2),
    }


def build_base_summary(
    args: argparse.Namespace,
    total_input_rows: int,
    selected_rows: list[dict[str, str]],
    candidates: list[Candidate],
    sequences: dict[str, str],
    duplicates: int,
    output_paths: dict[str, str],
) -> dict[str, Any]:
    sequence_status_counts = Counter(candidate.sequence_status for candidate in candidates)
    return {
        "results_csv": str(Path(args.results)),
        "structures_dir": str(Path(args.structures)),
        "result_filter": args.result,
        "reason_filter": args.reason,
        "total_input_rows": total_input_rows,
        "candidate_rows_selected": len(selected_rows),
        "unique_candidates": len(candidates),
        "duplicates_collapsed": duplicates,
        "fasta_sequences_written": len(sequences),
        "sequence_status_counts": dict(sequence_status_counts),
        "sequence_length_stats": sequence_length_stats(candidates),
        "blast_fields": BLAST_FIELDS,
        "hit_thresholds": {
            "max_evalue": args.hit_evalue,
            "min_pident": args.min_pident,
            "min_qcov": args.min_qcov,
        },
        "output_paths": output_paths,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    total = int(summary.get("unique_candidates", 0))
    status_counts = summary.get("annotation_status_counts", {})
    sequence_counts = summary.get("sequence_status_counts", {})
    lines = [
        "# BFVD Candidate BLASTP Annotation Summary",
        "",
        f"- Candidate rows selected: {summary.get('candidate_rows_selected', 0)}",
        f"- Unique candidate proteins: {total}",
        f"- FASTA sequences written: {summary.get('fasta_sequences_written', 0)}",
        f"- Duplicate rows collapsed: {summary.get('duplicates_collapsed', 0)}",
        f"- Sequence status counts: {sequence_counts}",
        f"- Sequence length stats: {summary.get('sequence_length_stats', {})}",
        "",
        "## Annotation Counts",
        "",
        "| Category | Count | Percent of unique candidates |",
        "|---|---:|---:|",
    ]
    if status_counts:
        for status, count in sorted(status_counts.items()):
            lines.append(f"| {status} | {count} | {percent(int(count), total)} |")
    else:
        lines.append("| not_run | 0 | 0.00% |")

    lines.extend(
        [
            "",
            "## BLAST Hit Counts",
            "",
            f"- Queries with any BLAST hit: {summary.get('queries_with_any_blast_hit', 'not_run')}",
            f"- Queries with passing BLAST hit: {summary.get('queries_with_passing_hit', 'not_run')}",
            f"- Hit thresholds: {summary.get('hit_thresholds', {})}",
        ]
    )

    if summary.get("top_kingdoms"):
        lines.extend(["", "## Top Kingdoms", ""])
        for kingdom, count in summary["top_kingdoms"]:
            lines.append(f"- {kingdom}: {count}")

    if summary.get("top_species"):
        lines.extend(["", "## Top Species", ""])
        for species, count in summary["top_species"][:10]:
            lines.append(f"- {species}: {count}")

    if summary.get("top_annotation_labels"):
        lines.extend(["", "## Top Annotation Labels", ""])
        for label, count in summary["top_annotation_labels"][:15]:
            lines.append(f"- {label}: {count}")

    path.write_text("\n".join(lines) + "\n")


def print_key_numbers(summary: dict[str, Any]) -> None:
    print("[OK] Candidate rows selected:", summary.get("candidate_rows_selected", 0))
    print("[OK] Unique candidate proteins:", summary.get("unique_candidates", 0))
    print("[OK] FASTA sequences written:", summary.get("fasta_sequences_written", 0))
    print("[OK] Sequence status counts:", summary.get("sequence_status_counts", {}))
    if "annotation_status_counts" in summary:
        print("[OK] Annotation status counts:", summary["annotation_status_counts"])
        print("[OK] Queries with any BLAST hit:", summary.get("queries_with_any_blast_hit", 0))
        print("[OK] Queries with passing BLAST hit:", summary.get("queries_with_passing_hit", 0))
    print("[OK] Summary JSON:", summary["output_paths"]["summary_json"])
    print("[OK] Summary Markdown:", summary["output_paths"]["summary_md"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_path = Path(args.results)
    structures_dir = Path(args.structures)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_csv = out_dir / "candidate_manifest.csv"
    fasta_path = out_dir / "candidate_sequences.faa"
    blast_tsv = Path(args.blast_tsv) if args.blast_tsv else out_dir / "blastp.tsv"
    annotation_csv = out_dir / "blastp_annotations.csv"
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"

    selected_rows, total_input_rows = read_candidate_rows(results_path, args.result, args.reason)
    candidates, sequences, duplicates = build_candidates(
        selected_rows=selected_rows,
        structures_dir=structures_dir,
        min_query_length=args.min_query_length,
        recursive=not args.no_recursive_search,
    )

    if candidates:
        write_candidate_manifest(candidates, candidates_csv)
    else:
        candidates_csv.write_text("")
    write_fasta(sequences, fasta_path)

    output_paths = {
        "candidate_manifest": str(candidates_csv),
        "fasta": str(fasta_path),
        "blast_tsv": str(blast_tsv),
        "annotation_csv": str(annotation_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }
    summary = build_base_summary(
        args=args,
        total_input_rows=total_input_rows,
        selected_rows=selected_rows,
        candidates=candidates,
        sequences=sequences,
        duplicates=duplicates,
        output_paths=output_paths,
    )

    if args.dry_run:
        summary["blast_status"] = "not_run_dry_run"
        write_json(summary_json, summary)
        write_markdown_summary(summary_md, summary)
        print_key_numbers(summary)
        return 0

    if not sequences:
        raise ValueError("No FASTA sequences were written; cannot run or parse BLAST annotations.")

    blast_command: list[str] | None = None
    if args.blast_tsv:
        summary["blast_status"] = "parsed_existing_tsv"
    elif blast_tsv.exists() and not args.force:
        summary["blast_status"] = "reused_existing_tsv"
    else:
        blast_command = run_blastp(args, fasta_path, blast_tsv)
        summary["blast_status"] = "ran_blastp"
        summary["blast_command"] = blast_command

    hits_by_query = read_blast_hits(blast_tsv)
    _, annotation_summary = annotate_candidates(candidates, hits_by_query, args, annotation_csv)
    summary.update(annotation_summary)
    write_json(summary_json, summary)
    write_markdown_summary(summary_md, summary)
    print_key_numbers(summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
