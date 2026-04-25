from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from external_methods.foldseek.runner import (
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
    GeneratedStructureSet,
    foldseek_query_aliases,
    generate_structure_chains,
)


@dataclass(frozen=True)
class StructureSearchBaselineRun:
    generated_chains: GeneratedStructureSet
    results: list[FoldseekResult]
    output_path: str | None = None


def run_structure_search_baseline(
    structure_input: str | Path,
    output_dir: str | Path,
    *,
    target_db: str | Path | None = None,
    reference_structures: str | Path | None = None,
    foldseek_executable: str | Path | None = None,
    output_path: str | Path | None = None,
    min_residues: int = DEFAULT_MIN_RESIDUES,
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
) -> StructureSearchBaselineRun:
    if (target_db is None) == (reference_structures is None):
        raise ValueError("Provide exactly one of target_db or reference_structures.")

    output = Path(output_dir).expanduser().resolve()
    generated = generate_structure_chains(
        structure_input,
        output / "query_chains",
        min_residues=min_residues,
    )
    target = target_db if target_db is not None else reference_structures
    assert target is not None

    results = run_baseline(
        generated.chain_dir,
        target,
        foldseek_executable=foldseek_executable,
        work_dir=output / "foldseek_work",
        output_path=output_path,
        query_ids=[record.sample_id for record in generated.records],
        query_aliases=foldseek_query_aliases(generated.records),
        build_target_db=reference_structures is not None,
        create_index=create_index,
        alignment_type=alignment_type,
        score_mode=score_mode,
        score_threshold=score_threshold,
        min_query_coverage=min_query_coverage,
        min_target_coverage=min_target_coverage,
        evalue=evalue,
        max_seqs=max_seqs,
        extra_args=extra_args,
        command_prefix=command_prefix,
        timeout=timeout,
    )
    return StructureSearchBaselineRun(
        generated_chains=generated,
        results=results,
        output_path=str(Path(output_path).expanduser().resolve()) if output_path else None,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Foldseek global-TMalign on chain structures from PDB/CIF/mmCIF inputs."
    )
    parser.add_argument("structure_input", help="PDB/CIF/mmCIF query file or directory.")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Working directory for generated chains, metadata, and Foldseek output.",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--target-db", help="Prebuilt Foldseek target database prefix.")
    target_group.add_argument(
        "--reference-structures",
        help="Reference PDB/CIF/mmCIF file or directory to convert with foldseek createdb.",
    )
    parser.add_argument("--foldseek", help="Foldseek executable. Defaults to FOLDSEEK_BIN or foldseek.")
    parser.add_argument("--out", help="Optional normalized CSV output path.")
    parser.add_argument(
        "--min-residues",
        type=int,
        default=DEFAULT_MIN_RESIDUES,
        help=f"Minimum CA residue count required for a chain. Default: {DEFAULT_MIN_RESIDUES}.",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Run foldseek createindex for --reference-structures before searching.",
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

    run = run_structure_search_baseline(
        args.structure_input,
        args.out_dir,
        target_db=args.target_db,
        reference_structures=args.reference_structures,
        foldseek_executable=args.foldseek,
        output_path=args.out,
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
    )

    barrel_count = sum(result.result == "BARREL" for result in run.results)
    print(f"Generated chains: {len(run.generated_chains.records)}")
    print(f"Rows: {len(run.results)}")
    print(f"BARREL: {barrel_count}")
    print(f"NON_BARREL: {len(run.results) - barrel_count}")
    if run.output_path:
        print(f"Output: {run.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
