from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from external_methods.isitabarrel.contact_maps import (
    DEFAULT_CA_CUTOFF,
    DEFAULT_LOCAL_EXCLUSION,
    DEFAULT_MIN_RESIDUES,
    GeneratedContactMapSet,
    generate_structure_contact_maps,
)
from external_methods.isitabarrel.runner import (
    DEFAULT_DECISION_COLUMN,
    IsItABarrelResult,
    run_baseline,
)


@dataclass(frozen=True)
class StructureMapBaselineRun:
    generated_maps: GeneratedContactMapSet
    results: list[IsItABarrelResult]
    output_path: str | None = None


def run_structure_map_baseline(
    structure_input: str | Path,
    output_dir: str | Path,
    *,
    script_path: str | Path | None = None,
    output_path: str | Path | None = None,
    cutoff: float = DEFAULT_CA_CUTOFF,
    local_exclusion: int = DEFAULT_LOCAL_EXCLUSION,
    min_residues: int = DEFAULT_MIN_RESIDUES,
    decision_column: str = DEFAULT_DECISION_COLUMN,
    python_executable: str | None = None,
    extra_args: Sequence[str] | None = None,
    timeout: float | None = None,
) -> StructureMapBaselineRun:
    output = Path(output_dir).expanduser().resolve()
    generated = generate_structure_contact_maps(
        structure_input,
        output,
        cutoff=cutoff,
        local_exclusion=local_exclusion,
        min_residues=min_residues,
    )
    results = run_baseline(
        generated.protid_list_path,
        generated.map_dir,
        script_path=script_path,
        work_dir=output / "isitabarrel_work",
        output_path=output_path,
        decision_column=decision_column,
        python_executable=python_executable,
        extra_args=extra_args,
        timeout=timeout,
    )
    return StructureMapBaselineRun(
        generated_maps=generated,
        results=results,
        output_path=str(Path(output_path).expanduser().resolve()) if output_path else None,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run IsItABarrel on structure-derived contact maps."
    )
    parser.add_argument("structure_input", help="PDB/CIF/mmCIF file or directory.")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Working directory for generated maps, metadata, and upstream output.",
    )
    parser.add_argument(
        "--script",
        help="Path to the official isitabarrel.py script. "
        "Defaults to the ISITABARREL_SCRIPT environment variable.",
    )
    parser.add_argument("--out", help="Optional normalized CSV output path.")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_CA_CUTOFF,
        help=f"CA-CA contact cutoff in Angstrom. Default: {DEFAULT_CA_CUTOFF}.",
    )
    parser.add_argument(
        "--local-exclusion",
        type=int,
        default=DEFAULT_LOCAL_EXCLUSION,
        help=(
            "Zero contacts where sequence distance is <= this value. "
            f"Default: {DEFAULT_LOCAL_EXCLUSION}."
        ),
    )
    parser.add_argument(
        "--min-residues",
        type=int,
        default=DEFAULT_MIN_RESIDUES,
        help=(
            "Minimum CA residue count required for a chain. "
            f"Default: {DEFAULT_MIN_RESIDUES}, which avoids an upstream "
            "IsItABarrel indexing failure on very short chains."
        ),
    )
    parser.add_argument(
        "--decision-column",
        default=DEFAULT_DECISION_COLUMN,
        help=f"Score column used for BARREL/NON_BARREL decisions. Default: {DEFAULT_DECISION_COLUMN}.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the upstream script.",
    )
    parser.add_argument("--timeout", type=float, help="Optional subprocess timeout in seconds.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args, extra_args = parser.parse_known_args(argv)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    run = run_structure_map_baseline(
        args.structure_input,
        args.out_dir,
        script_path=args.script,
        output_path=args.out,
        cutoff=args.cutoff,
        local_exclusion=args.local_exclusion,
        min_residues=args.min_residues,
        decision_column=args.decision_column,
        python_executable=args.python,
        extra_args=extra_args,
        timeout=args.timeout,
    )

    barrel_count = sum(result.result == "BARREL" for result in run.results)
    print(f"Generated maps: {len(run.generated_maps.records)}")
    print(f"Rows: {len(run.results)}")
    print(f"BARREL: {barrel_count}")
    print(f"NON_BARREL: {len(run.results) - barrel_count}")
    if run.output_path:
        print(f"Output: {run.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
