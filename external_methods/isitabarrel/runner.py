from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DECISION_COLUMN = "CC2_TO_H4"
BASELINE_NAME = "isitabarrel_structure_map"


@dataclass(frozen=True)
class IsItABarrelResult:
    sample_id: str
    result: str
    score: float
    decision_column: str
    cc2: float | None = None
    bss: float | None = None
    h4: float | None = None
    cc2_to_h4: float | None = None
    cc2_to_bss: float | None = None
    bss_to_cc2_to_bss: float | None = None

    def as_row(self) -> dict[str, object]:
        return {
            "baseline": BASELINE_NAME,
            "sample_id": self.sample_id,
            "result": self.result,
            "score": self.score,
            "decision_column": self.decision_column,
            "CC2": self.cc2,
            "BSS": self.bss,
            "H4": self.h4,
            "CC2_TO_H4": self.cc2_to_h4,
            "CC2_TO_BSS": self.cc2_to_bss,
            "BSS_TO_CC2_TO_BSS": self.bss_to_cc2_to_bss,
        }


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _require_path(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _score_to_result(score: float) -> str:
    return "BARREL" if score > 0 else "NON_BARREL"


def load_results_tsv(
    results_path: str | Path,
    decision_column: str = DEFAULT_DECISION_COLUMN,
) -> list[IsItABarrelResult]:
    results_file = _require_path(results_path, "IsItABarrel results TSV")
    rows: list[IsItABarrelResult] = []

    with results_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Empty IsItABarrel results file: {results_file}")
        if "MAP_NAME" not in reader.fieldnames:
            raise ValueError("IsItABarrel results must include a MAP_NAME column.")
        if decision_column not in reader.fieldnames:
            raise ValueError(
                f"Decision column {decision_column!r} is missing from {results_file}."
            )

        for row in reader:
            sample_id = (row.get("MAP_NAME") or "").strip()
            if not sample_id:
                continue
            score = _parse_float(row.get(decision_column))
            if score is None:
                raise ValueError(f"Missing score for {sample_id!r} in {decision_column}.")
            rows.append(
                IsItABarrelResult(
                    sample_id=sample_id,
                    result=_score_to_result(score),
                    score=score,
                    decision_column=decision_column,
                    cc2=_parse_float(row.get("CC2")),
                    bss=_parse_float(row.get("BSS")),
                    h4=_parse_float(row.get("H4")),
                    cc2_to_h4=_parse_float(row.get("CC2_TO_H4")),
                    cc2_to_bss=_parse_float(row.get("CC2_TO_BSS")),
                    bss_to_cc2_to_bss=_parse_float(row.get("BSS_TO_CC2_TO_BSS")),
                )
            )

    return rows


def write_results_csv(results: Iterable[IsItABarrelResult], output_path: str | Path) -> Path:
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "baseline",
        "sample_id",
        "result",
        "score",
        "decision_column",
        "CC2",
        "BSS",
        "H4",
        "CC2_TO_H4",
        "CC2_TO_BSS",
        "BSS_TO_CC2_TO_BSS",
    ]
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.as_row())

    return output_file


def _resolve_script(script_path: str | Path | None) -> Path:
    if script_path is None:
        script_path = os.environ.get("ISITABARREL_SCRIPT")
    if script_path is None:
        raise ValueError(
            "Provide the official isitabarrel.py with --script or ISITABARREL_SCRIPT."
        )
    return _require_path(script_path, "IsItABarrel script")


def _invoke_isitabarrel(
    script_path: Path,
    protid_list: Path,
    map_dir: Path,
    work_dir: Path,
    python_executable: str,
    extra_args: Sequence[str],
    timeout: float | None,
) -> None:
    command = [
        python_executable,
        str(script_path),
        str(protid_list),
        str(map_dir),
        *extra_args,
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
            "IsItABarrel baseline failed with exit code "
            f"{completed.returncode}.\nstdout:\n{stdout[-2000:]}\nstderr:\n{stderr[-2000:]}"
        )


def run_baseline(
    protid_list: str | Path,
    map_dir: str | Path,
    *,
    script_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    decision_column: str = DEFAULT_DECISION_COLUMN,
    python_executable: str | None = None,
    extra_args: Sequence[str] | None = None,
    timeout: float | None = None,
) -> list[IsItABarrelResult]:
    protid_file = _require_path(protid_list, "Protein-id list")
    maps = _require_path(map_dir, "Contact-map directory")
    if not maps.is_dir():
        raise NotADirectoryError(f"Contact-map path is not a directory: {maps}")

    script = _resolve_script(script_path)
    py_exec = python_executable or sys.executable
    args = list(extra_args or [])

    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="cooper_beta_isitabarrel_") as temp_dir:
            run_dir = Path(temp_dir)
            _invoke_isitabarrel(script, protid_file, maps, run_dir, py_exec, args, timeout)
            results = load_results_tsv(run_dir / "results.tsv", decision_column)
    else:
        run_dir = Path(work_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        _invoke_isitabarrel(script, protid_file, maps, run_dir, py_exec, args, timeout)
        results = load_results_tsv(run_dir / "results.tsv", decision_column)

    if output_path is not None:
        write_results_csv(results, output_path)
    return results


def _print_summary(results: Sequence[IsItABarrelResult], output_path: str | Path | None) -> None:
    counts = Counter(result.result for result in results)
    print(f"Rows: {len(results)}")
    print(f"BARREL: {counts.get('BARREL', 0)}")
    print(f"NON_BARREL: {counts.get('NON_BARREL', 0)}")
    if output_path is not None:
        print(f"Output: {Path(output_path).expanduser().resolve()}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and normalize the external IsItABarrel baseline."
    )
    parser.add_argument("protid_list", help="File with one protein id per line.")
    parser.add_argument("map_dir", help="Directory with <protein-id>.pkl contact maps.")
    parser.add_argument(
        "--script",
        help="Path to the official isitabarrel.py script. "
        "Defaults to the ISITABARREL_SCRIPT environment variable.",
    )
    parser.add_argument(
        "--work-dir",
        help="Working directory for the upstream script. Defaults to a temporary directory.",
    )
    parser.add_argument("--out", help="Optional normalized CSV output path.")
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

    results = run_baseline(
        args.protid_list,
        args.map_dir,
        script_path=args.script,
        work_dir=args.work_dir,
        output_path=args.out,
        decision_column=args.decision_column,
        python_executable=args.python,
        extra_args=extra_args,
        timeout=args.timeout,
    )
    _print_summary(results, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
