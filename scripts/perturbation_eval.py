#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import shutil
import sys
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFIO, PDBIO, MMCIFParser, PDBParser

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}
DEFAULT_NOISE_SIGMAS = "0,0.25,0.5,1.0,1.5,2.0"
DEFAULT_SLICE_STEPS = "0.5,0.75,1.0,1.25,1.5,2.0"
DEFAULT_NOISE_SEEDS = "0"


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _level_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _discover_structures(folder: Path) -> list[Path]:
    folder = folder.resolve()
    if not folder.exists():
        raise FileNotFoundError(str(folder))
    if folder.is_file():
        return [folder]
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in STRUCTURE_SUFFIXES
    )


def _copy_selected_structures(source_dir: Path, output_dir: Path, limit: int | None) -> None:
    files = _discover_structures(source_dir)
    if limit is not None:
        files = files[: max(0, int(limit))]
    if not files:
        raise ValueError(f"No structure files found in {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = source_dir.resolve()
    for index, source in enumerate(files):
        if source_dir.is_file():
            relative = source.name
        else:
            try:
                relative = str(source.resolve().relative_to(source_dir))
            except ValueError:
                relative = source.name
        destination = output_dir / relative
        if destination.exists():
            destination = output_dir / f"{index:05d}_{source.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


@contextlib.contextmanager
def _limited_input_dirs(
    positive_dir: Path,
    negative_dir: Path,
    max_files_per_split: int | None,
) -> Iterator[tuple[Path, Path]]:
    if max_files_per_split is None:
        yield positive_dir, negative_dir
        return

    with tempfile.TemporaryDirectory(prefix="cooper_beta_perturb_subset_") as temp_dir:
        temp_path = Path(temp_dir)
        subset_positive = temp_path / "positive"
        subset_negative = temp_path / "negative"
        _copy_selected_structures(positive_dir, subset_positive, max_files_per_split)
        _copy_selected_structures(negative_dir, subset_negative, max_files_per_split)
        yield subset_positive, subset_negative


def _read_structure(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))


def _stable_seed(base_seed: int, relative_name: str) -> int:
    payload = f"{int(base_seed)}\0{relative_name}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def _perturb_structure_file(
    source: Path,
    destination: Path,
    *,
    sigma: float,
    seed: int,
    relative_name: str,
    atoms: str,
) -> None:
    structure = _read_structure(source)
    rng = np.random.default_rng(_stable_seed(seed, relative_name))

    for atom in structure.get_atoms():
        if atoms == "ca" and atom.get_name().strip().upper() != "CA":
            continue
        atom.set_coord(atom.get_coord() + rng.normal(loc=0.0, scale=float(sigma), size=3))

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = MMCIFIO() if destination.suffix.lower() in {".cif", ".mmcif"} else PDBIO()
    writer.set_structure(structure)
    writer.save(str(destination))


def _write_perturbed_split(
    source_dir: Path,
    output_dir: Path,
    *,
    sigma: float,
    seed: int,
    atoms: str,
) -> None:
    source_dir = source_dir.resolve()
    files = _discover_structures(source_dir)
    if not files:
        raise ValueError(f"No structure files found in {source_dir}")

    for index, source in enumerate(files):
        if source_dir.is_file():
            relative = source.name
        else:
            try:
                relative = str(source.resolve().relative_to(source_dir))
            except ValueError:
                relative = source.name
        output_suffix = ".cif" if source.suffix.lower() in {".cif", ".mmcif"} else ".pdb"
        destination = output_dir / Path(relative).with_suffix(output_suffix)
        if destination.exists():
            destination = output_dir / f"{index:05d}_{source.stem}{output_suffix}"
        _perturb_structure_file(
            source,
            destination,
            sigma=sigma,
            seed=seed,
            relative_name=relative,
            atoms=atoms,
        )


@contextlib.contextmanager
def _noise_input_dirs(
    positive_dir: Path,
    negative_dir: Path,
    *,
    sigma: float,
    seed: int,
    atoms: str,
    keep_root: Path | None,
    experiment_name: str,
) -> Iterator[tuple[Path, Path]]:
    if float(sigma) == 0.0:
        yield positive_dir, negative_dir
        return

    if keep_root is not None:
        root = keep_root / experiment_name
        root.mkdir(parents=True, exist_ok=True)
        cleanup = contextlib.nullcontext(root)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="cooper_beta_perturb_noise_")

    with cleanup as temp_dir:
        root = Path(temp_dir)
        perturbed_positive = root / "positive"
        perturbed_negative = root / "negative"
        _write_perturbed_split(
            positive_dir,
            perturbed_positive,
            sigma=sigma,
            seed=seed,
            atoms=atoms,
        )
        _write_perturbed_split(
            negative_dir,
            perturbed_negative,
            sigma=sigma,
            seed=seed,
            atoms=atoms,
        )
        yield perturbed_positive, perturbed_negative


def _summarize_row(row: dict[str, object]) -> str:
    parts: list[str] = []
    if row.get("chain_f1") is not None:
        parts.append(
            "chain: "
            f"R={row['chain_recall']:.4f} "
            f"P={row['chain_precision']:.4f} "
            f"F1={row['chain_f1']:.4f} "
            f"MCC={row['chain_mcc']:.4f}"
        )
    if row.get("file_f1") is not None:
        parts.append(
            "file: "
            f"R={row['file_recall']:.4f} "
            f"P={row['file_precision']:.4f} "
            f"F1={row['file_f1']:.4f} "
            f"MCC={row['file_mcc']:.4f}"
        )
    return " | ".join(parts) if parts else "metrics unavailable"


def _ordered_summary(dataframe):
    preferred_columns = [
        "exp",
        "perturbation_mode",
        "coordinate_noise_sigma",
        "coordinate_noise_seed",
        "coordinate_noise_atoms",
        "slicer_step_size",
        "chain_recall",
        "chain_precision",
        "chain_specificity",
        "chain_f1",
        "chain_balanced_accuracy",
        "chain_mcc",
        "chain_TP",
        "chain_FP",
        "chain_TN",
        "chain_FN",
        "file_recall",
        "file_precision",
        "file_specificity",
        "file_f1",
        "file_balanced_accuracy",
        "file_mcc",
        "file_TP",
        "file_FP",
        "file_TN",
        "file_FN",
        "chain_accuracy",
        "file_accuracy",
        "chain_csv",
        "file_csv",
    ]
    ordered_columns = [column for column in preferred_columns if column in dataframe.columns]
    ordered_columns += [column for column in dataframe.columns if column not in ordered_columns]
    return dataframe[ordered_columns]


def run_perturbation_suite(
    *,
    positive_dir: Path,
    negative_dir: Path,
    workers: int | None,
    prepare_workers: int | None,
    save_dir: Path,
    metric_level: str,
    modes: str,
    noise_sigmas: list[float],
    noise_seeds: list[int],
    step_sizes: list[float],
    noise_atoms: str,
    max_files_per_split: int | None,
    keep_perturbed_structures: bool,
) -> Path | None:
    from cooper_beta.evaluation.runner import evaluate

    if pd is None:
        raise RuntimeError("pandas is required (pip install 'cooper-beta[eval]').")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = save_dir / f"perturbation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_root = output_dir / "perturbed_structures" if keep_perturbed_structures else None

    rows: list[dict[str, object]] = []
    print("\n=== Perturbation suite ===")
    print(f"Output dir: {output_dir}\n")

    with _limited_input_dirs(positive_dir, negative_dir, max_files_per_split) as (
        base_positive_dir,
        base_negative_dir,
    ):
        if modes in {"noise", "both"}:
            for seed in noise_seeds:
                for sigma in noise_sigmas:
                    exp = f"noise_sigma_{_level_token(sigma)}_seed_{seed}"
                    print(f"[{exp}] atoms={noise_atoms}")
                    with _noise_input_dirs(
                        base_positive_dir,
                        base_negative_dir,
                        sigma=sigma,
                        seed=seed,
                        atoms=noise_atoms,
                        keep_root=keep_root,
                        experiment_name=exp,
                    ) as (run_positive_dir, run_negative_dir):
                        row = evaluate(
                            true_dir=run_positive_dir,
                            false_dir=run_negative_dir,
                            workers=workers,
                            prepare_workers=prepare_workers,
                            save_dir=output_dir,
                            metric_level=metric_level,
                            tag=f"{timestamp}_{exp}",
                            detector_overrides=None,
                            print_metric_tables=False,
                        )
                    row.update(
                        {
                            "exp": exp,
                            "perturbation_mode": "coordinate_noise",
                            "coordinate_noise_sigma": float(sigma),
                            "coordinate_noise_seed": int(seed),
                            "coordinate_noise_atoms": noise_atoms,
                        }
                    )
                    rows.append(row)
                    print("  " + _summarize_row(row) + "\n")

        if modes in {"step", "both"}:
            for step_size in step_sizes:
                exp = f"slice_step_{_level_token(step_size)}"
                print(f"[{exp}]")
                row = evaluate(
                    true_dir=base_positive_dir,
                    false_dir=base_negative_dir,
                    workers=workers,
                    prepare_workers=prepare_workers,
                    save_dir=output_dir,
                    metric_level=metric_level,
                    tag=f"{timestamp}_{exp}",
                    detector_overrides={"slicer.step_size": float(step_size)},
                    print_metric_tables=False,
                )
                row.update(
                    {
                        "exp": exp,
                        "perturbation_mode": "slice_step_size",
                        "slicer_step_size": float(step_size),
                    }
                )
                rows.append(row)
                print("  " + _summarize_row(row) + "\n")

    if not rows:
        print("No perturbation experiments were selected.")
        return None

    dataframe = _ordered_summary(pd.DataFrame(rows))
    output_path = output_dir / f"perturbation_summary_{timestamp}.csv"
    dataframe.to_csv(output_path, index=False)

    display_columns = [
        column
        for column in [
            "exp",
            "perturbation_mode",
            "coordinate_noise_sigma",
            "coordinate_noise_seed",
            "slicer_step_size",
            "chain_f1",
            "chain_mcc",
            "file_f1",
            "file_mcc",
        ]
        if column in dataframe.columns
    ]
    print("=== Perturbation summary ===")
    if display_columns:
        print(dataframe[display_columns].to_string(index=False))
    print(f"\nSaved: {output_path}\n")
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Cooper-Beta perturbation evaluations without changing detector code. "
            "Modes: coordinate noise via temporary perturbed structures, and Z-slice "
            "step-size sensitivity via slicer.step_size overrides."
        )
    )
    parser.add_argument(
        "--positives",
        "--true",
        dest="true",
        required=True,
        help="Directory containing positive examples.",
    )
    parser.add_argument(
        "--negatives",
        "--false",
        dest="false",
        required=True,
        help="Directory containing negative examples.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Analysis workers.")
    parser.add_argument(
        "--prepare",
        type=int,
        default=None,
        help="Preparation workers (default: follows --workers).",
    )
    parser.add_argument(
        "--save-dir",
        default="eval_outputs",
        help="Directory where perturbation outputs are written.",
    )
    parser.add_argument(
        "--metric-level",
        choices=["chain", "file", "both"],
        default="both",
        help="Which metric level to compute.",
    )
    parser.add_argument(
        "--modes",
        choices=["noise", "step", "both"],
        default="both",
        help="Perturbation modes to run.",
    )
    parser.add_argument(
        "--noise-sigmas",
        default=DEFAULT_NOISE_SIGMAS,
        help=f"Comma-separated coordinate noise sigma values in Angstroms (default: {DEFAULT_NOISE_SIGMAS}).",
    )
    parser.add_argument(
        "--noise-seeds",
        default=DEFAULT_NOISE_SEEDS,
        help=f"Comma-separated random seeds for coordinate noise (default: {DEFAULT_NOISE_SEEDS}).",
    )
    parser.add_argument(
        "--noise-atoms",
        choices=["ca", "all"],
        default="ca",
        help="Atoms to perturb for coordinate-noise structures (default: ca).",
    )
    parser.add_argument(
        "--step-sizes",
        default=DEFAULT_SLICE_STEPS,
        help=f"Comma-separated slicer.step_size values (default: {DEFAULT_SLICE_STEPS}).",
    )
    parser.add_argument(
        "--max-files-per-split",
        type=int,
        default=None,
        help="Optional lightweight subset size per positive/negative split.",
    )
    parser.add_argument(
        "--keep-perturbed-structures",
        action="store_true",
        help="Keep generated noisy structures under the output directory.",
    )
    args = parser.parse_args(argv)

    try:
        noise_sigmas = _parse_float_list(args.noise_sigmas)
        step_sizes = _parse_float_list(args.step_sizes)
        noise_seeds = _parse_int_list(args.noise_seeds)
        if any(value < 0.0 for value in noise_sigmas):
            raise ValueError("--noise-sigmas values must be >= 0.")
        if any(value <= 0.0 for value in step_sizes):
            raise ValueError("--step-sizes values must be > 0.")
        if args.max_files_per_split is not None and args.max_files_per_split <= 0:
            raise ValueError("--max-files-per-split must be > 0 when provided.")

        run_perturbation_suite(
            positive_dir=Path(args.true),
            negative_dir=Path(args.false),
            workers=args.workers,
            prepare_workers=args.prepare,
            save_dir=Path(args.save_dir),
            metric_level=args.metric_level,
            modes=args.modes,
            noise_sigmas=noise_sigmas,
            noise_seeds=noise_seeds,
            step_sizes=step_sizes,
            noise_atoms=args.noise_atoms,
            max_files_per_split=args.max_files_per_split,
            keep_perturbed_structures=args.keep_perturbed_structures,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
