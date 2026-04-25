from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa

DEFAULT_MIN_RESIDUES = 15
SUPPORTED_EXTENSIONS = {".pdb", ".ent", ".cif", ".mmcif"}


@dataclass(frozen=True)
class GeneratedSequence:
    sample_id: str
    source_path: str
    chain_id: str
    n_residues: int
    sequence: str


@dataclass(frozen=True)
class GeneratedFastaSet:
    output_dir: str
    fasta_path: str
    residue_mapping_path: str
    records: list[GeneratedSequence]


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "blank"


def _structure_extension(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return Path(name).suffix


def _structure_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".gz"):
        name = name[:-3]
    while Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
        name = Path(name).stem
    return _safe_id(name)


def _is_structure_path(path: Path) -> bool:
    return path.is_file() and _structure_extension(path) in SUPPORTED_EXTENSIONS


def discover_structure_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Structure input does not exist: {path}")
    if path.is_file():
        if not _is_structure_path(path):
            raise ValueError(f"Unsupported structure extension: {path}")
        return [path]

    files = [candidate for candidate in sorted(path.rglob("*")) if _is_structure_path(candidate)]
    if not files:
        raise ValueError(f"No supported structure files found in {path}")
    return files


def _decompress_gzip_to_temp_if_needed(path: Path) -> Path | None:
    with path.open("rb") as handle:
        if handle.read(2) != b"\x1f\x8b":
            return None

    suffix = _structure_extension(path) or ".pdb"
    fd, temp_name = tempfile.mkstemp(suffix=suffix)
    with gzip.open(path, "rb") as source, os.fdopen(fd, "wb") as target:
        shutil.copyfileobj(source, target)
    return Path(temp_name)


def _parse_structure(path: Path):
    temp_path = _decompress_gzip_to_temp_if_needed(path)
    parse_path = temp_path or path
    extension = _structure_extension(parse_path)
    parser = (
        MMCIFParser(QUIET=True)
        if extension in {".cif", ".mmcif"}
        else PDBParser(QUIET=True, PERMISSIVE=True, get_header=False)
    )
    try:
        return parser.get_structure(_structure_stem(path), str(parse_path))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _chain_residues(chain) -> list[object]:
    residues = []
    for residue in chain.get_unpacked_list():
        if is_aa(residue, standard=False) and "CA" in residue:
            residues.append(residue)
    return residues


def _one_letter_code(residue) -> str:
    return protein_letters_3to1_extended.get(residue.get_resname().upper(), "X")


def _wrap_fasta(sequence: str, width: int = 60) -> str:
    return "\n".join(sequence[index : index + width] for index in range(0, len(sequence), width))


def _write_residue_mapping(
    mapping_rows: Iterable[dict[str, object]],
    residue_mapping_path: Path,
) -> None:
    fieldnames = [
        "sample_id",
        "sequence_index",
        "source_file",
        "chain_id",
        "residue_name",
        "residue_number",
        "insertion_code",
        "one_letter_code",
    ]
    with residue_mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapping_rows)


def generate_structure_fasta(
    structure_input: str | Path,
    output_dir: str | Path,
    *,
    min_residues: int = DEFAULT_MIN_RESIDUES,
) -> GeneratedFastaSet:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    fasta_path = output / "sequences.fasta"
    residue_mapping_path = output / "residue_mapping.csv"

    records: list[GeneratedSequence] = []
    mapping_rows: list[dict[str, object]] = []
    seen_ids: Counter[str] = Counter()

    for structure_path in discover_structure_files(structure_input):
        structure = _parse_structure(structure_path)
        model = structure[0]
        for chain in model.get_chains():
            residues = _chain_residues(chain)
            if len(residues) < min_residues:
                continue

            chain_id = str(chain.id).strip() or "blank"
            base_id = f"{_structure_stem(structure_path)}_{_safe_id(chain_id)}"
            seen_ids[base_id] += 1
            sample_id = base_id if seen_ids[base_id] == 1 else f"{base_id}_{seen_ids[base_id]}"
            sequence = "".join(_one_letter_code(residue) for residue in residues)

            records.append(
                GeneratedSequence(
                    sample_id=sample_id,
                    source_path=str(structure_path),
                    chain_id=chain_id,
                    n_residues=len(residues),
                    sequence=sequence,
                )
            )

            for index, residue in enumerate(residues):
                residue_id = residue.get_id()
                insertion_code = str(residue_id[2]).strip()
                mapping_rows.append(
                    {
                        "sample_id": sample_id,
                        "sequence_index": index,
                        "source_file": str(structure_path),
                        "chain_id": chain_id,
                        "residue_name": residue.get_resname(),
                        "residue_number": residue_id[1],
                        "insertion_code": insertion_code,
                        "one_letter_code": _one_letter_code(residue),
                    }
                )

    if not records:
        raise ValueError("No protein chains with enough CA residues were found.")

    with fasta_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record.sample_id}\n{_wrap_fasta(record.sequence)}\n")

    _write_residue_mapping(mapping_rows, residue_mapping_path)

    return GeneratedFastaSet(
        output_dir=str(output),
        fasta_path=str(fasta_path),
        residue_mapping_path=str(residue_mapping_path),
        records=records,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate chain-level FASTA inputs for the PRED-TMBB2 adapter."
    )
    parser.add_argument("structure_input", help="PDB/CIF/mmCIF file or directory.")
    parser.add_argument("--out-dir", required=True, help="Directory for FASTA and metadata.")
    parser.add_argument(
        "--min-residues",
        type=int,
        default=DEFAULT_MIN_RESIDUES,
        help=f"Minimum CA residue count required for a chain. Default: {DEFAULT_MIN_RESIDUES}.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    generated = generate_structure_fasta(
        args.structure_input,
        args.out_dir,
        min_residues=args.min_residues,
    )

    print(f"Generated sequences: {len(generated.records)}")
    print(f"FASTA: {generated.fasta_path}")
    print(f"Residue mapping: {generated.residue_mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
