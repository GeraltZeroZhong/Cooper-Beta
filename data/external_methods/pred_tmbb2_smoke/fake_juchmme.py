from __future__ import annotations

import argparse
from pathlib import Path


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    current_sequence: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records.append((current_id, "".join(current_sequence)))
            current_id = line[1:].split(None, 1)[0]
            current_sequence = []
        else:
            current_sequence.append(line)

    if current_id is not None:
        records.append((current_id, "".join(current_sequence)))
    return records


def _fake_topology(sample_id: str, length: int) -> tuple[str, float, float]:
    if "nonbarrel" in sample_id:
        return "I" * length, -5.0, 0.82

    motif = "MMOOMMIIMMOIIIIIIIII"
    repeated = (motif * ((length // len(motif)) + 1))[:length]
    return repeated, 4.5, 0.93


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fasta", required=True)
    parser.add_argument("-a")
    parser.add_argument("-e")
    parser.add_argument("-m")
    parser.add_argument("-c")
    args = parser.parse_args()

    print("JUCHMME :: fake smoke runner")
    for sample_id, sequence in _read_fasta(Path(args.fasta)):
        topology, logodds, reliability = _fake_topology(sample_id, len(sequence))
        print(f"ID: >{sample_id}")
        print(f"SQ: {sequence}")
        print(
            "CC:\t"
            f"len = {float(len(sequence))}\t"
            f"logodds = {logodds}\t"
            "maxProb = 0.99\t"
            "(-logprob/lng) = 1.8"
        )
        print("LS: 12.0")
        print(f"LR: {reliability}")
        print(f"LP: {topology}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
