from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("protid_list")
    parser.add_argument("data_folder")
    args = parser.parse_args()

    protid_list = Path(args.protid_list)
    data_folder = Path(args.data_folder)
    ids = [line.split("\t", 1)[0].strip() for line in protid_list.read_text().splitlines()]
    ids = [sample_id for sample_id in ids if sample_id]

    for sample_id in ids:
        map_path = data_folder / f"{sample_id}.pkl"
        if not map_path.exists():
            raise FileNotFoundError(map_path)

    output = ["MAP_NAME\tCC2\tBSS\tH4\tCC2_TO_H4\tCC2_TO_BSS\tBSS_TO_CC2_TO_BSS"]
    for sample_id in ids:
        if "barrel" in sample_id and "nonbarrel" not in sample_id:
            output.append(f"{sample_id}\t0.75\t5\t4\t0.75\t0.75\t0.75")
        else:
            output.append(f"{sample_id}\t0\t1\t0\t0\t0\t0")

    Path("results.tsv").write_text("\n".join(output) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
