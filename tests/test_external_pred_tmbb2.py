from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runner = importlib.import_module("external_methods.pred_tmbb2.runner")
sequences = importlib.import_module("external_methods.pred_tmbb2.sequences")
structure_sequence = importlib.import_module("external_methods.pred_tmbb2.structure_sequence")

parse_juchmme_stdout = runner.parse_juchmme_stdout
run_baseline = runner.run_baseline
generate_structure_fasta = sequences.generate_structure_fasta
run_structure_sequence_baseline = structure_sequence.run_structure_sequence_baseline

SMOKE_DATA = ROOT / "data" / "external_methods" / "pred_tmbb2_smoke"
STRUCTURE_SMOKE_DATA = ROOT / "data" / "external_methods" / "isitabarrel_structure_smoke"


def test_parse_juchmme_stdout_from_fixture():
    output = (SMOKE_DATA / "juchmme_stdout.txt").read_text(encoding="utf-8")

    results = parse_juchmme_stdout(output)

    assert [result.sample_id for result in results] == ["toy_barrel", "toy_nonbarrel"]
    assert [result.result for result in results] == ["BARREL", "NON_BARREL"]
    assert results[0].tm_strands == 3
    assert results[0].score == 3.0
    assert results[0].reliability == 0.93
    assert results[1].logodds == -5.0


def test_run_pred_tmbb2_adapter_reads_fasta_and_writes_normalized_results(tmp_path: Path):
    output_csv = tmp_path / "normalized.csv"

    results = run_baseline(
        SMOKE_DATA / "input.fasta",
        work_dir=tmp_path / "work",
        output_path=output_csv,
        command_prefix=[sys.executable, str(SMOKE_DATA / "fake_juchmme.py")],
    )

    assert [result.result for result in results] == ["BARREL", "NON_BARREL"]
    assert output_csv.exists()

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["baseline"] == "pred_tmbb2_single_juchmme"
    assert rows[0]["sample_id"] == "toy_barrel"
    assert rows[0]["result"] == "BARREL"
    assert rows[0]["decision_rule"] == "LP_tm_strands>=3"
    assert rows[1]["sample_id"] == "toy_nonbarrel"
    assert float(rows[1]["score"]) == 0.0


def test_generate_structure_fasta_from_pdb_fixture(tmp_path: Path):
    generated = generate_structure_fasta(
        STRUCTURE_SMOKE_DATA / "toy_barrel.pdb",
        tmp_path / "generated",
    )

    assert [record.sample_id for record in generated.records] == ["toy_barrel_A"]
    assert generated.records[0].sequence == "AVGSTLIFAVGSTLIF"
    assert Path(generated.fasta_path).read_text(encoding="utf-8") == (
        ">toy_barrel_A\nAVGSTLIFAVGSTLIF\n"
    )
    assert Path(generated.residue_mapping_path).exists()


def test_run_structure_sequence_baseline_smoke(tmp_path: Path):
    output_csv = tmp_path / "structure_baseline.csv"

    run = run_structure_sequence_baseline(
        STRUCTURE_SMOKE_DATA / "toy_barrel.pdb",
        tmp_path / "structure_work",
        output_path=output_csv,
        command_prefix=[sys.executable, str(SMOKE_DATA / "fake_juchmme.py")],
    )

    assert [record.sample_id for record in run.generated_fasta.records] == ["toy_barrel_A"]
    assert [result.result for result in run.results] == ["BARREL"]

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["baseline"] == "pred_tmbb2_single_juchmme"
    assert rows[0]["sample_id"] == "toy_barrel_A"
    assert rows[0]["result"] == "BARREL"
