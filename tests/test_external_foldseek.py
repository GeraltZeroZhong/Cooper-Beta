from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runner = importlib.import_module("external_methods.foldseek.runner")
structures = importlib.import_module("external_methods.foldseek.structures")
structure_search = importlib.import_module("external_methods.foldseek.structure_search")

load_hits_tsv = runner.load_hits_tsv
summarize_hits = runner.summarize_hits
run_baseline = runner.run_baseline
generate_structure_chains = structures.generate_structure_chains
foldseek_query_aliases = structures.foldseek_query_aliases
run_structure_search_baseline = structure_search.run_structure_search_baseline

SMOKE_DATA = ROOT / "data" / "external_methods" / "foldseek_smoke"
STRUCTURE_SMOKE_DATA = ROOT / "data" / "external_methods" / "isitabarrel_structure_smoke"


def test_load_and_summarize_foldseek_hits_from_fixture():
    hits = load_hits_tsv(SMOKE_DATA / "hits.tsv")

    results = summarize_hits(
        hits,
        query_ids=["toy_barrel_A", "toy_nonbarrel_A", "missing_A"],
    )

    assert [result.sample_id for result in results] == [
        "toy_barrel_A",
        "toy_nonbarrel_A",
        "missing_A",
    ]
    assert [result.result for result in results] == ["BARREL", "NON_BARREL", "NON_BARREL"]
    assert results[0].score == 0.72
    assert results[0].decision_rule == "min_qtmscore_ttmscore>=0.5;qcov>=0.6;tcov>=0.6"
    assert results[1].eligible_hit_count == 0
    assert results[2].hit_count == 0


def test_summarize_foldseek_hits_can_alias_and_ignore_targets():
    hits = load_hits_tsv(SMOKE_DATA / "hits.tsv")

    aliased_results = summarize_hits(
        hits,
        query_ids=["toy_barrel_A"],
        target_aliases={"ref_barrel_A": "reference_barrel_A"},
    )
    assert aliased_results[0].best_target == "reference_barrel_A"

    filtered_results = summarize_hits(
        hits,
        query_ids=["toy_barrel_A"],
        target_aliases={"ref_barrel_A": "toy_barrel_A"},
        ignore_target_ids_by_query={"toy_barrel_A": {"toy_barrel_A"}},
    )

    assert filtered_results[0].result == "NON_BARREL"
    assert filtered_results[0].hit_count == 0
    assert filtered_results[0].ignored_target_hit_count == 1


def test_generate_structure_chains_from_pdb_fixture(tmp_path: Path):
    generated = generate_structure_chains(
        STRUCTURE_SMOKE_DATA / "toy_barrel.pdb",
        tmp_path / "generated",
    )

    assert [record.sample_id for record in generated.records] == ["toy_barrel_A"]
    assert Path(generated.records[0].chain_path).exists()
    assert Path(generated.manifest_path).exists()
    assert Path(generated.residue_mapping_path).exists()

    chain_text = Path(generated.records[0].chain_path).read_text(encoding="utf-8")
    assert " A   1" in chain_text
    assert " A  16" in chain_text
    assert foldseek_query_aliases(generated.records)["toy_barrel_A.pdb_A"] == "toy_barrel_A"


def test_run_foldseek_adapter_with_fake_runner(tmp_path: Path):
    generated = generate_structure_chains(
        STRUCTURE_SMOKE_DATA / "toy_barrel.pdb",
        tmp_path / "generated",
    )
    output_csv = tmp_path / "normalized.csv"

    results = run_baseline(
        generated.chain_dir,
        SMOKE_DATA,
        work_dir=tmp_path / "work",
        output_path=output_csv,
        query_ids=[record.sample_id for record in generated.records],
        query_aliases=foldseek_query_aliases(generated.records),
        command_prefix=[sys.executable, str(SMOKE_DATA / "fake_foldseek.py")],
    )

    assert [result.result for result in results] == ["BARREL"]
    assert results[0].best_target == "ref_barrel_A"

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["baseline"] == "foldseek_tmalign_structure_search"
    assert rows[0]["sample_id"] == "toy_barrel_A"
    assert rows[0]["result"] == "BARREL"
    assert float(rows[0]["score"]) == 0.72


def test_run_structure_search_baseline_smoke(tmp_path: Path):
    output_csv = tmp_path / "structure_baseline.csv"

    run = run_structure_search_baseline(
        STRUCTURE_SMOKE_DATA / "toy_barrel.pdb",
        tmp_path / "structure_work",
        reference_structures=STRUCTURE_SMOKE_DATA,
        output_path=output_csv,
        command_prefix=[sys.executable, str(SMOKE_DATA / "fake_foldseek.py")],
    )

    assert [record.sample_id for record in run.generated_chains.records] == ["toy_barrel_A"]
    assert [result.result for result in run.results] == ["BARREL"]

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["baseline"] == "foldseek_tmalign_structure_search"
    assert rows[0]["sample_id"] == "toy_barrel_A"
    assert rows[0]["result"] == "BARREL"
