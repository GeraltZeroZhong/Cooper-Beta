from __future__ import annotations

from pathlib import Path

import pytest

import cooper_beta.pipeline_workers as pipeline_workers
from cooper_beta.config import build_config
from cooper_beta.pipeline import (
    detect,
    discover_input_files,
    resolve_analysis_worker_count,
    resolve_prepare_worker_count,
    run_pipeline,
    run_pipeline_result,
)
from cooper_beta.results import print_results_summary, write_results_csv


def test_prepare_worker_default_heuristic_follows_analysis_workers():
    assert resolve_prepare_worker_count(None, 1) == 1
    assert resolve_prepare_worker_count(None, 2) == 2
    assert resolve_prepare_worker_count(None, 4) == 4
    assert resolve_prepare_worker_count(None, 7) == 7
    assert resolve_prepare_worker_count(None, 8) == 8


def test_analysis_worker_override_is_respected():
    assert resolve_analysis_worker_count(6, cpu_reserve=1) == 6


def test_run_pipeline_resolves_dssp_path_once(tmp_path: Path, monkeypatch):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n")
    output_file = tmp_path / "results.csv"
    cfg = build_config(
        {
            "input.path": str(tmp_path),
            "output.csv_path": str(output_file),
            "runtime.workers": 1,
            "runtime.prepare_workers": 1,
        }
    )
    calls = []

    def fake_require_dssp_binary(explicit_path):
        calls.append(explicit_path)
        return "/opt/dssp/mkdssp"

    def fake_iter_prepared_payload_batches(files, cfg_arg, prepare_workers, **kwargs):
        assert kwargs["show_progress"] is True
        assert files == [str(input_file)]
        assert prepare_workers == 1
        assert cfg_arg.runtime.dssp_bin_path == "/opt/dssp/mkdssp"
        yield [{"filename": "toy.pdb", "chain": "A", "residues_data": []}]

    def fake_run_analysis_stream(payload_batches, cfg_arg, workers, *, on_results=None, **kwargs):
        assert kwargs["show_progress"] is True
        batches = list(payload_batches)
        assert batches == [[{"filename": "toy.pdb", "chain": "A", "residues_data": []}]]
        assert workers == 1
        assert cfg_arg.runtime.dssp_bin_path == "/opt/dssp/mkdssp"
        rows = [{"filename": "toy.pdb", "chain": "A", "result": "FILTERED_OUT"}]
        if on_results is not None:
            on_results(rows)
        return rows

    monkeypatch.setattr("cooper_beta.pipeline.require_dssp_binary", fake_require_dssp_binary)
    monkeypatch.setattr(
        "cooper_beta.pipeline.iter_prepared_payload_batches",
        fake_iter_prepared_payload_batches,
    )
    monkeypatch.setattr("cooper_beta.pipeline.run_analysis_stream", fake_run_analysis_stream)
    monkeypatch.setattr("cooper_beta.pipeline.print_results_summary", lambda *args, **kwargs: None)

    rows = run_pipeline(cfg)

    assert rows == [{"filename": "toy.pdb", "chain": "A", "result": "FILTERED_OUT"}]
    assert calls == [None]


def test_detect_returns_structured_result_without_csv(tmp_path: Path, monkeypatch):
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n")
    output_file = tmp_path / "results.csv"
    cfg = build_config(
        {
            "input.path": str(tmp_path),
            "output.csv_path": str(output_file),
            "runtime.workers": 1,
            "runtime.prepare_workers": 1,
        }
    )

    def fake_iter_prepared_payload_batches(files, cfg_arg, prepare_workers, **kwargs):
        assert kwargs["show_progress"] is False
        assert files == [str(input_file)]
        assert prepare_workers == 1
        yield [{"filename": "toy.pdb", "chain": "A", "residues_data": []}]

    def fake_run_analysis_stream(payload_batches, cfg_arg, workers, *, on_results=None, **kwargs):
        assert kwargs["show_progress"] is False
        assert list(payload_batches) == [[{"filename": "toy.pdb", "chain": "A", "residues_data": []}]]
        assert workers == 1
        assert on_results is None
        return [{"filename": "toy.pdb", "chain": "A", "result": "FILTERED_OUT", "reason": "short"}]

    monkeypatch.setattr("cooper_beta.pipeline.require_dssp_binary", lambda explicit_path: "/opt/dssp/mkdssp")
    monkeypatch.setattr(
        "cooper_beta.pipeline.iter_prepared_payload_batches",
        fake_iter_prepared_payload_batches,
    )
    monkeypatch.setattr("cooper_beta.pipeline.run_analysis_stream", fake_run_analysis_stream)

    result = detect(str(tmp_path), config=cfg, write_csv=False, print_summary=False)

    assert result.output_path is None
    assert result.input_files == [str(input_file)]
    assert result.result_counts == {"FILTERED_OUT": 1}
    assert result.rows[0].filename == "toy.pdb"
    assert result.rows[0].reason == "short"
    assert not output_file.exists()


def test_detect_rejects_config_and_overrides_together():
    with pytest.raises(TypeError, match="overrides"):
        detect("input.pdb", config=build_config(), overrides={"runtime.workers": 1})


def test_run_pipeline_reports_all_prepare_failures(tmp_path: Path, monkeypatch):
    input_file = tmp_path / "bad.pdb"
    input_file.write_text("not a structure\n")
    output_file = tmp_path / "results.csv"
    cfg = build_config(
        {
            "input.path": str(tmp_path),
            "output.csv_path": str(output_file),
            "runtime.workers": 1,
            "runtime.prepare_workers": 1,
        }
    )
    exceptions = __import__("cooper_beta.exceptions", fromlist=["InputValidationError"])

    def fake_iter_prepared_payload_batches(files, cfg_arg, prepare_workers, *, on_errors=None, **kwargs):
        del files, cfg_arg, prepare_workers, kwargs
        if on_errors is not None:
            on_errors(["bad.pdb: parse failed"])
        if False:
            yield []

    monkeypatch.setattr("cooper_beta.pipeline.require_dssp_binary", lambda explicit_path: "/opt/dssp/mkdssp")
    monkeypatch.setattr(
        "cooper_beta.pipeline.iter_prepared_payload_batches",
        fake_iter_prepared_payload_batches,
    )

    with pytest.raises(exceptions.InputValidationError, match="failed during preparation"):
        run_pipeline_result(cfg, print_summary=False, strict_input=True, show_progress=False)

    csv_text = output_file.read_text(encoding="utf-8")
    assert "bad.pdb" in csv_text
    assert "parse failed" in csv_text


def test_discover_input_files_is_case_insensitive(tmp_path: Path):
    mixed_case = tmp_path / "example.mmCIF"
    mixed_case.write_text("data_example\n")
    ignored = tmp_path / "notes.txt"
    ignored.write_text("ignore me\n")

    assert discover_input_files(str(tmp_path), [".pdb", ".cif", ".mmcif"], strict=True) == [
        str(mixed_case)
    ]


def test_run_analysis_stream_batches_payloads_and_invokes_sink(monkeypatch):
    cfg = build_config({"runtime.analysis_batch_size": 2})
    payloads = [
        {"filename": f"toy-{index}.pdb", "chain": "A", "residues_data": []}
        for index in range(5)
    ]
    analyzed_batches = []

    def fake_analyze_payload_batch(payload_batch, cfg_arg):
        assert cfg_arg is cfg
        analyzed_batches.append([payload["filename"] for payload in payload_batch])
        return [
            {
                "filename": payload["filename"],
                "chain": payload["chain"],
                "result": "FILTERED_OUT",
            }
            for payload in payload_batch
        ]

    written_rows = []
    monkeypatch.setattr(
        pipeline_workers,
        "analyze_payload_batch",
        fake_analyze_payload_batch,
    )

    rows = pipeline_workers.run_analysis_stream(
        [payloads],
        cfg,
        workers=1,
        on_results=written_rows.extend,
    )

    assert analyzed_batches == [
        ["toy-0.pdb", "toy-1.pdb"],
        ["toy-2.pdb", "toy-3.pdb"],
        ["toy-4.pdb"],
    ]
    assert rows == written_rows
    assert len(rows) == 5


def test_print_results_summary_limits_console_but_writes_full_csv(tmp_path: Path, capsys):
    rows = [
        {"filename": "a.pdb", "chain": "A", "result": "FILTERED_OUT", "result_stage": "prefilter"},
        {"filename": "b.pdb", "chain": "A", "result": "NON_BARREL", "result_stage": "decision"},
        {"filename": "c.pdb", "chain": "A", "result": "BARREL", "result_stage": "decision"},
    ]
    output_file = tmp_path / "results.csv"

    print_results_summary(rows, str(output_file), summary_limit=1)

    captured = capsys.readouterr().out
    assert "Rows: 3" in captured
    assert "omitted 2 row(s)" in captured
    assert "a.pdb" in captured
    assert "b.pdb" not in captured
    assert output_file.read_text().count("\n") == 4


def test_write_results_csv_writes_schema_for_empty_results(tmp_path: Path):
    output_file = tmp_path / "empty.csv"

    write_results_csv([], str(output_file))

    header = output_file.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("filename,chain,result,result_stage")
