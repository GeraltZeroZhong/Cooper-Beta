from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_import_package():
    pkg = importlib.import_module("cooper_beta")
    assert hasattr(pkg, "Config")
    assert hasattr(pkg, "main")
    assert hasattr(pkg, "extract_chain_slices")


def test_import_pipeline():
    pipe = importlib.import_module("cooper_beta.pipeline")
    assert hasattr(pipe, "main")


def test_build_config_supports_nested_and_legacy_overrides():
    config_mod = importlib.import_module("cooper_beta.config")

    cfg = config_mod.build_config(
        {
            "runtime.dssp_bin_path": "/tmp/mkdssp",
            "MIN_POINTS_PER_SLICE": 11,
        }
    )

    assert cfg.runtime.dssp_bin_path == "/tmp/mkdssp"
    assert cfg.analyzer.fit.min_points_per_slice == 11
    assert config_mod.Config.DSSP_BIN_PATH == "/tmp/mkdssp"
    assert config_mod.Config.MIN_POINTS_PER_SLICE == 11


def test_dataclass_defaults_match_hydra_scientific_defaults():
    config_mod = importlib.import_module("cooper_beta.config")

    direct = config_mod.AppConfig()
    hydra = config_mod.build_config()

    assert direct.analyzer.fit.max_rmse == hydra.analyzer.fit.max_rmse
    assert direct.analyzer.fit.min_inlier_frac == hydra.analyzer.fit.min_inlier_frac
    assert direct.analyzer.decision.min_scored_layers == hydra.analyzer.decision.min_scored_layers
    assert direct.analyzer.decision.near_miss_rescue.soft_nn_enabled == (
        hydra.analyzer.decision.near_miss_rescue.soft_nn_enabled
    )
    assert direct.analyzer.rules.angle.max_gap_deg == hydra.analyzer.rules.angle.max_gap_deg
    assert direct.analyzer.rules.angle.order.min_local_frac == (
        hydra.analyzer.rules.angle.order.min_local_frac
    )


def test_build_config_rejects_invalid_user_values():
    config_mod = importlib.import_module("cooper_beta.config")
    exceptions = importlib.import_module("cooper_beta.exceptions")

    with pytest.raises(exceptions.ConfigValidationError, match="barrel_valid_ratio"):
        config_mod.build_config({"analyzer.decision.barrel_valid_ratio": 1.5})


def test_require_dssp_binary_reports_help(monkeypatch: pytest.MonkeyPatch):
    runtime = importlib.import_module("cooper_beta.runtime")

    monkeypatch.setattr(runtime.Config, "DSSP_BIN_PATH", None)
    monkeypatch.setattr(runtime, "_resolve_executable", lambda _: None)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="Cooper-Beta requires DSSP"):
        runtime.require_dssp_binary()


def test_require_dssp_binary_rejects_invalid_explicit_path(monkeypatch: pytest.MonkeyPatch):
    runtime = importlib.import_module("cooper_beta.runtime")

    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/usr/bin/mkdssp")

    with pytest.raises(RuntimeError, match="Configured DSSP executable"):
        runtime.require_dssp_binary("/definitely/missing/mkdssp")


def test_cli_requires_explicit_input_path(capsys: pytest.CaptureFixture[str]):
    cli = importlib.import_module("cooper_beta.cli")

    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert exc_info.value.code == 2
    assert "input path is required" in capsys.readouterr().err


def test_cli_accepts_hydra_override_before_path(capsys: pytest.CaptureFixture[str]):
    cli = importlib.import_module("cooper_beta.cli")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["runtime.workers=1", "missing.pdb"])

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "Input path does not exist" in stderr
    assert "input path is required" not in stderr


def test_cli_reports_bad_output_path_without_traceback(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    cli = importlib.import_module("cooper_beta.cli")
    pipeline = importlib.import_module("cooper_beta.pipeline")
    input_file = tmp_path / "toy.pdb"
    input_file.write_text("HEADER\n")

    monkeypatch.setattr(pipeline, "require_dssp_binary", lambda explicit_path: "/opt/dssp/mkdssp")

    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(input_file), "--out", str(tmp_path)])

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "Error:" in stderr
    assert "Output CSV path points to a directory" in stderr
    assert "Traceback" not in stderr


def test_cli_check_env_rejects_invalid_configured_dssp(
    capsys: pytest.CaptureFixture[str],
):
    cli = importlib.import_module("cooper_beta.cli")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--check-env", "runtime.dssp_bin_path=/definitely/missing/mkdssp"])

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "Configured DSSP executable" in stderr
    assert "Traceback" not in stderr


def test_evaluate_rejects_invalid_metric_level_before_detector_runs(tmp_path: Path):
    runner = importlib.import_module("cooper_beta.evaluation.runner")

    with pytest.raises(ValueError, match="metric_level"):
        runner.evaluate(
            true_dir=tmp_path / "true",
            false_dir=tmp_path / "false",
            workers=1,
            prepare_workers=1,
            save_dir=tmp_path / "out",
            metric_level="sample",
            tag="bad",
        )


def test_evaluation_file_outputs_preserve_all_error_files(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    runner = importlib.import_module("cooper_beta.evaluation.runner")
    metrics = importlib.import_module("cooper_beta.evaluation.metrics")

    positive = pd.DataFrame(
        [
            {
                "filename": "broken.pdb",
                "source_path": str(tmp_path / "true" / "broken.pdb"),
                "chain": "",
                "result": "ERROR",
                "reason": "parse failed",
            }
        ]
    )
    negative = pd.DataFrame(
        [
            {
                "filename": "ok.pdb",
                "source_path": str(tmp_path / "false" / "ok.pdb"),
                "chain": "A",
                "result": "NON_BARREL",
                "decision_score": 0.0,
                "score_adjust": 0.0,
            }
        ]
    )

    _, _, aggregated = runner.save_outputs(positive, negative, tmp_path / "out", "unit")
    error_row = aggregated.loc[aggregated["filename"].eq("broken.pdb")].iloc[0]
    assert bool(error_row["all_error"]) is True
    assert bool(error_row["use_for_metrics"]) is False

    _, extra = metrics.compute_file_metrics(aggregated)
    assert extra["dropped_true_error_files"] == 1
    assert extra["n_true_files"] == 0
    assert extra["n_false_files"] == 1


def test_path_executed_module_entrypoints_have_helpful_smoke_output():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}

    version = subprocess.run(
        [sys.executable, str(root / "src" / "cooper_beta" / "__main__.py"), "--version"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert version.returncode == 0
    assert "cooper-beta" in version.stdout

    help_result = subprocess.run(
        [
            sys.executable,
            str(root / "src" / "cooper_beta" / "evaluation" / "__main__.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert help_result.returncode == 0
    assert "--metric-level" in help_result.stdout
