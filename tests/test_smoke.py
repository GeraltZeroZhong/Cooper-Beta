from __future__ import annotations

import importlib

import pytest


def test_import_package():
    pkg = importlib.import_module("cooper_beta")
    assert hasattr(pkg, "Config")
    assert hasattr(pkg, "main")


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
