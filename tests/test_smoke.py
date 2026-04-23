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


def test_require_dssp_binary_reports_help(monkeypatch: pytest.MonkeyPatch):
    runtime = importlib.import_module("cooper_beta.runtime")

    monkeypatch.setattr(runtime.Config, "DSSP_BIN_PATH", None)
    monkeypatch.setattr(runtime, "_resolve_executable", lambda _: None)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="Cooper-Beta requires DSSP"):
        runtime.require_dssp_binary()
