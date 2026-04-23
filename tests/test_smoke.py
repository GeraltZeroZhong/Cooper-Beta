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


def test_require_dssp_binary_reports_help(monkeypatch: pytest.MonkeyPatch):
    runtime = importlib.import_module("cooper_beta.runtime")

    monkeypatch.setattr(runtime.Config, "DSSP_BIN_PATH", None)
    monkeypatch.setattr(runtime, "_resolve_executable", lambda _: None)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="Cooper-Beta requires DSSP"):
        runtime.require_dssp_binary()
