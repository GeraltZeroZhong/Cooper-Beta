from __future__ import annotations

import importlib


def test_import_package():
    pkg = importlib.import_module("cooper_beta")
    assert hasattr(pkg, "Config")
    assert hasattr(pkg, "main")


def test_import_pipeline():
    pipe = importlib.import_module("cooper_beta.pipeline")
    assert hasattr(pipe, "main")
