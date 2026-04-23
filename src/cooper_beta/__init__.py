"""
cooper_beta

A small toolkit/pipeline to detect beta-barrel-like protein chains from PDB/mmCIF structures.
"""
from __future__ import annotations

__all__ = [
    "AppConfig",
    "Config",
    "ProteinLoader",
    "PCAAligner",
    "ProteinSlicer",
    "BarrelAnalyzer",
    "build_config",
    "find_dssp_binary",
    "require_dssp_binary",
    "main",
]

from .alignment import PCAAligner
from .analyzer import BarrelAnalyzer
from .config import AppConfig, Config, build_config
from .loader import ProteinLoader
from .pipeline import main
from .runtime import find_dssp_binary, require_dssp_binary
from .slicer import ProteinSlicer
