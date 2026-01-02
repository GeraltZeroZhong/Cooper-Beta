# -*- coding: utf-8 -*-
"""
cooper_beta

A small toolkit/pipeline to detect beta-barrel-like protein chains from PDB/mmCIF structures.
"""
from __future__ import annotations

__all__ = [
    "Config",
    "ProteinLoader",
    "PCAAligner",
    "ProteinSlicer",
    "BarrelAnalyzer",
    "main",
]

from .config import Config
from .loader import ProteinLoader
from .alignment import PCAAligner
from .slicer import ProteinSlicer
from .analyzer import BarrelAnalyzer
from .pipeline import main
