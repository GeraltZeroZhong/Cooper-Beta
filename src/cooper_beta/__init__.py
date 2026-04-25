"""
cooper_beta

A small toolkit/pipeline to detect beta-barrel-like protein chains from PDB/mmCIF structures.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cooper-beta")
except PackageNotFoundError:  # pragma: no cover - editable tree before metadata exists
    __version__ = "0.0.0"

__all__ = [
    "AppConfig",
    "AnalysisReport",
    "Config",
    "ConfigValidationError",
    "ChainNotFoundError",
    "CooperBetaError",
    "DetectionResult",
    "DsspError",
    "DsspNotFoundError",
    "InputValidationError",
    "LayerDiagnostic",
    "PipelineRunResult",
    "PreparedChainPayload",
    "ProteinLoader",
    "PCAAligner",
    "ProteinSlicer",
    "ResidueRecord",
    "StructureParseError",
    "BarrelAnalyzer",
    "build_config",
    "detect",
    "find_dssp_binary",
    "require_dssp_binary",
    "main",
    "__version__",
]

from .alignment import PCAAligner
from .analyzer import BarrelAnalyzer
from .config import AppConfig, Config, build_config
from .exceptions import (
    ChainNotFoundError,
    ConfigValidationError,
    CooperBetaError,
    DsspError,
    DsspNotFoundError,
    InputValidationError,
    StructureParseError,
)
from .loader import ProteinLoader
from .models import (
    AnalysisReport,
    DetectionResult,
    LayerDiagnostic,
    PipelineRunResult,
    PreparedChainPayload,
    ResidueRecord,
)
from .pipeline import detect, main
from .runtime import find_dssp_binary, require_dssp_binary
from .slicer import ProteinSlicer
