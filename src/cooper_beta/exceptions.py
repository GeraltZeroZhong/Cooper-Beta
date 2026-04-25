from __future__ import annotations


class CooperBetaError(Exception):
    """Base exception for Cooper-Beta user-facing errors."""


class ConfigValidationError(CooperBetaError, ValueError):
    """Raised when a Cooper-Beta configuration contains invalid values."""


class InputValidationError(CooperBetaError, ValueError):
    """Raised when an input path or file set cannot be used for analysis."""


class DsspNotFoundError(CooperBetaError, RuntimeError):
    """Raised when the DSSP executable cannot be found."""


class DsspError(CooperBetaError, RuntimeError):
    """Raised when DSSP fails while preparing a structure."""


class StructureParseError(CooperBetaError, ValueError):
    """Raised when a PDB/mmCIF structure cannot be parsed."""


class ChainNotFoundError(CooperBetaError, KeyError):
    """Raised when a requested chain is not present in a structure."""
