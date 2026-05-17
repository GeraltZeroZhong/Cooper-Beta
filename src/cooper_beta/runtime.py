from __future__ import annotations

import os
import shutil
import sys

from .config import Config
from .exceptions import DsspNotFoundError


def _resolve_executable(candidate: str | None) -> str | None:
    if not candidate:
        return None
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return os.path.abspath(candidate)
    return shutil.which(candidate)


def find_dssp_binary(explicit_path: str | None = None) -> str | None:
    if explicit_path:
        candidate_path = os.path.expanduser(explicit_path)
        if (
            os.path.isabs(candidate_path)
            or os.sep in candidate_path
            or (os.altsep is not None and os.altsep in candidate_path)
        ):
            if os.path.isfile(candidate_path) and os.access(candidate_path, os.X_OK):
                return os.path.abspath(candidate_path)
            return None
        return _resolve_executable(explicit_path)
    for candidate in (Config.DSSP_BIN_PATH,):
        resolved = _resolve_executable(candidate)
        if resolved:
            return resolved
    return shutil.which("mkdssp") or shutil.which("dssp")


def dssp_requirement_message() -> str:
    return (
        "Cooper-Beta requires DSSP (`mkdssp` or `dssp`) before analysis can run.\n"
        "Install DSSP with your system package manager or conda, then make sure the binary is on "
        "PATH.\n"
        "If DSSP is installed in a non-standard location, set "
        "`runtime.dssp_bin_path=/absolute/path/to/mkdssp` in the Hydra config or "
        "`cooper_beta.config.Config.DSSP_BIN_PATH` for legacy code."
    )


def require_dssp_binary(explicit_path: str | None = None) -> str:
    dssp_bin = find_dssp_binary(explicit_path)
    if not dssp_bin:
        if explicit_path:
            raise DsspNotFoundError(
                "Configured DSSP executable was not found or is not executable: "
                f"{explicit_path}"
            )
        raise DsspNotFoundError(dssp_requirement_message())
    return dssp_bin


def runtime_summary(explicit_path: str | None = None, *, require_dssp: bool = True) -> dict[str, str]:
    dssp_path = require_dssp_binary(explicit_path) if require_dssp else find_dssp_binary(explicit_path)
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "dssp": dssp_path or "not found",
    }
