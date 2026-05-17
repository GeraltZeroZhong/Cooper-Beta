from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def config_hash(config: object) -> str:
    encoded = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _executable_version(command: str | None) -> str | None:
    if not command:
        return None
    try:
        completed = subprocess.run(
            [command, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output.splitlines()[0] if output else None


def _git_output(args: list[str], repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _input_file_state(path_value: str) -> dict[str, object]:
    path = Path(path_value).expanduser()
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except OSError:
        return {
            "path": str(path),
            "exists": False,
            "size": None,
            "mtime_ns": None,
            "sha256": None,
        }
    return {
        "path": str(resolved),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _file_sha256(resolved),
    }


def _git_untracked_file_state(repo_root: Path) -> list[dict[str, str | None]]:
    output = _git_output(["ls-files", "--others", "--exclude-standard"], repo_root)
    if not output:
        return []
    states: list[dict[str, str | None]] = []
    for relative_path in sorted(path for path in output.splitlines() if path.strip()):
        path = repo_root / relative_path
        if not path.is_file():
            continue
        states.append(
            {
                "path": relative_path,
                "sha256": _file_sha256(path),
            }
        )
    return states


def _git_state() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    commit = _git_output(["rev-parse", "HEAD"], repo_root)
    status = _git_output(["status", "--porcelain"], repo_root)
    diff = _git_output(["diff", "--binary", "HEAD"], repo_root)
    untracked_files = _git_untracked_file_state(repo_root)
    untracked_payload = json.dumps(untracked_files, sort_keys=True, separators=(",", ":"))
    diff_payload = "\n".join(part for part in (status, diff, untracked_payload) if part)
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_porcelain": status,
        "untracked_files": untracked_files,
        "diff_sha256": (
            hashlib.sha256(diff_payload.encode("utf-8")).hexdigest()
            if diff_payload
            else None
        ),
    }


def build_run_manifest(
    *,
    config: object,
    input_files: list[str],
    output_path: str | None,
) -> dict[str, object]:
    dssp_path = getattr(getattr(config, "runtime", None), "dssp_bin_path", None)
    return {
        "schema_version": 1,
        "config_hash": config_hash(config),
        "config": _jsonable(config),
        "input_files": list(input_files),
        "input_file_state": [_input_file_state(path) for path in input_files],
        "output_path": output_path,
        "runtime": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "packages": {
            "cooper-beta": _package_version("cooper-beta"),
            "biopython": _package_version("biopython"),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "opencv-python-headless": _package_version("opencv-python-headless"),
            "hydra-core": _package_version("hydra-core"),
        },
        "executables": {
            "dssp": dssp_path,
            "dssp_version": _executable_version(dssp_path),
        },
        "source": {
            "git": _git_state(),
        },
    }


def write_run_manifest(
    *,
    config: object,
    input_files: list[str],
    output_path: str,
) -> Path:
    manifest_path = Path(f"{output_path}.manifest.json").expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        config=config,
        input_files=input_files,
        output_path=output_path,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path
