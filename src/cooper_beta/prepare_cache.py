from __future__ import annotations

import hashlib
import json
import os
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import AppConfig

PREPARE_CACHE_VERSION = 2
PREPARE_PRODUCER_SCHEMA_VERSION = 1
_RESIDUE_CACHE_KEYS = (
    "res_id",
    "resseq",
    "icode",
    "hetfield",
    "res_uid",
    "chain",
    "coord",
    "is_sheet",
)
_RES_UID_CACHE_KEYS = ("chain", "hetfield", "resseq", "icode")


def default_prepare_cache_dir() -> str:
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(xdg_cache_home) if xdg_cache_home else (Path.home() / ".cache")
    return str(cache_root / "cooper-beta" / "prepare")


def resolve_prepare_cache_dir(configured_dir: str | None = None) -> Path:
    cache_dir = configured_dir or default_prepare_cache_dir()
    return Path(cache_dir).expanduser().resolve()


def _file_state(file_path: str) -> dict[str, int | str]:
    path = Path(file_path).expanduser().resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _executable_state(path_value: str | None) -> dict[str, int | str] | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    try:
        resolved = path.resolve()
        stat = resolved.stat()
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return {"path": str(path_value)}
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _prepare_config_state(cfg: AppConfig) -> dict[str, object]:
    dssp_path = cfg.runtime.dssp_bin_path
    if not dssp_path:
        from .runtime import find_dssp_binary

        dssp_path = find_dssp_binary()
    return {
        "cache_version": PREPARE_CACHE_VERSION,
        "dssp_bin_path": str(dssp_path or ""),
        "dssp_bin_state": _executable_state(dssp_path),
        "fail_on_dssp_error": bool(cfg.runtime.fail_on_dssp_error),
        "producer": {
            "schema_version": PREPARE_PRODUCER_SCHEMA_VERSION,
            "cooper_beta": _package_version("cooper-beta"),
            "biopython": _package_version("biopython"),
        },
    }


def build_prepare_cache_key(file_path: str, cfg: AppConfig) -> str:
    payload = {
        "file": _file_state(file_path),
        "prepare": _prepare_config_state(cfg),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_cache_path(file_path: str, cfg: AppConfig) -> Path:
    cache_dir = resolve_prepare_cache_dir(cfg.runtime.prepare_cache_dir)
    cache_key = build_prepare_cache_key(file_path, cfg)
    return cache_dir / cache_key[:2] / f"{cache_key}.json"


def _normalize_residue(residue: object) -> dict[str, object] | None:
    if not isinstance(residue, dict):
        return None
    coord = residue.get("coord")
    try:
        coord_list = [float(value) for value in coord]  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
    if len(coord_list) != 3:
        return None

    normalized = {key: residue[key] for key in _RESIDUE_CACHE_KEYS if key in residue}
    normalized["coord"] = coord_list
    normalized["is_sheet"] = bool(residue.get("is_sheet", False))
    if "res_id" in normalized:
        try:
            normalized["res_id"] = int(normalized["res_id"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    if "resseq" in normalized:
        try:
            normalized["resseq"] = int(normalized["resseq"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    if "chain" in normalized:
        normalized["chain"] = str(normalized["chain"])
    if "icode" in normalized:
        normalized["icode"] = str(normalized["icode"])
    if "hetfield" in normalized:
        normalized["hetfield"] = str(normalized["hetfield"])
    res_uid = normalized.get("res_uid")
    if isinstance(res_uid, dict):
        uid = {key: res_uid[key] for key in _RES_UID_CACHE_KEYS if key in res_uid}
        if "resseq" in uid:
            try:
                uid["resseq"] = int(uid["resseq"])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
        for key in ("chain", "hetfield", "icode"):
            if key in uid:
                uid[key] = str(uid[key])
        normalized["res_uid"] = uid
    elif "res_uid" in normalized:
        del normalized["res_uid"]
    return normalized


def _normalize_payloads(payloads: object) -> list[dict[str, object]] | None:
    if not isinstance(payloads, list):
        return None

    normalized_payloads: list[dict[str, object]] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            return None
        residues = payload.get("residues_data")
        if not isinstance(residues, list):
            return None
        normalized_residues = []
        for residue in residues:
            normalized_residue = _normalize_residue(residue)
            if normalized_residue is None:
                return None
            normalized_residues.append(normalized_residue)
        normalized_payloads.append(
            {
                "filename": str(payload.get("filename", "")),
                "source_path": str(payload.get("source_path", "")),
                "chain": str(payload.get("chain", "")),
                "residues_data": normalized_residues,
            }
        )
    return normalized_payloads


def load_prepare_payloads(file_path: str, cfg: AppConfig) -> list[dict[str, object]] | None:
    if not cfg.runtime.prepare_cache_enabled:
        return None

    cache_path = prepare_cache_path(file_path, cfg)
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            envelope = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None

    if not isinstance(envelope, dict) or envelope.get("cache_version") != PREPARE_CACHE_VERSION:
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None

    payloads = _normalize_payloads(envelope.get("payloads"))
    if payloads is None:
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None
    return payloads


def store_prepare_payloads(file_path: str, cfg: AppConfig, payloads: list[dict[str, object]]) -> None:
    if not cfg.runtime.prepare_cache_enabled:
        return

    cache_path = prepare_cache_path(file_path, cfg)
    normalized_payloads = _normalize_payloads(payloads)
    if normalized_payloads is None:
        return

    tmp_path: str | None = None
    try:
        cache_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            cache_path.parent.chmod(0o700)
        except OSError:
            pass
        fd, tmp_path = tempfile.mkstemp(prefix=f"{cache_path.stem}.", suffix=".tmp", dir=cache_path.parent)
        envelope = {
            "cache_version": PREPARE_CACHE_VERSION,
            "payloads": normalized_payloads,
        }
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, separators=(",", ":"))
        os.replace(tmp_path, cache_path)
        tmp_path = None
    except (OSError, TypeError, ValueError):
        pass
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
