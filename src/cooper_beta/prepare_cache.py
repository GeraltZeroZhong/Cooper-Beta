from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

from .config import AppConfig

PREPARE_CACHE_VERSION = 1


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
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _prepare_config_state(cfg: AppConfig) -> dict[str, int | str | bool]:
    return {
        "cache_version": PREPARE_CACHE_VERSION,
        "min_chain_residues": int(cfg.input.min_chain_residues),
        "dssp_bin_path": str(cfg.runtime.dssp_bin_path or ""),
        "fail_on_dssp_error": bool(cfg.runtime.fail_on_dssp_error),
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
    return cache_dir / cache_key[:2] / f"{cache_key}.pkl"


def load_prepare_payloads(file_path: str, cfg: AppConfig) -> list[dict[str, object]] | None:
    if not cfg.runtime.prepare_cache_enabled:
        return None

    cache_path = prepare_cache_path(file_path, cfg)
    try:
        with cache_path.open("rb") as handle:
            payloads = pickle.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, EOFError, pickle.PickleError, AttributeError, TypeError, ValueError):
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None

    if not isinstance(payloads, list):
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
    tmp_path: str | None = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f"{cache_path.stem}.", suffix=".tmp", dir=cache_path.parent)
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payloads, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        tmp_path = None
    except OSError:
        pass
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
