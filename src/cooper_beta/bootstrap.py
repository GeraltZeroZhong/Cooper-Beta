from __future__ import annotations

import os

from .constants import THREAD_ENV_DEFAULTS


def configure_thread_environment() -> None:
    """Limit BLAS/OpenMP thread fan-out when multiprocessing is used."""
    for env_name, env_value in THREAD_ENV_DEFAULTS.items():
        os.environ.setdefault(env_name, env_value)
