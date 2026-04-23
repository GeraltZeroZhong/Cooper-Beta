from __future__ import annotations

from cooper_beta.pipeline import resolve_analysis_worker_count, resolve_prepare_worker_count


def test_prepare_worker_default_heuristic_follows_analysis_workers():
    assert resolve_prepare_worker_count(None, 1) == 1
    assert resolve_prepare_worker_count(None, 2) == 2
    assert resolve_prepare_worker_count(None, 4) == 4
    assert resolve_prepare_worker_count(None, 7) == 7
    assert resolve_prepare_worker_count(None, 8) == 8


def test_analysis_worker_override_is_respected():
    assert resolve_analysis_worker_count(6, cpu_reserve=1) == 6
