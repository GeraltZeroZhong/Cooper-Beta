from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .config import AppConfig, build_config, sync_legacy_config
from .pipeline_workers import iter_prepared_payload_batches, run_analysis_stream
from .results import ResultCsvWriter, print_results_summary
from .runtime import require_dssp_binary


def discover_input_files(input_path: str, allowed_suffixes: list[str]) -> list[str]:
    """Resolve a directory or single structure file into an explicit file list."""
    path = Path(input_path)
    if path.is_dir():
        files: list[Path] = []
        for suffix in allowed_suffixes:
            files.extend(path.rglob(f"*{suffix}"))
        return [str(file_path) for file_path in sorted(files)]
    return [str(path)]


def os_cpu_count() -> int:
    import os

    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return max(1, len(affinity(0)))
    return os.cpu_count() or 1


def resolve_analysis_worker_count(configured_workers: int | None, cpu_reserve: int) -> int:
    """Choose a sensible default analysis worker count from available CPUs."""
    if configured_workers is not None:
        return max(1, int(configured_workers))

    available_cpus = os_cpu_count()
    return max(1, available_cpus - max(0, cpu_reserve))


def resolve_prepare_worker_count(configured_workers: int | None, analysis_workers: int) -> int:
    if configured_workers is not None:
        return max(1, int(configured_workers))
    return max(1, analysis_workers)


def apply_runtime_overrides(
    cfg: AppConfig,
    *,
    input_path: str | None = None,
    workers: int | None = None,
    prepare_workers: int | None = None,
    out_csv: str | None = None,
) -> AppConfig:
    updated = deepcopy(cfg)
    if input_path is not None:
        updated.input.path = str(input_path)
    if workers is not None:
        updated.runtime.workers = int(workers)
    if prepare_workers is not None:
        updated.runtime.prepare_workers = int(prepare_workers)
    if out_csv is not None:
        updated.output.csv_path = str(out_csv)
    sync_legacy_config(updated)
    return updated


def run_pipeline(cfg: AppConfig) -> list[dict[str, object]]:
    """Run the full beta-barrel detection pipeline from a resolved config."""
    files = discover_input_files(cfg.input.path, cfg.input.allowed_suffixes)
    if Path(cfg.input.path).is_dir() and not files:
        allowed = "/".join(cfg.input.allowed_suffixes)
        print(f"No {allowed} files found in: {cfg.input.path}")
        return []

    cfg.runtime.dssp_bin_path = require_dssp_binary(cfg.runtime.dssp_bin_path)
    sync_legacy_config(cfg)

    analysis_workers = resolve_analysis_worker_count(cfg.runtime.workers, cfg.runtime.cpu_reserve)
    prepare_workers = resolve_prepare_worker_count(cfg.runtime.prepare_workers, analysis_workers)

    print(
        f"\nRunning streaming pipeline with {prepare_workers} prepare worker(s) "
        f"and {analysis_workers} analysis worker(s)..."
    )
    payload_batches = iter_prepared_payload_batches(files, cfg, prepare_workers)
    with ResultCsvWriter(cfg.output.csv_path) as writer:
        results = run_analysis_stream(
            payload_batches,
            cfg,
            analysis_workers,
            on_results=writer.write_rows,
        )

    if not results:
        print("No analyzable chain payloads were produced.")
        print(f"\nResults written to: {cfg.output.csv_path}")
        return []

    print_results_summary(
        results,
        cfg.output.csv_path,
        summary_limit=cfg.output.summary_limit,
        write_csv=False,
    )
    return results


def main(
    input_path: str | None = None,
    *,
    workers: int | None = None,
    prepare_workers: int | None = None,
    out_csv: str | None = None,
    cfg: AppConfig | None = None,
    overrides: dict[str, object] | list[str] | None = None,
) -> list[dict[str, object]]:
    """
    Backward-compatible entry point with optional Hydra overrides.
    """
    resolved_cfg = cfg or build_config(overrides)
    resolved_cfg = apply_runtime_overrides(
        resolved_cfg,
        input_path=input_path,
        workers=workers,
        prepare_workers=prepare_workers,
        out_csv=out_csv,
    )
    return run_pipeline(resolved_cfg)
