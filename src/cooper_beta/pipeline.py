from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .config import AppConfig, build_config, sync_legacy_config
from .pipeline_workers import collect_payloads, run_analysis
from .results import print_results_summary
from .runtime import require_dssp_binary


def discover_input_files(input_path: str, allowed_suffixes: list[str]) -> list[str]:
    """Resolve a directory or single structure file into an explicit file list."""
    path = Path(input_path)
    if path.is_dir():
        files: list[Path] = []
        for suffix in allowed_suffixes:
            files.extend(path.glob(f"*{suffix}"))
        return [str(file_path) for file_path in sorted(files)]
    return [str(path)]


def resolve_worker_count(configured_workers: int | None, cpu_reserve: int) -> int:
    """Choose a sensible default worker count from available CPUs."""
    if configured_workers is not None:
        return max(1, int(configured_workers))

    cpu_count = os_cpu_count()
    return max(1, cpu_count - max(0, cpu_reserve))


def os_cpu_count() -> int:
    import os

    return os.cpu_count() or 1


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

    require_dssp_binary(cfg.runtime.dssp_bin_path)

    analysis_workers = resolve_worker_count(cfg.runtime.workers, cfg.runtime.cpu_reserve)
    prepare_workers = cfg.runtime.prepare_workers
    if prepare_workers is None:
        prepare_workers = analysis_workers
    prepare_workers = max(1, int(prepare_workers))

    payloads = collect_payloads(files, cfg, prepare_workers)
    if not payloads:
        print("No analyzable chain payloads were produced.")
        return []

    print(f"\nRunning analysis with {analysis_workers} worker(s)...")
    results = run_analysis(payloads, cfg, analysis_workers)
    print_results_summary(results, cfg.output.csv_path)
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
