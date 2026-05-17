from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .bootstrap import configure_thread_environment
from .config import AppConfig, build_config, sync_legacy_config, validate_config
from .exceptions import InputValidationError
from .models import PipelineRunResult

configure_thread_environment()

from .pipeline_workers import iter_prepared_payload_batches, run_analysis_stream  # noqa: E402
from .provenance import write_run_manifest  # noqa: E402
from .results import ResultCsvWriter, print_results_summary, write_results_csv  # noqa: E402
from .runtime import require_dssp_binary  # noqa: E402


def discover_input_files(
    input_path: str,
    allowed_suffixes: list[str],
    *,
    strict: bool = False,
) -> list[str]:
    """Resolve a directory or single structure file into an explicit file list."""
    if not str(input_path).strip():
        raise InputValidationError("Input path is required.")
    path = Path(input_path).expanduser()
    normalized_suffixes = tuple(str(suffix).lower() for suffix in allowed_suffixes)
    if not path.exists():
        if strict:
            raise InputValidationError(f"Input path does not exist: {path}")
        return [str(path)]

    if path.is_dir():
        files = sorted(
            file_path
            for file_path in path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in normalized_suffixes
        )
        if strict and not files:
            allowed = ", ".join(allowed_suffixes)
            raise InputValidationError(f"No structure files ({allowed}) were found in: {path}")
        return [str(file_path) for file_path in sorted(files)]

    if strict and normalized_suffixes and path.suffix.lower() not in normalized_suffixes:
        allowed = ", ".join(allowed_suffixes)
        raise InputValidationError(
            f"Input file has unsupported suffix {path.suffix!r}. Expected one of: {allowed}."
        )
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


def _prepare_error_rows(errors: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for error in errors:
        source, _, detail = error.partition(":")
        source = source.strip()
        rows.append(
            {
                "filename": Path(source).name if source else "",
                "source_path": source,
                "chain": "",
                "result": "ERROR",
                "result_stage": "prepare",
                "reason": detail.strip() or error,
            }
        )
    return rows


def _ordered_result_rows(
    rows: list[dict[str, object]],
    input_files: list[str],
) -> list[dict[str, object]]:
    """Return rows in input file order, then chain order, for reproducible output."""
    file_order: dict[str, int] = {}
    for index, file_path in enumerate(input_files):
        basename = Path(file_path).name
        file_order.setdefault(str(file_path), index)
        file_order.setdefault(str(Path(file_path).expanduser().resolve()), index)
        file_order.setdefault(basename, index)

    def sort_key(row: dict[str, object]) -> tuple[int, str, str, str, str]:
        filename = str(row.get("filename", ""))
        source_path = str(row.get("source_path", ""))
        primary_id = source_path or filename
        return (
            file_order.get(primary_id, file_order.get(filename, len(file_order))),
            source_path,
            filename,
            str(row.get("chain", "")),
            str(row.get("result_stage", "")),
        )

    return sorted(rows, key=sort_key)


def run_pipeline_result(
    cfg: AppConfig,
    *,
    write_csv: bool = True,
    print_summary: bool = True,
    strict_input: bool = False,
    show_progress: bool = True,
) -> PipelineRunResult:
    """Run the full beta-barrel detection pipeline and return structured results."""
    configure_thread_environment()
    cfg = deepcopy(cfg)
    validate_config(cfg)

    files = discover_input_files(cfg.input.path, cfg.input.allowed_suffixes, strict=strict_input)
    if Path(cfg.input.path).expanduser().is_dir() and not files:
        allowed = "/".join(cfg.input.allowed_suffixes)
        if print_summary:
            print(f"No {allowed} files found in: {cfg.input.path}")
        if write_csv:
            with ResultCsvWriter(cfg.output.csv_path):
                pass
            write_run_manifest(config=cfg, input_files=files, output_path=cfg.output.csv_path)
        output_path = cfg.output.csv_path if write_csv else None
        return PipelineRunResult.from_rows([], input_files=files, output_path=output_path, config=cfg)

    cfg.runtime.dssp_bin_path = require_dssp_binary(cfg.runtime.dssp_bin_path)
    sync_legacy_config(cfg)

    analysis_workers = resolve_analysis_worker_count(cfg.runtime.workers, cfg.runtime.cpu_reserve)
    prepare_workers = resolve_prepare_worker_count(cfg.runtime.prepare_workers, analysis_workers)
    if write_csv and Path(cfg.output.csv_path).expanduser().is_dir():
        raise InputValidationError(
            f"Output CSV path points to a directory: {cfg.output.csv_path}"
        )

    if print_summary:
        print(
            f"\nRunning streaming pipeline with {prepare_workers} prepare worker(s) "
            f"and {analysis_workers} analysis worker(s)..."
        )
    prepare_errors: list[str] = []

    def record_prepare_errors(errors: list[str]) -> None:
        prepare_errors.extend(errors)

    payload_batches = iter_prepared_payload_batches(
        files,
        cfg,
        prepare_workers,
        on_errors=record_prepare_errors,
        show_progress=show_progress,
    )
    results = run_analysis_stream(
        payload_batches,
        cfg,
        analysis_workers,
        show_progress=show_progress,
    )
    prepare_rows = _prepare_error_rows(prepare_errors)

    all_results = _ordered_result_rows([*results, *prepare_rows], files)
    if prepare_rows and not results:
        if print_summary:
            print_results_summary(
                all_results,
                cfg.output.csv_path,
                summary_limit=cfg.output.summary_limit,
                write_csv=write_csv,
                output_written=write_csv,
            )
        elif write_csv:
            write_results_csv(all_results, cfg.output.csv_path)
        if write_csv:
            write_run_manifest(config=cfg, input_files=files, output_path=cfg.output.csv_path)
        raise InputValidationError(
            f"All {len(files)} input file(s) failed during preparation."
        )

    if not all_results:
        if print_summary:
            print("No analyzable chain payloads were produced.")
            if write_csv:
                print(f"\nResults written to: {cfg.output.csv_path}")
        if write_csv:
            write_results_csv([], cfg.output.csv_path)
            write_run_manifest(config=cfg, input_files=files, output_path=cfg.output.csv_path)
        output_path = cfg.output.csv_path if write_csv else None
        return PipelineRunResult.from_rows([], input_files=files, output_path=output_path, config=cfg)

    if print_summary:
        print_results_summary(
            all_results,
            cfg.output.csv_path,
            summary_limit=cfg.output.summary_limit,
            write_csv=write_csv,
            output_written=write_csv,
        )
    elif write_csv:
        write_results_csv(all_results, cfg.output.csv_path)
    if write_csv:
        write_run_manifest(config=cfg, input_files=files, output_path=cfg.output.csv_path)
    output_path = cfg.output.csv_path if write_csv else None
    return PipelineRunResult.from_rows(
        all_results,
        input_files=files,
        output_path=output_path,
        config=cfg,
    )


def run_pipeline(cfg: AppConfig) -> list[dict[str, object]]:
    """Run the full beta-barrel detection pipeline from a resolved config."""
    return run_pipeline_result(cfg).raw_rows()


def detect(
    input_path: str,
    *,
    config: AppConfig | None = None,
    cfg: AppConfig | None = None,
    overrides: dict[str, object] | list[str] | None = None,
    workers: int | None = None,
    prepare_workers: int | None = None,
    output: str | None = None,
    write_csv: bool | None = None,
    print_summary: bool = False,
    show_progress: bool | None = None,
    strict_input: bool = True,
) -> PipelineRunResult:
    """
    Public Python API for running detection with structured results.

    CSV output is written only when ``output`` is provided or ``write_csv=True``.
    """
    if config is not None and cfg is not None:
        raise TypeError("Pass only one of `config` or `cfg`.")
    if overrides is not None and (config is not None or cfg is not None):
        raise TypeError("Pass `overrides` only when Cooper-Beta builds the config for you.")
    resolved_cfg = config or cfg or build_config(overrides)
    if workers is None and resolved_cfg.runtime.workers is None:
        workers = 1
    if prepare_workers is None and resolved_cfg.runtime.prepare_workers is None:
        prepare_workers = 1
    resolved_cfg = apply_runtime_overrides(
        resolved_cfg,
        input_path=input_path,
        workers=workers,
        prepare_workers=prepare_workers,
        out_csv=output,
    )
    should_write_csv = bool(output) if write_csv is None else bool(write_csv)
    should_show_progress = bool(print_summary) if show_progress is None else bool(show_progress)
    return run_pipeline_result(
        resolved_cfg,
        write_csv=should_write_csv,
        print_summary=print_summary,
        show_progress=should_show_progress,
        strict_input=strict_input,
    )


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
    if cfg is not None and overrides is not None:
        raise TypeError("Pass `overrides` only when Cooper-Beta builds the config for you.")
    resolved_cfg = cfg or build_config(overrides)
    resolved_cfg = apply_runtime_overrides(
        resolved_cfg,
        input_path=input_path,
        workers=workers,
        prepare_workers=prepare_workers,
        out_csv=out_csv,
    )
    return run_pipeline_result(
        resolved_cfg,
        write_csv=True,
        print_summary=True,
        strict_input=True,
        show_progress=True,
    ).raw_rows()
