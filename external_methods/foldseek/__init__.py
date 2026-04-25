"""Foldseek global-TMalign structure-search baseline adapter."""

from .runner import (
    BASELINE_NAME,
    DEFAULT_ALIGNMENT_TYPE,
    DEFAULT_FORMAT_FIELDS,
    DEFAULT_MIN_QUERY_COVERAGE,
    DEFAULT_MIN_TARGET_COVERAGE,
    DEFAULT_SCORE_MODE,
    DEFAULT_SCORE_THRESHOLD,
    FoldseekHit,
    FoldseekResult,
    load_hits_tsv,
    run_baseline,
    summarize_hits,
    write_results_csv,
)
from .structure_search import StructureSearchBaselineRun, run_structure_search_baseline
from .structures import (
    DEFAULT_MIN_RESIDUES,
    GeneratedStructureChain,
    GeneratedStructureSet,
    foldseek_query_aliases,
    generate_structure_chains,
)

__all__ = [
    "BASELINE_NAME",
    "DEFAULT_ALIGNMENT_TYPE",
    "DEFAULT_FORMAT_FIELDS",
    "DEFAULT_MIN_QUERY_COVERAGE",
    "DEFAULT_MIN_RESIDUES",
    "DEFAULT_MIN_TARGET_COVERAGE",
    "DEFAULT_SCORE_MODE",
    "DEFAULT_SCORE_THRESHOLD",
    "FoldseekHit",
    "FoldseekResult",
    "GeneratedStructureChain",
    "GeneratedStructureSet",
    "StructureSearchBaselineRun",
    "foldseek_query_aliases",
    "generate_structure_chains",
    "load_hits_tsv",
    "run_baseline",
    "run_structure_search_baseline",
    "summarize_hits",
    "write_results_csv",
]
