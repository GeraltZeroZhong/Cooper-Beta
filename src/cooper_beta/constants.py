from __future__ import annotations

from math import pi

DEFAULT_INPUT_PATH = ""
DEFAULT_OUTPUT_CSV = "cooper_beta_results.csv"
DEFAULT_ALLOWED_SUFFIXES = (".pdb", ".cif", ".mmcif")
DEFAULT_SLICE_STEP_SIZE = 1.0
DEFAULT_FILL_SHEET_HOLE_LENGTH = 1

DEFAULT_MIN_CHAIN_RESIDUES = 20
DEFAULT_MIN_SHEET_RESIDUES = 10
DEFAULT_MIN_INFORMATIVE_SLICES = 5
MIN_NEAREST_NEIGHBOR_POINTS = 3
MIN_ANGULAR_GAP_POINTS = 5
MIN_SEQUENCE_ANGLE_ORDER_POINTS = 6

SEQUENCE_INTERSECTION_OFFSET = 0.5
ROBUST_SIGMA_SCALE = 1.4826
THREE_SIGMA_MULTIPLIER = 3.0
RAD_TO_DEG = 180.0 / pi
FULL_ROTATION_DEG = 360.0

EPSILON = 1e-12
TOLERANCE = 1e-9
COVARIANCE_FLOOR = 1e-12

RESULT_BARREL = "BARREL"
RESULT_NON_BARREL = "NON_BARREL"
RESULT_FILTERED_OUT = "FILTERED_OUT"
RESULT_ERROR = "ERROR"
LEGACY_RESULT_SKIP = "SKIP"

DEFAULT_RESULT_COLUMNS = (
    "filename",
    "chain",
    "result",
    "result_stage",
    "decision_score",
    "decision_basis",
    "decision_threshold",
    "score_raw",
    "score_adjust",
    "valid_layers",
    "scored_layers",
    "total_layers",
    "valid_layer_frac",
    "scored_layer_frac",
    "junk_layers",
    "invalid_layers",
    "avg_radius",
    "chain_residues",
    "sheet_residues",
    "informative_slices",
    "reason",
    "all_adjusted_layers",
    "all_layers",
)

DEFAULT_SUMMARY_COLUMNS = (
    "filename",
    "chain",
    "result",
    "result_stage",
    "decision_score",
    "decision_basis",
    "layer_counts",
    "reason",
)

SUMMARY_COLUMN_WIDTHS = {
    "filename": 20,
    "chain": 5,
    "result": 14,
    "result_stage": 10,
    "decision_score": 8,
    "decision_basis": 8,
    "layer_counts": 18,
    "reason": 56,
}

THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
