# Cooper-Beta

## Overview
Cooper-Beta is a Python pipeline that detects beta-barrel-like protein chains from PDB/mmCIF
structures. It parses structures, runs DSSP to identify secondary structure, slices aligned
coordinates, evaluates elliptical fits per slice, and produces a CSV summary indicating
whether each chain is classified as a beta-barrel.

## Features
- Supports PDB, CIF, and mmCIF inputs (single file or directory).
- Parallel preparation (DSSP/structure parsing) and analysis phases.
- Robust beta-barrel scoring with adjustable rules (nearest-neighbor spacing, angular coverage,
  and sequence/angle order consistency).
- CSV output with per-chain classification metrics and failure reasons.
- CLI and module entry points (`cooper-beta`, `python -m cooper_beta`).

## Installation
### Requirements
- Python 3.10+
- DSSP binary (`mkdssp` or `dssp`) available on your PATH
- Core dependencies: NumPy, SciPy, Biopython
- Optional: pandas and tqdm for richer summaries/progress output

`pip` can install Python packages, but it cannot install system binaries such as
DSSP for you. Cooper-Beta now checks for DSSP before analysis starts. If the
binary is missing, the program fails fast with a clear message telling you to
install DSSP first or set `src/cooper_beta/config.py -> Config.DSSP_BIN_PATH`.

### Recommended: one-command setup
This repository ships with `environment.yml` and `scripts/setup_env.sh`. This is
the recommended setup path because it installs DSSP with `mamba` / `conda`
alongside the Python environment and then installs this project:

```bash
bash scripts/setup_env.sh
```

If you also want the test and lint tooling, run:

```bash
bash scripts/setup_env.sh --dev
```

What the setup script does:
- Prefer `mamba`, `micromamba`, or `conda`, and install `dssp` from `conda-forge`
- Fall back to `apt install dssp` plus a local `.venv` if no conda-style tool is available but `apt-get` exists
- Print the exact activation command at the end

### Install from source
```bash
pip install -e .
```

### Optional extras
```bash
pip install -e ".[full]"
```

### Validate the runtime environment
```bash
cooper-beta --check-env
```

If the output shows both the Python version and the DSSP path, the runtime
environment is ready.

## Usage
### Command-line
```bash
# Analyze a directory (default: data/)
cooper-beta data/ --workers 8 --prepare-workers 4 --out cooper_beta_results.csv

# Analyze a single file
cooper-beta path/to/structure.pdb

# Module entry point
python -m cooper_beta data/ --workers 8 --prepare-workers 4
```

### Legacy entry point
```bash
python main.py data/ 8 4
```

## Configuration
Configuration lives in `src/cooper_beta/config.py` (`Config` class). Common settings include:
- `DSSP_BIN_PATH`: explicit path to DSSP if it is not discoverable on PATH.
- Slicing controls: `SLICE_STEP_SIZE`.
- Fitting controls: `MIN_POINTS_PER_SLICE`, `MAX_FIT_RMSE`, `MIN_AXIS`, `MAX_AXIS`,
  `MAX_FLATTENING`.
- Decision logic: `BARREL_VALID_RATIO`, `USE_ADJUSTED_SCORE`, `MIN_SCORED_LAYER_FRAC`.
- Rules for quality control: `NN_RULE_ENABLED`, `ANGLE_RULE_ENABLED`,
  `ANGLE_ORDER_RULE_ENABLED` and their thresholds.

Adjust these constants to tune recall/precision for your dataset.

## Folder Structure
```
.
├── main.py                  # Legacy entry point
├── pyproject.toml           # Project metadata and dependencies
├── src/
│   └── cooper_beta/
│       ├── __init__.py       # Package exports
│       ├── __main__.py       # Module entry point
│       ├── cli.py            # CLI definition
│       ├── config.py         # Tunable parameters
│       ├── loader.py         # Structure parsing + DSSP
│       ├── alignment.py      # PCA-based alignment
│       ├── slicer.py         # Coordinate slicing
│       ├── analyzer.py       # Beta-barrel analysis
│       └── pipeline.py       # Pipeline orchestration
├── scripts/                  # Helper scripts
└── tests/                    # Test suite
```
