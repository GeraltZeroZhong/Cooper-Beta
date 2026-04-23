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
install DSSP first or set `runtime.dssp_bin_path=/absolute/path/to/mkdssp`.

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
# Analyze a directory with legacy shortcuts
cooper-beta data/ --workers 8 --prepare-workers 4 --out cooper_beta_results.csv

# Analyze a single file
cooper-beta path/to/structure.pdb

# Hydra-style overrides
cooper-beta input.path=data runtime.workers=8 runtime.prepare_workers=4 output.csv_path=cooper_beta_results.csv

# Module entry point
python -m cooper_beta input.path=data runtime.workers=8 runtime.prepare_workers=4
```

### Legacy entry point
```bash
python main.py data/ 8 4
```

## Configuration
Configuration is now managed by Hydra YAML files under `src/cooper_beta/conf/`.
The default root config composes:
- `runtime`: worker counts, DSSP path, strict DSSP behavior.
- `input`: input path, accepted suffixes, chain/sheet/slice prefilters.
- `slicer`: slice thickness and short-hole filling.
- `analyzer`: ellipse-fit thresholds, decision thresholds, nearest-neighbor rule,
  angle-gap rule, and sequence-angle order rule.
- `output`: result CSV path.

Examples:

```bash
# Point to a custom DSSP binary
cooper-beta runtime.dssp_bin_path=/opt/dssp/bin/mkdssp

# Tune geometric thresholds
cooper-beta analyzer.fit.max_rmse=2.5 analyzer.rules.angle.max_gap_deg=70

# Disable one rule for ablation
cooper-beta analyzer.rules.nearest_neighbor.enabled=false
```

The legacy flat `Config` class still exists for backward compatibility in Python
code, but the supported configuration surface is now Hydra.

## Folder Structure
```
.
├── main.py                  # Legacy entry point
├── pyproject.toml           # Project metadata and dependencies
├── src/
│   └── cooper_beta/
│       ├── __init__.py       # Package exports
│       ├── __main__.py       # Module entry point
│       ├── bootstrap.py      # Thread-environment bootstrap
│       ├── cli.py            # CLI definition
│       ├── config.py         # Hydra schema + legacy compatibility layer
│       ├── conf/             # Hydra YAML configuration tree
│       ├── loader.py         # Structure parsing + DSSP
│       ├── alignment.py      # PCA-based alignment
│       ├── slicer.py         # Coordinate slicing
│       ├── ellipse.py        # Rotated ellipse fitting
│       ├── analysis_utils.py # Geometric helper functions
│       ├── analyzer.py       # Beta-barrel analysis orchestration
│       ├── pipeline.py       # Pipeline orchestration
│       ├── pipeline_workers.py # Preparation + analysis workers
│       ├── results.py        # Summary/CSV output helpers
│       └── evaluation/       # Evaluation and ablation utilities
├── scripts/                  # Helper scripts
└── tests/                    # Test suite
```
