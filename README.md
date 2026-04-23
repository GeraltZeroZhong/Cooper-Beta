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

`pip` 只能安装 Python 依赖，不能替你安装 DSSP 这样的系统二进制。Cooper-Beta
现在会在启动分析前主动检查 DSSP；如果没找到，会直接报错并提示用户先安装 DSSP，
或者在 `src/cooper_beta/config.py` 里设置 `DSSP_BIN_PATH`。

### Recommended: one-command setup
仓库现在提供了 `environment.yml` 和 `scripts/setup_env.sh`。优先推荐这条路径，因为
它会通过 `mamba` / `conda` 把 DSSP 一起装进环境里，再安装本项目：

```bash
bash scripts/setup_env.sh
```

如果你还想把测试和 lint 工具也一起装上，可以用：

```bash
bash scripts/setup_env.sh --dev
```

脚本行为：
- 优先使用 `mamba` / `micromamba` / `conda`，并从 `conda-forge` 安装 `dssp`
- 如果没有 conda 类工具，但系统有 `apt-get`，则会回退到 `apt install dssp` + `.venv`
- 最后会提示你如何激活环境

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

如果输出里能看到 Python 版本和 DSSP 路径，说明运行环境已经就绪。

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
