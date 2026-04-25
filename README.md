# Cooper-Beta

Cooper-Beta detects beta-barrel-like protein chains in PDB, CIF, and mmCIF
structures. It parses structures with Biopython, runs DSSP, slices beta-sheet
C-alpha coordinates, fits ellipses to cross sections, applies geometric
consistency rules, and writes a chain-level CSV.

## Quick Start

Cooper-Beta requires Python 3.10 or newer and a DSSP executable
(`mkdssp` or `dssp`) on `PATH`.

```bash
pip install cooper-beta
cooper-beta --check-env
cooper-beta path/to/structures --out cooper_beta_results.csv
```

For a source checkout:

```bash
pip install -e ".[full]"
cooper-beta path/to/structures --workers 8 --prepare-workers 8
```

If DSSP is not on `PATH`, pass its location with Hydra-style overrides:

```bash
cooper-beta path/to/structures runtime.dssp_bin_path=/absolute/path/to/mkdssp
```

## Installation

Install from PyPI:

```bash
pip install cooper-beta
```

Install optional evaluation and progress-reporting dependencies:

```bash
pip install "cooper-beta[full]"
```

Install development tools from a source checkout:

```bash
pip install -e ".[full,dev]"
```

This repository also includes `environment.yml` and `scripts/setup_env.sh` for a
Conda/Mamba environment that installs DSSP from `conda-forge`:

```bash
bash scripts/setup_env.sh --dev
```

## Minimal Examples

Command line:

```bash
cooper-beta data/ --out results.csv
python -m cooper_beta data/ runtime.workers=4 output.csv_path=results.csv
```

Python:

```python
from cooper_beta import build_config, main

cfg = build_config({"runtime.dssp_bin_path": "/usr/bin/mkdssp"})
rows = main("path/to/structures", workers=4, out_csv="results.csv", cfg=cfg)
print(rows[0])
```

## Main API

- `cooper_beta.main(...)`: run the full detection pipeline.
- `cooper_beta.build_config(...)`: build an `AppConfig` from Hydra-style or
  legacy overrides.
- `cooper_beta.ProteinLoader`: parse structures and collect per-chain C-alpha
  and DSSP annotations.
- `cooper_beta.PCAAligner`: align chain coordinates with PCA.
- `cooper_beta.ProteinSlicer`: convert aligned beta-sheet segments into slice
  intersections.
- `cooper_beta.BarrelAnalyzer`: score slice geometry and produce per-layer
  diagnostics.

The CLI accepts both legacy shortcuts (`--workers`, `--prepare-workers`, `--out`)
and Hydra-style overrides such as `analyzer.rules.angle.max_gap_deg=160`.
Large directory runs use bounded prepare/analysis batches and write the output
CSV incrementally. The console summary is capped by default; set
`output.summary_limit=-1` to print every row.

## Output

The result CSV includes one row per chain with:

- `result`: `BARREL`, `NON_BARREL`, `FILTERED_OUT`, or `ERROR`
- `decision_score`, `decision_basis`, and `decision_threshold`
- raw and adjusted scores
- valid, scored, junk, invalid, and total layer counts
- chain, sheet, and informative-slice counts
- a short reason for the final decision

## Changelog

### 0.1.0

- Initial public release.
- CLI and Python API for PDB/CIF/mmCIF beta-barrel-like chain detection.
- DSSP-backed secondary-structure parsing.
- Ellipse fitting, PCA axis search, geometric rules, and CSV output.
- Evaluation helpers and ablation utilities.

## License

Cooper-Beta is released under the MIT License. See `LICENSE`.
