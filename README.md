# Cooper-Beta

Cooper-Beta detects beta-barrel-like protein chains in PDB, CIF, and mmCIF
structures. It parses structures with Biopython, runs DSSP, slices beta-sheet
C-alpha coordinates, fits ellipses to cross sections, applies geometric
consistency rules, and returns chain-level results.

## Quick Start

Cooper-Beta requires Python 3.10 or newer and a DSSP executable (`mkdssp` or
`dssp`) on `PATH`.

```bash
pip install cooper-beta
cooper-beta --check-env
cooper-beta path/to/structures --out cooper_beta_results.csv
```

If DSSP is installed outside `PATH`, pass its location as a configuration
override:

```bash
cooper-beta path/to/structures runtime.dssp_bin_path=/absolute/path/to/mkdssp
```

## Installation

Install the detector:

```bash
pip install cooper-beta
```

Install optional tools:

```bash
pip install "cooper-beta[eval]"   # pandas for evaluation helpers
pip install "cooper-beta[full]"   # all optional extras
```

For development from a source checkout:

```bash
pip install -e ".[full,dev]"
```

The repository also includes `environment.yml` and `scripts/setup_env.sh` for a
Conda or Mamba environment that installs DSSP from `conda-forge`:

```bash
bash scripts/setup_env.sh --dev
```

## Command Line

Run Cooper-Beta on a single file or a directory:

```bash
cooper-beta path/to/structure.cif --out results.csv
cooper-beta path/to/structures --workers 8 --prepare-workers 8 --out results.csv
```

Useful options:

- `--check-env`: print the Python executable and resolved DSSP executable.
- `--workers`: number of analysis worker processes.
- `--prepare-workers`: number of structure-preparation worker processes.
- `--out`: output CSV path.
- `--version`: print the installed Cooper-Beta version.

Advanced configuration uses Hydra-style `KEY=VALUE` overrides:

```bash
cooper-beta path/to/structures \
  runtime.dssp_bin_path=/absolute/path/to/mkdssp \
  analyzer.rules.angle.max_gap_deg=160 \
  output.summary_limit=-1
```

## Python API

The recommended Python entry point is `detect`, which returns a structured
`PipelineRunResult`. CSV output is written only when `output` is provided or
`write_csv=True`.

```python
from cooper_beta import detect

run = detect(
    "path/to/structures",
    workers=4,
    output="results.csv",
    overrides={"runtime.dssp_bin_path": "/usr/bin/mkdssp"},
)

print(run.result_counts)
for row in run.rows:
    print(row.filename, row.chain, row.result, row.reason)
```

Public interfaces:

- `cooper_beta.detect(...)`: run detection and return structured results.
- `cooper_beta.main(...)`: backward-compatible entry point returning row dicts.
- `cooper_beta.build_config(...)`: build an `AppConfig` from overrides.
- `cooper_beta.PipelineRunResult`: complete run result with `rows`,
  `input_files`, `output_path`, and `result_counts`.
- `cooper_beta.DetectionResult`: one chain-level result row.
- `cooper_beta.ProteinLoader`: parse structures and collect per-chain C-alpha
  and DSSP annotations.
- `cooper_beta.PCAAligner`, `ProteinSlicer`, and `BarrelAnalyzer`: lower-level
  analysis components for custom workflows.

User-facing failures raise Cooper-Beta exceptions such as
`InputValidationError`, `DsspNotFoundError`, `DsspError`, `StructureParseError`,
and `ChainNotFoundError`.

## Output

The result CSV includes one row per chain. Core columns include:

- `filename` and `chain`
- `result`: `BARREL`, `NON_BARREL`, `FILTERED_OUT`, or `ERROR`
- `result_stage` and `reason`
- `decision_score`, `decision_basis`, and `decision_threshold`
- `score_raw` and `score_adjust`
- `valid_layers`, `scored_layers`, `total_layers`, `junk_layers`, and
  `invalid_layers`
- `chain_residues`, `sheet_residues`, and `informative_slices`

Large directory runs use bounded prepare and analysis batches, and the CLI writes
the CSV incrementally. The console summary is capped by default; set
`output.summary_limit=-1` to print every row.

## Evaluation Helpers

Evaluation utilities are available after installing `cooper-beta[eval]`:

```bash
cooper-beta-eval \
  --positives path/to/positive-structures \
  --negatives path/to/negative-structures \
  --save-dir evaluation-results
python -m cooper_beta.evaluation \
  --positives path/to/positive-structures \
  --negatives path/to/negative-structures \
  --ablation
```

The GitHub repository also contains helper scripts for local datasets and
Cooper-Beta CSV outputs. Local structure datasets, manual review notes, and
research-only helper scripts are intentionally excluded from the package
artifacts.

## External Baselines

Evaluation-only adapters for external methods live in `external_methods/`.
The first adapter supports `isitabarrel_structure_map`, which generates
structure-derived contact-map pickles from PDB/CIF/mmCIF inputs, invokes an
external `isitabarrel.py` checkout, and normalizes its `results.tsv` output.
The upstream AGPL-3.0 code is not vendored into Cooper-Beta. These external
baseline adapters are kept for repository-level reproducibility and are
intentionally excluded from PyPI package artifacts.

## Changelog

### 0.1.0

- Initial public release.
- CLI and Python API for PDB/CIF/mmCIF beta-barrel-like chain detection.
- DSSP-backed secondary-structure parsing.
- Ellipse fitting, PCA axis search, geometric rules, and CSV output.
- Evaluation helpers and ablation utilities.

## License

Cooper-Beta is released under the MIT License. See `LICENSE`.
