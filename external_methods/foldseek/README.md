# Foldseek global TM-align baseline

This adapter searches query chains against known beta-barrel chains with
[Foldseek](https://github.com/steineggerlab/foldseek). It uses global TM-align
and reports the method as `foldseek_tmalign_structure_search`.

Foldseek is distributed under GPL-3.0. Install the binary and supply its
reference databases separately. The Cooper-Beta adapter is MIT-licensed and
handles chain export, program execution, and result normalization.

## Upstream sources

- Foldseek repository: <https://github.com/steineggerlab/foldseek>
- Foldseek paper: van Kempen M. et al. [Fast and accurate protein structure search with Foldseek](https://doi.org/10.1038/s41587-023-01773-0). Nature Biotechnology. 2024.

## Decision rule

The adapter uses `--alignment-type 1`, Foldseek's global `TMalign` mode, to
compare whole-chain folds. Foldseek's default mode uses local 3Di+AA alignment.

The normalized decision defaults are:

- `score`: `min(qtmscore, ttmscore)` for the best hit that meets both coverage
  thresholds; if none meets them, the best remaining hit's score is retained
  and the result is `NON_BARREL`. A query with no hits receives score zero.
- `result`: `BARREL` when `score >= 0.50`, query coverage is at least `0.60`,
  and target coverage is at least `0.60`; otherwise `NON_BARREL`
- `decision_rule`: recorded in each row so thresholds can be recalibrated

The normalized CSV keeps the best target ID, coverage, `qtmscore`, `ttmscore`,
`alntmscore`, Foldseek score fields, and hit counts.

## Reference database

For repeated evaluation, build a custom reference database once from curated
beta-barrel chain structures:

```bash
foldseek createdb path/to/reference_barrel_chains ref_barrel_db
foldseek createindex ref_barrel_db tmp_index
```

Dataset evaluation requires a reference set and a homology-group manifest.
Assign every query and reference chain to a group derived independently of
the evaluation labels. Before selecting a hit, the evaluator excludes
references that share the query's group or PDB ID, and references with identical
exported single-chain PDB content. SHA-256 comparisons identify identical
exports across source names and group assignments. Homology groups cover
related structures deposited under different PDB IDs.

## Search from structures

When starting from PDB or mmCIF query structures, generate one single-chain PDB
record per analyzable chain and run the baseline:

```bash
python external_methods/foldseek/structure_search.py \
  path/to/query_structures \
  --out-dir eval_outputs/foldseek_tmalign_structure_search \
  --target-db path/to/ref_barrel_db \
  --out eval_outputs/foldseek_tmalign_structure_search.csv
```

You can also point the adapter at reference structures directly and let it run
`foldseek createdb` in the working directory:

```bash
python external_methods/foldseek/structure_search.py \
  path/to/query_structures \
  --out-dir eval_outputs/foldseek_tmalign_structure_search \
  --reference-structures path/to/reference_barrel_chains \
  --create-index \
  --out eval_outputs/foldseek_tmalign_structure_search.csv
```

The generator writes:

- `query_chains/chains/<sample_id>.pdb`: one exported chain per sample
- `query_chains/chain_manifest.csv`: source file and chain metadata
- `query_chains/residue_mapping.csv`: chain-file residue index to source residue
- `foldseek_work/foldseek_hits.tsv`: raw Foldseek TSV before normalization

## Dataset evaluation

The dataset evaluator takes file labels from the positive and negative input
directories. Each file is one observation and is predicted positive when any
eligible chain receives `BARREL`. For chain metrics, provide target-chain
manifests for both positive and negative splits before evaluation.
Each target manifest must have exactly the columns `relative_path,author_chain_id` and
exactly one row for every structure file. Chain metrics use only these selected
chains; partner chains remain unlabeled.

```bash
python external_methods/foldseek/evaluate_dataset.py \
  --positive-dir path/to/positive_structures \
  --negative-dir path/to/negative_structures \
  --reference-dir path/to/reference_barrel_chains \
  --group-manifest path/to/foldseek_family_groups.csv \
  --out-dir eval_outputs \
  --foldseek path/to/foldseek \
  --metric-level both \
  --positive-target-manifest path/to/positive_target_chains.csv \
  --negative-target-manifest path/to/negative_target_chains.csv \
  --tag run-1.0.0
```

Supply the reference set with `--reference-dir`. The group manifest must have
exactly the columns `split,relative_path,author_chain_id,group_id` and cover
every generated chain in the `positive`, `negative`, and `reference` inputs.
Keep the group assignments with the benchmark. `author_chain_id` must contain
the exact, non-blank chain identifier assigned by the depositor.

Each run creates a new UTC timestamped directory under `--out-dir`, preserving
earlier runs. It writes:

- `chain_predictions.csv`: all predictions; `is_target_author_chain` identifies
  the manifest-selected author chain, and only those rows have labels
- `file_results.csv`: file-level labels and predictions
- `target_chain_results.csv`: present only for `chain`/`both` metrics
- `summary.csv` and `summary.md`: metrics at the selected file and/or chain level
- `evaluation_manifest.json`: run status, settings, sampling rules, reference
  exclusion categories and counts, and hashes of inputs, the executable,
  exported chains, and output files

Set scientific labels in the input splits and target manifests before running;
label corrections require a new evaluation. The evaluator stops if predictions
are missing, results are invalid, scores are non-finite, target chains are
unmatched, a structure has no eligible chain, or exported-chain content records
are incomplete or inconsistent.
