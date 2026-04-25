# Foldseek Global-TMalign Structure-Search Baseline

This adapter treats [Foldseek](https://github.com/steineggerlab/foldseek) as
an external Cooper-Beta evaluation baseline. Cooper-Beta labels this variant
`foldseek_tmalign_structure_search` because it searches query chains against a
reference set of known beta-barrel chains using Foldseek's global TMalign mode.

Foldseek is GPL-3.0, while Cooper-Beta is MIT-licensed, so the upstream binary
and any Foldseek databases are not vendored here. This directory only contains
chain-file generation, invocation, and result normalization.

## Upstream Sources

- Foldseek repository: <https://github.com/steineggerlab/foldseek>
- Foldseek paper: van Kempen M. et al. Nature Biotechnology. 2024.

## Decision Rule

The main baseline uses `--alignment-type 1`, Foldseek's global TMalign mode.
This is intentionally different from Foldseek's default local 3Di+AA alignment:
the Cooper-Beta task asks whether a whole chain has a beta-barrel-like fold, not
whether it contains a locally similar motif.

The normalized decision defaults are:

- `score`: `min(qtmscore, ttmscore)` for the best eligible hit
- `result`: `BARREL` when `score >= 0.50`, query coverage is at least `0.60`,
  and target coverage is at least `0.60`; otherwise `NON_BARREL`
- `decision_rule`: recorded in each row so thresholds can be recalibrated

The normalized CSV keeps the best target id, coverage, `qtmscore`, `ttmscore`,
`alntmscore`, Foldseek score fields, and hit counts.

## Reference Database

For repeated evaluation, build a custom reference database once from curated
canonical beta-barrel chain structures:

```bash
foldseek createdb path/to/reference_barrel_chains ref_barrel_db
foldseek createindex ref_barrel_db tmp_index
```

The reference manifest should exclude evaluation-set chains to avoid leakage.
At minimum, record PDB id, chain id, source dataset, curation notes, and the
Foldseek version used to build the database.

## Structure-to-Search Workflow

When starting from PDB/CIF/mmCIF query structures, generate one single-chain PDB
record per analyzable chain and run the baseline:

```bash
python external_methods/foldseek/structure_search.py \
  path/to/query_structures \
  --out-dir eval_outputs/foldseek_tmalign_structure_search \
  --target-db /path/to/ref_barrel_db \
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

For smoke tests, this repository uses a tiny fake Foldseek runner under
`data/external_methods/foldseek_smoke/` so the adapter can be tested without
installing or vendoring GPL code.

## Cooper-Beta Dataset Evaluation

The dataset evaluator mirrors the existing external baseline table layout and
writes both raw-label and manual-reviewed outputs:

```bash
python external_methods/foldseek/evaluate_dataset.py \
  --positive-dir data/positive \
  --negative-dir data/negative \
  --out-dir eval_outputs/foldseek_tmalign_structure_search_YYYYMMDD_HHMMSS \
  --manual-manifest eval_outputs/notes_aware_manifest_20260425_021152/notes_aware_file_manifest.csv \
  --foldseek tools/foldseek/bin/foldseek \
  --tag YYYYMMDD_HHMMSS
```

By default, `data/positive` is used as the reference barrel structure set. The
evaluator excludes all reference target chains from the same PDB id as each
query before choosing the best hit, so positive examples cannot be classified
only by matching themselves. It writes:

- `eval_chain_results_<tag>_raw.csv`
- `eval_file_results_<tag>_raw.csv`
- `eval_chain_results_<tag>_manual_reviewed.csv`
- `eval_file_results_<tag>_manual_reviewed.csv`
- `foldseek_tmalign_structure_search_summary_<tag>.csv`
- `foldseek_tmalign_structure_search_summary_<tag>.md`

The full global-TMalign run is compute-heavy. On the bundled complete
positive/negative data, the 2026-04-25 run expanded to 1992 positive-reference
chains and 500 negative query chains, used about 627 MB for outputs, and the
Foldseek binary itself used about 28 MB.
