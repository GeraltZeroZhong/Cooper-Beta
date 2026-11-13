# PRED-TMBB2 single-sequence JUCHMME baseline

This adapter runs PRED-TMBB2 single-sequence topology prediction with local
JUCHMME and derives a binary decision from the predicted topology. It reports
the method as `pred_tmbb2_single_juchmme`.

Install the GPL-3.0 JUCHMME Java package and trained parameter files separately.
The MIT-licensed Cooper-Beta adapter handles FASTA generation, program
execution, and result normalization.

## Upstream sources

- PRED-TMBB2 web page: <https://hannibal.dib.uth.gr/PRED-TMBB2/>
- JUCHMME source and releases: <https://github.com/pbagos/juchmme>
- PRED-TMBB2 paper: Tsirigos KD, Elofsson A, Bagos PG. [PRED-TMBB2: improved topology prediction and detection of beta-barrel outer membrane proteins](https://doi.org/10.1093/bioinformatics/btw444). Bioinformatics. 2016.

## Local installation

Download and unpack a JUCHMME release, then provide the release root with
`--juchmme-dir` or the `PRED_TMBB2_JUCHMME_DIR` environment variable. The
adapter expects the standard release paths:

- `bin/`: compiled Java classes
- `models/tmbb2.mdel`
- `tables/A_TMBB2_TRAINED`
- `tables/E_TMBB2_TRAINED`
- `conf/conf.tmbb`

## Decision rule

JUCHMME emits topology strings and scores. The published PRED-TMBB2
discrimination pipeline can also combine these with signal-peptide and
Pfam/OMPdb features. This adapter classifies chains from the predicted topology:

- `result`: `BARREL` when the selected topology field has at least three
  predicted `M` runs, otherwise `NON_BARREL`
- `score`: number of predicted `M` runs (membrane beta-strand segments)
- default selected topology: `LP`

The normalized CSV also keeps `logodds`, reliability, algorithm score, sequence
length, and the raw topology string so downstream analyses can use a different
threshold if needed.

Dataset evaluation requires both the reported sequence length and topology
length to match the declared complete polymer-sequence length. It recomputes
the strand count and binary decision from the topology and stops if these
values disagree with the upstream records.

## Run existing FASTA sequences

```bash
python external_methods/pred_tmbb2/runner.py \
  path/to/sequences.fasta \
  --juchmme-dir path/to/juchmme \
  --out eval_outputs/pred_tmbb2_single_juchmme.csv
```

## Sequences from structures

When starting from PDB or mmCIF structures, generate one chain-level FASTA
record per declared protein polymer and run the baseline:

```bash
python external_methods/pred_tmbb2/structure_sequence.py \
  path/to/structures \
  --out-dir eval_outputs/pred_tmbb2_single_juchmme \
  --juchmme-dir path/to/juchmme \
  --out eval_outputs/pred_tmbb2_single_juchmme.csv
```

The generator writes:

- `sequences.fasta`: one FASTA record per chain
- `residue_mapping.csv`: every complete polymer-sequence position, its source
  monomer, exact `author_chain_id`, label-chain and entity identifiers, sequence
  source, and sequence SHA-256
- `juchmme_work/`: upstream working directory

The generator uses complete declared polymer sequences. PDB input must
contain complete `SEQRES` records. mmCIF input must contain consistent
`_entity_poly`, `_entity_poly_seq`, and `_struct_asym` categories, plus an exact
label-to-author-chain mapping from `_pdbx_poly_seq_scheme` and/or `_atom_site`.
Missing or ambiguous sequence declarations stop sequence generation.

## Dataset evaluation

Dataset evaluation takes file labels from the positive and negative input
directories. Each file is one observation and is predicted positive when any
generated chain receives `BARREL`:

```bash
python external_methods/pred_tmbb2/evaluate_dataset.py \
  --positive-dir path/to/positive_structures \
  --negative-dir path/to/negative_structures \
  --juchmme-dir path/to/juchmme \
  --out-dir eval_outputs \
  --metric-level file \
  --tag run-1.0.0
```

For chain metrics, provide both `--positive-target-manifest` and
`--negative-target-manifest` and choose `--metric-level chain` or `both`.
Each CSV must have exactly `relative_path,author_chain_id` columns and exactly one target
chain for every structure file. Chain metrics use only these selected chains;
partner chains remain unlabeled. Set the input labels and included structures
before evaluation; corrections require a new run.

Every invocation creates a fresh UTC timestamped directory below `--out-dir`
containing `chain_predictions.csv` (whose `is_target_author_chain` field marks
the manifest-selected author chain), `file_results.csv`, optional
`target_chain_results.csv`, `summary.csv`, `summary.md`, and
`evaluation_manifest.json`. The manifest records run status (`running`,
`failed`, or `complete`), settings, sampling rules, the Java executable,
JUCHMME installation files, and hashes of inputs and outputs. The evaluator
stops if predictions are missing, a result is `ERROR`, scores are non-finite,
target chains are unmatched, or a structure has no eligible predicted chain.
