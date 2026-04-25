# PRED-TMBB2 Single-Sequence JUCHMME Baseline

This adapter treats PRED-TMBB2 single-sequence topology prediction as an
external Cooper-Beta evaluation baseline. Cooper-Beta labels this variant
`pred_tmbb2_single_juchmme` because it uses the local JUCHMME implementation
and derives a binary decision from the predicted topology.

The upstream JUCHMME package is GPL-3.0, while Cooper-Beta is MIT-licensed, so
the upstream Java code and trained parameter files are not vendored here. This
directory only contains invocation, FASTA generation, and result normalization.

## Upstream Sources

- PRED-TMBB2 web page: <https://hannibal.dib.uth.gr/PRED-TMBB2/>
- JUCHMME source and releases: <https://github.com/pbagos/juchmme>
- PRED-TMBB2 paper: Tsirigos KD, Elofsson A, Bagos PG. Bioinformatics. 2016.

## Expected Local Layout

Download and unpack a JUCHMME release, then provide the release root with
`--juchmme-dir` or the `PRED_TMBB2_JUCHMME_DIR` environment variable. The
adapter expects the standard release paths:

- `bin/`: compiled Java classes
- `models/tmbb2.mdel`
- `tables/A_TMBB2_TRAINED`
- `tables/E_TMBB2_TRAINED`
- `conf/conf.tmbb`

## Decision Rule

JUCHMME emits topology strings and scores, but the full PRED-TMBB2 web-server
discrimination pipeline also included optional signal-peptide and Pfam/OMPdb
features. This adapter therefore uses an explicit topology-derived decision:

- `result`: `BARREL` when the selected topology field has at least three
  predicted `M` runs, otherwise `NON_BARREL`
- `score`: number of predicted `M` runs
- default selected topology: `LP`

The normalized CSV also keeps `logodds`, reliability, algorithm score, sequence
length, and the raw topology string so downstream analyses can use a different
threshold if needed.

## Existing FASTA Example

```bash
python external_methods/pred_tmbb2/runner.py \
  path/to/sequences.fasta \
  --juchmme-dir /path/to/juchmme_git \
  --out eval_outputs/pred_tmbb2_single_juchmme.csv
```

## Structure-to-Sequence Workflow

When starting from PDB/CIF/mmCIF structures, generate one chain-level FASTA
record per analyzable chain and run the baseline:

```bash
python external_methods/pred_tmbb2/structure_sequence.py \
  path/to/structures \
  --out-dir eval_outputs/pred_tmbb2_single_juchmme \
  --juchmme-dir /path/to/juchmme_git \
  --out eval_outputs/pred_tmbb2_single_juchmme.csv
```

The generator writes:

- `sequences.fasta`: one FASTA record per chain
- `residue_mapping.csv`: sequence index to source residue mapping
- `juchmme_work/`: upstream working directory

For smoke tests, this repository uses a tiny fake JUCHMME runner under
`data/external_methods/pred_tmbb2_smoke/` so the adapter can be tested without
vendoring or downloading GPL code.
