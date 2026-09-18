# External Baseline Methods

These adapters run external methods and normalize their predictions for
comparison with Cooper-Beta. Run them from a source checkout; the upstream
programs and reference databases are installed or supplied separately.

| Method and usage guide | Input | Prediction |
| --- | --- | --- |
| [Foldseek](foldseek/README.md) | Protein structures and a reference panel of barrel chains | Global TMalign similarity with score and coverage thresholds |
| [IsItABarrel](isitabarrel/README.md) | Contact maps generated from protein coordinates | Structure-derived contact-map classification |
| [PRED-TMBB2 / JUCHMME](pred_tmbb2/README.md) | Complete protein sequences, supplied as FASTA or extracted from structure declarations | Classification from predicted membrane beta-strand segments |

Each guide lists the upstream source, license, dependencies, commands, decision
rule, and output files. The normalized method identifiers are
`foldseek_tmalign_structure_search`, `isitabarrel_structure_map`, and
`pred_tmbb2_single_juchmme`.

## Dataset Evaluation

The Foldseek and PRED-TMBB2 dataset evaluators use one observation per structure
file by default. A file is predicted positive when any of its eligible chains
is predicted positive. For chain metrics, supply positive and negative
target-chain manifests with exactly one target per file; partner chains do not
enter the chain-level metrics.

Foldseek dataset evaluation also requires an explicit reference panel and
homology-group assignments. Its guide explains how same-group, same-PDB, and
identical-chain references are excluded before selecting a hit. Evaluators save
predictions, metrics, and run metadata in a new output directory for each run.
