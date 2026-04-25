# External Methods

This directory contains adapters for external baseline methods used during
evaluation. Adapters live outside the `cooper_beta` package so external
licensing, data, and runtime requirements stay separate from the MIT-licensed
detector.

Each method should provide:

- a short README with the upstream source, license, expected inputs, and output
  interpretation;
- a small runner that invokes the external method or parses its output;
- smoke-test fixtures under `data/external_methods/` when the runner needs
  project-local test data.

Current adapters:

- `isitabarrel_structure_map`: structure-derived contact-map baseline.
- `pred_tmbb2_single_juchmme`: sequence-only topology baseline using an
  external JUCHMME/PRED-TMBB2 checkout.
