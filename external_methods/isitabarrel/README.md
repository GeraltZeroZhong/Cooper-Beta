# IsItABarrel structure-map baseline

This adapter runs
[IsItABarrel](https://github.com/SluskyLab/isitabarrel) on contact maps derived
from PDB/mmCIF coordinates and reports the method as
`isitabarrel_structure_map`. The [published method](https://doi.org/10.1073/pnas.2220762120)
uses evolutionary contact maps.

Install the official IsItABarrel script separately under its AGPL-3.0 license.
The MIT-licensed Cooper-Beta adapter handles contact-map generation, program
execution, and result parsing.

## Runner inputs

The upstream script expects:

- a protein-ID list, one ID per line or as the first tab-separated column;
- a directory containing one contact-map pickle per ID, named `<id>.pkl`;
- an official `isitabarrel.py` checkout, provided with `--script` or the
  `ISITABARREL_SCRIPT` environment variable.

The upstream program writes `results.tsv` in its working directory. The adapter
parses that file and normalizes each row into:

- `baseline`: `isitabarrel_structure_map`
- `sample_id`: upstream `MAP_NAME`
- `result`: `BARREL` when the selected score is greater than zero, otherwise
  `NON_BARREL`
- `score`: selected decision score, defaulting to `CC2_TO_H4`, which the
  upstream comments recommend for reproducing the publication results

## Contact maps from structures

Generate contact maps from PDB or mmCIF structures and run IsItABarrel:

```bash
python external_methods/isitabarrel/structure_map.py \
  path/to/structures \
  --out-dir eval_outputs/isitabarrel_structure_map \
  --script path/to/isitabarrel.py \
  --out eval_outputs/isitabarrel_structure_map.csv
```

The generator writes:

- `maps/<sample_id>.pkl`: NumPy `float32` `L x L` contact maps
- `protid_list.tsv`: sample IDs for the upstream script
- `residue_mapping.csv`: matrix index to source residue mapping

The default map records C-alpha contacts at distances of 8.0 Å or less. Entries
within two positions of the diagonal are set to zero, using the order of
residues with observed C-alpha coordinates. The adapter processes chains with
at least 15 C-alpha residues to satisfy the upstream contact-filtering minimum.

## Run existing maps

```bash
python external_methods/isitabarrel/runner.py \
  path/to/protid_list.tsv \
  path/to/contact_maps \
  --script path/to/isitabarrel.py \
  --out eval_outputs/isitabarrel_structure_map.csv
```
