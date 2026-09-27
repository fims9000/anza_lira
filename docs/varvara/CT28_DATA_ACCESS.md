# CT28 raw-data access for the current JUNCTION+CT task

Date: 2026-09-27  
Canonical branch: `research/coronary-connectivity-repair`

## Why this file exists

The frozen CT28 **PAIR** baseline can now be reproduced without raw CCTA because its feature table and predictions are committed in compact form.

The new **JUNCTION+CT** task is different: it creates new image-conditioned features, so the original CCTA volumes are required again.

Raw medical CT is intentionally not committed to Git.

## Exact cohort

Use the same frozen 28-patient cohort and do not change the split.

Train — 17:

`953, 955, 956, 959, 960, 963, 964, 967, 969, 970, 971, 975, 976, 977, 979, 982, 983`

Validation — 5:

`957, 961, 965, 966, 974`

Held-out test — 6:

`954, 958, 972, 973, 980, 984`

## Original CCTA source

Official ImageCAS Kaggle dataset:

`xiaoweixumedicalai/imagecas`

Dataset page:

`https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas`

The CT28 runner was deliberately restricted to cases whose ImageCAS entries are contained in the 801–1000 multipart archive. The historical execution used:

- `801-1000.z04`;
- final split member `801-1000.change2zip`, renamed locally to `801-1000.zip`.

There is no need to download the complete ~89 GB dataset for this cohort.

## Integrity checks — mandatory

Before extracting any new JUNCTION CT features, match every raw CT against:

`artifacts/varvara/ct28_pair/ct_alignment.csv`

and:

`artifacts/varvara/ct28_pair/expected_ct_geometry.csv`

The existing verified protocol checks:

- patient ID;
- NIfTI shape;
- voxel spacing;
- affine;
- raw CT SHA256;
- ImageCAS-X mask provenance.

All 28 previously used CTs passed these checks.

Coordinate conversion is frozen as:

`ImageCAS-X VTK LPS -> ImageCAS NIfTI RAS`

by negating x and y before applying the inverse CT affine.

Do not silently resample a mismatched patient into the experiment.

## Existing extractor

The frozen PAIR radial extractor is preserved at:

`scripts/research/ccta_graph_lira_safe_repair/extract_radial_features_z04.py.gz.b64`

Restore it with:

```bash
base64 -d scripts/research/ccta_graph_lira_safe_repair/extract_radial_features_z04.py.gz.b64 \
  | gzip -d > extract_radial_features_z04.py
```

It is a reference implementation for CT coordinate handling and radial sampling. For JUNCTION+CT, reuse the verified data/alignment layer but create a junction-specific candidate-aligned representation rather than forcing a pair corridor into a bifurcation problem.

## What not to use

Do not use the four old BDMAP/Hugging Face pilot cases as if they were ImageCAS-X IDs 953/956/957/960. Those row-derived identities were explicitly rejected during the earlier alignment audit.

Do not use branch/anatomical names as inference features. They may be used to construct or audit controlled ground truth only.

## Before running held-out test

Freeze on train/validation:

- JUNCTION candidate construction;
- CT representation;
- model family;
- feature list;
- threshold-selection rule.

Then apply unchanged to test patients 954, 958, 972, 973, 980, 984.
