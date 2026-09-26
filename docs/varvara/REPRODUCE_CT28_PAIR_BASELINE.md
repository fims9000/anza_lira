# Reproduce the CT28 PAIR baseline

Date: 2026-09-26

This note explains where the published CT28 pair numbers come from and how to reproduce them.

## What the experiment is

Controlled local binary relation task:

- positive = true 4 mm within-branch continuation;
- negative = nearby wrong branch matched by geometry;
- anatomical labels are used only to construct ground truth, not as input features.

Cohort:

- 17 train patients;
- 5 validation patients;
- 6 held-out test patients;
- 1,360 rows total.

## Frozen image representation

For each candidate corridor:

- 17 positions along the candidate;
- center HU profile;
- orthogonal local cross-section;
- 8 angular samples;
- radii 1, 2 and 3 mm;
- HU clipping [-300, 1200];
- radial/background contrast summaries.

## Frozen local models

The CT28 runner compares three lightweight models:

1. geometry only;
2. radial CT only;
3. geometry + radial CT.

For all three:

`StandardScaler + LogisticRegression(C=1, class_weight="balanced")`

The operating threshold is selected on validation only:

**maximize recall subject to FPR <= 5%.**

## Canonical extractor

Restore:

```bash
base64 -d scripts/research/ccta_graph_lira_safe_repair/extract_radial_features_z04.py.gz.b64 \
  | gzip -d > extract_radial_features_z04.py
```

The frozen cohort metadata are in:

`experiments/ccta_graph_lira_safe_repair/expanded_ct_28case/`

The original ImageCAS raw CT is intentionally not stored in Git.

## Expected outputs

The extractor produces:

- `ct_alignment.csv`;
- `expanded_relation_features.csv`;
- `expanded_relation_summary.csv`;
- `expanded_relation_predictions.csv`;
- `protocol.json`.

The full feature table contains no raw CCTA volume, so it is suitable as a collaboration artifact.

## Expected held-out result

For the combined model:

- AUROC 0.9847;
- recall 82.63%;
- FPR 1.80%;
- precision 97.87%.

If a reproduction materially differs, first check:

1. patient split;
2. exact frozen pair plan;
3. LPS -> RAS x/y sign conversion;
4. CT affine alignment;
5. HU clipping;
6. validation-only threshold selection.

## Current repository status

Compact metrics, patient-level outputs, alignment and provenance are committed.

The complete generated `expanded_relation_features.csv` and row-level `expanded_relation_predictions.csv` are generated outputs rather than source data. They should be placed in the collaboration artifact directory when regenerated; they must not be reconstructed from test metrics.

Do not substitute fabricated rows or re-create them from aggregate statistics.
