# Reproduce the CT28 PAIR baseline

Date: 2026-09-27

This note explains where the CT28 PAIR numbers come from and how to reproduce them without downloading raw multi-GB CCTA.

## What the experiment is

Controlled local binary relation task:

- positive = true 4 mm within-branch continuation;
- negative = nearby wrong branch matched by geometry;
- anatomical labels are used only to construct ground truth, not as model input features.

Cohort:

- 17 train patients;
- 5 validation patients;
- 6 held-out test patients;
- 1,360 rows total.

Frozen row counts:

- train: 758 = 379 positive + 379 negative;
- validation: 268 = 134 + 134;
- test: 334 = 167 + 167.

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

The CT28 runner compares:

1. `geometry` — geometry-only logistic baseline;
2. `radial_hu_summary_v1` — radial CT only;
3. `geometry_plus_radial_v1` — geometry + radial CT.

For all three:

`StandardScaler + LogisticRegression(C=1, class_weight="balanced")`

The operating threshold is selected on validation only:

**maximize recall subject to FPR <= 5%.**

Important: the separate strong `geometry_hgb` baseline is a different model. It is reproduced by `run_expanded_geometry_baselines.py.gz.b64`.

## Fast reproduction — no raw CCTA required

The full 1,360-row feature table is committed losslessly as a compressed text payload.

From the repository root:

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_features.py
```

This writes:

`artifacts/varvara/ct28_pair/expanded_relation_features.csv`

and verifies SHA256:

`819181674acb9a358dd822e55acf3f5721cf8d46b82c0e2655e7af842b0df864`

Then retrain/resave the three local PAIR models:

```bash
python scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_from_features.py \
  artifacts/varvara/ct28_pair/expanded_relation_features.csv \
  --out-dir ct28_pair_models
```

Expected outputs:

- `geometry.joblib`;
- `radial_hu_summary_v1.joblib`;
- `geometry_plus_radial_v1.joblib`;
- `summary.csv`;
- `predictions.csv`;
- `protocol.json`.

The regenerated table has already been checked: retraining reproduces the persisted score arrays to floating-point precision (~1e-13 maximum absolute difference or smaller) and the binary predictions match exactly.

## Persisted row-level artifacts

Already committed:

- `artifacts/varvara/ct28_pair/relation_pair_plan.csv`;
- `artifacts/varvara/ct28_pair/expanded_relation_predictions.csv`;
- `artifacts/varvara/ct28_pair/expanded_relation_summary.csv`;
- `artifacts/varvara/ct28_pair/protocol.json`;
- alignment / expected geometry / bootstrap summaries.

Therefore a collaborator does **not** need to regenerate raw CT features just to understand or reproduce the frozen PAIR baseline.

## Expected held-out result

For `geometry_plus_radial_v1`:

- AUROC 0.9846893040;
- recall 82.6347%;
- FPR 1.7964%;
- precision 97.8723%;
- TP / FP / FN / TN = 138 / 3 / 29 / 164.

If a reproduction differs materially, check:

1. patient split;
2. exact frozen pair plan;
3. feature column order;
4. validation-only threshold selection;
5. scikit-learn environment/version if serialized models are being compared.

## Full raw-CT regeneration

Only needed if changing the CT representation itself.

Canonical extractor:

```bash
base64 -d scripts/research/ccta_graph_lira_safe_repair/extract_radial_features_z04.py.gz.b64 \
  | gzip -d > extract_radial_features_z04.py
```

The original ImageCAS raw CT is intentionally not stored in Git. Exact cohort/alignment provenance is committed separately.

## Scientific boundary

This reproduces the **local binary PAIR relation** result.

It does not reproduce a full CT-conditioned Graph-LIRA result because the corresponding patient-general 28-case JUNCTION+CT evidence is the current open task.
