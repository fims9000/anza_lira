# CT28 PAIR collaboration artifacts

Purpose: make the frozen 28-patient PAIR baseline immediately reproducible without mixing it with the old six-patient graph-integration pilot.

## Current status: complete

The generated CT28 non-image artifacts have now been regenerated from the frozen pair plan and original matched CCTA and persisted for collaboration.

Committed directly in this directory:

- `ct_alignment.csv` — exact 28-patient raw-CT alignment / SHA provenance;
- `expected_ct_geometry.csv` — frozen expected ImageCAS-X geometry;
- `pair_plan_counts.csv` — frozen patient/split/class counts;
- `headline_test.csv` — compact held-out test result;
- `paired_bootstrap_ci.csv` — paired patient-cluster uncertainty against the strong geometry reference;
- `relation_pair_plan.csv` — exact 1,360-row frozen pair plan;
- `expanded_relation_predictions.csv` — row-level scores/predictions;
- `expanded_relation_summary.csv` — train/val/test and patient-level model summary;
- `protocol.json` — frozen extraction/evaluation protocol.

The full feature table is stored losslessly as:

`payload/expanded_relation_features.csv.xz.b64`

Restore it with:

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_features.py
```

The restore helper verifies SHA256:

`819181674acb9a358dd822e55acf3f5721cf8d46b82c0e2655e7af842b0df864`

## Retrain the PAIR models without raw CT

After restoring the feature table:

```bash
python scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_from_features.py \
  artifacts/varvara/ct28_pair/expanded_relation_features.csv \
  --out-dir ct28_pair_models
```

This trains and saves:

- `geometry.joblib`;
- `radial_hu_summary_v1.joblib`;
- `geometry_plus_radial_v1.joblib`;

plus `summary.csv`, `predictions.csv` and `protocol.json`.

The regenerated feature table was independently retrained in the handoff check. The three score arrays reproduce the persisted predictions to numerical floating-point precision (maximum absolute differences ~1e-13 or smaller) and all thresholded predictions are identical.

## Expected held-out PAIR result

For `geometry_plus_radial_v1`:

- AUROC: 0.9846893040;
- recall: 0.8263473054;
- FPR: 0.0179640719;
- precision: 0.9787234043;
- TP / FP / FN / TN: 138 / 3 / 29 / 164.

## Important model naming

- `geometry_hgb`: separate strong HGB geometry-only PAIR baseline;
- `geometry` inside the CT28 runner: StandardScaler + LogisticRegression geometry-only model;
- `radial_hu_summary_v1`: StandardScaler + LogisticRegression on radial CT summaries;
- `geometry_plus_radial_v1`: StandardScaler + LogisticRegression on geometry + radial CT summaries;
- canonical four-class Graph-LIRA relation head: a different scene-level HGB predicting NONE / PAIR / JUNCTION / BOTH.

Do not treat these objects as interchangeable.

The strong `geometry_hgb` remains reproducible from:

`scripts/research/ccta_graph_lira_safe_repair/run_expanded_geometry_baselines.py.gz.b64`

using the committed `relation_pair_plan.csv`.

## Why raw CT is absent

The raw ImageCAS archive is multi-gigabyte medical image data and is intentionally not versioned in Git. It is only needed to regenerate image features from scratch; it is **not** needed to reproduce/retrain the already-frozen CT28 PAIR baseline from the committed feature payload.

## Current scientific task

Do not re-tune the PAIR baseline.

Proceed to:

1. 28-patient JUNCTION+CT evidence;
2. PAIR_CT + JUNCTION_CT scene-level relation head;
3. frozen Graph-LIRA evaluation;
4. only then radial-vs-CNN-vs-ANZA ablation.
