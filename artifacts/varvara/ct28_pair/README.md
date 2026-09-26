# CT28 PAIR collaboration artifacts

Purpose: make the current 28-patient PAIR baseline easy to reproduce without mixing it with the old six-patient graph-integration pilot.

## Committed here

- `ct_alignment.csv` — exact 28-patient raw-CT alignment / SHA provenance;
- `expected_ct_geometry.csv` — frozen expected ImageCAS-X geometry;
- `pair_plan_counts.csv` — frozen patient/split/class counts;
- `headline_test.csv` — compact held-out test result for the three CT28 local models;
- `paired_bootstrap_ci.csv` — paired patient-cluster uncertainty against the strong geometry reference.

## Generated files not yet persisted

The original real CT28 execution also produced:

- `expanded_relation_features.csv`;
- `expanded_relation_predictions.csv`;
- `expanded_relation_summary.csv`;
- `protocol.json`.

The full feature/prediction tables were generated in the execution workspace but were not persisted into Git before that workspace was recycled. They must **not** be reconstructed from aggregate metrics.

They are reproducible from the frozen extractor plus the raw ImageCAS archive.

## Retraining without raw CT

Once `expanded_relation_features.csv` has been regenerated, use:

`scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_from_features.py`

It trains and saves:

- `geometry.joblib`;
- `radial_hu_summary_v1.joblib`;
- `geometry_plus_radial_v1.joblib`;

plus summary/predictions/protocol.

This script exists specifically so a collaborator does not need to touch raw CCTA again after feature extraction.

## Why raw CT is absent

The raw ImageCAS archive is multi-gigabyte medical image data and is intentionally not versioned in Git.

Use:

`docs/varvara/REPRODUCE_CT28_PAIR_BASELINE.md`

and:

`docs/research/ccta_graph_lira_safe_repair/EXPANDED_CT_28CASE_EXECUTION_AND_ROADMAP.md`

for exact source/provenance instructions.

## Important model naming

- `geometry_hgb`: strong separate HGB geometry baseline;
- `geometry` inside the CT28 runner: StandardScaler + LogisticRegression geometry-only model;
- `radial_hu_summary_v1`: StandardScaler + LogisticRegression on radial CT summaries;
- `geometry_plus_radial_v1`: StandardScaler + LogisticRegression on geometry + radial CT summaries;
- canonical four-class Graph-LIRA relation head: a different scene-level HGB predicting NONE / PAIR / JUNCTION / BOTH.

Do not treat these objects as interchangeable.

## Current scientific task

Do not re-tune the PAIR baseline.

Proceed to:

1. 28-patient JUNCTION+CT evidence;
2. PAIR_CT + JUNCTION_CT scene-level relation head;
3. frozen Graph-LIRA evaluation;
4. only then radial-vs-CNN-vs-ANZA ablation.
