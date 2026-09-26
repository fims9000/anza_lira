# CT28 PAIR collaboration artifacts

Purpose: make the current 28-patient PAIR baseline easy to reproduce without mixing it with the old six-patient graph-integration pilot.

## Committed here

- `ct_alignment.csv` — exact 28-patient raw-CT alignment / SHA provenance;
- `expected_ct_geometry.csv` — frozen expected ImageCAS-X geometry;
- `pair_plan_counts.csv` — frozen patient/split/class counts.

## Generated files expected here

After running the frozen extractor:

- `expanded_relation_features.csv`;
- `expanded_relation_predictions.csv`;
- `expanded_relation_summary.csv`;
- `protocol.json`.

These generated files contain no raw CCTA image volume.

## Why raw CT is absent

The raw ImageCAS archive is multi-gigabyte medical image data and is intentionally not versioned in Git.

Use the source/provenance instructions in:

`docs/research/ccta_graph_lira_safe_repair/EXPANDED_CT_28CASE_EXECUTION_AND_ROADMAP.md`

## Important model naming

- `geometry_hgb`: strong separate HGB geometry baseline.
- `geometry` inside the CT28 runner: StandardScaler + LogisticRegression geometry-only model.
- `radial_hu_summary_v1`: StandardScaler + LogisticRegression on radial CT summaries.
- `geometry_plus_radial_v1`: StandardScaler + LogisticRegression on geometry + radial CT summaries.
- canonical four-class Graph-LIRA relation head: a different scene-level HGB predicting NONE/PAIR/JUNCTION/BOTH.

Do not treat these objects as interchangeable.
