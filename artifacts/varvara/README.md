# Collaboration artifact pack for CT28 / Graph-LIRA

This directory is a compact handoff for the current coronary-connectivity work.

Included small derived artifacts:
- ct28_alignment.csv
- ct28_pair_plan_counts.csv
- ct28_expected_ct_geometry.csv
- ct28_headline_test.csv
- ct28_paired_bootstrap_ci.csv

Canonical per-patient table:
results/ccta_graph_lira_safe_repair/2026-09-21/real_ct28_per_patient.csv

Raw ImageCAS CCTA is intentionally not stored in Git.

The full generated expanded_relation_features.csv and expanded_relation_predictions.csv were produced during the real CT28 execution but were not persisted into Git before the execution workspace was recycled. Do not reconstruct them from aggregate metrics. They must be regenerated from the frozen runner and exact raw CT.

Use:
- docs/varvara/ARTIFACT_MAP_AND_CURRENT_TASK_2026-09-26.md
- docs/varvara/CURRENT_TASK.md
- docs/varvara/NEGATIVE_RESULTS_THAT_MATTER.md
- docs/varvara/REPRODUCE_CT28_PAIR_BASELINE.md

Current task:
28-patient JUNCTION+CT -> CT-conditioned relation head -> frozen Graph-LIRA.
