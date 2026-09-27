# Generated artifact status

Date: 2026-09-27

## Status: CLOSED

The previous generated-artifact gap is closed.

The frozen CT28 PAIR artifacts were regenerated from the original matched CCTA / frozen pair plan and verified against the expected result.

Persisted in Git:

- `relation_pair_plan.csv`;
- `expanded_relation_predictions.csv`;
- `expanded_relation_summary.csv`;
- `protocol.json`;
- `ct_alignment.csv`;
- `expected_ct_geometry.csv`;
- compact metrics / bootstrap files;
- lossless compressed feature payload:
  `payload/expanded_relation_features.csv.xz.b64`.

Restore the full `expanded_relation_features.csv` with:

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_features.py
```

Expected restored SHA256:

`819181674acb9a358dd822e55acf3f5721cf8d46b82c0e2655e7af842b0df864`

The retraining check reproduced all three CT28 local model score arrays to floating-point precision and identical thresholded predictions.

Expected combined held-out result:

- AUROC 0.9846893040;
- recall 82.6347%;
- FPR 1.7964%;
- precision 97.8723%;
- TP/FP/FN/TN = 138/3/29/164.

Raw CCTA remains intentionally outside Git.

No collaborator action is required to reconstruct the old PAIR feature table before starting the next scientific task.
