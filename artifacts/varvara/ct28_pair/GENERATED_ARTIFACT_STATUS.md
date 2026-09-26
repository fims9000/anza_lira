# Generated artifact status

Date: 2026-09-26

## Already committed in the collaboration pack

- `ct_alignment.csv`
- `expected_ct_geometry.csv`
- `pair_plan_counts.csv`
- reproduction documentation
- standalone PAIR model trainer

## Still generated, not yet committed

- `expanded_relation_features.csv`
- `expanded_relation_predictions.csv`
- `expanded_relation_summary.csv`
- `protocol.json`

These files must be generated from the frozen CT28 extractor; they must not be reconstructed from aggregate metrics.

The user's Google Drive still contains the ten uploaded parts of `801-1000.z04`, so no new data upload is required.

At the time of this checkpoint the ChatGPT execution container became unavailable after the Drive parts were materialized, preventing safe concatenation/re-execution in this turn. This is an execution-environment blocker, not a missing-data or research-protocol blocker.

Exact resume action:

1. materialize the ten existing `GraphLIRA_801-1000_z04.part_000..009` files;
2. concatenate them in numeric order to `801-1000.z04`;
3. verify SHA256 against the previously reconstructed archive:
   `e113e44e4984e383da13637ef6ce18511b8509c8c19f9b30cfe0c216ad10d3f2`;
4. run the frozen radial extractor;
5. verify the held-out combined result reproduces approximately:
   AUROC 0.9847, recall 82.63%, FPR 1.80%, precision 97.87%;
6. commit the four generated non-image artifacts listed above into this directory.

No user action should be required unless the existing Drive parts become unavailable.
