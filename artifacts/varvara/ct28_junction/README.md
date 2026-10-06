# CT28 JUNCTION development artifact

Status: train/validation local-evidence checkpoint, 2026-10-06.

This directory preserves the current JUNCTION baseline state used before the
CT-conditioned Graph-LIRA stage.

## Direct files

- `baseline_metrics.json`
- `junction_baseline_predictions.csv`
- `restore_junction_features.py`

The full train/validation feature table is stored losslessly as split text
payloads under `payload/` to avoid one oversized GitHub contents upload.

Restore it with:

```bash
python artifacts/varvara/ct28_junction/restore_junction_features.py
```

Expected output:

`junction_relation_features_train_val.csv`

Expected SHA256:

`c58f0aef645ea0cf4552b846cf4140c052908f6457efeb4bc6cdc3efa7532483`

Other current file hashes from the supplied run:

- baseline predictions:
  `3ad4cb9ee946476247c1c839a84eae0fa277bbc7d94b96aa4dd9bce59ee0c3c7`
- baseline metrics:
  `8f97e61bb40643acc3d68163c15dec6462b77bcd5a00dfca49b1024d94a74186`

## Current split

- train: 17 patients, 107 source junctions, 1,273 candidates;
- validation: 5 patients, 36 source junctions, 536 candidates;
- total: 143 positives + 1,666 negatives;
- degree-3 only in the baseline;
- held-out JUNCTION test was not accessed.

## Reproducibility gap still to close

The exact candidate-plan / endpoint / selection-audit CSVs and the exact local
config used for this run were not supplied in the handoff files available to
the repository maintainer.

Varvara should copy the exact files from the run that produced the feature
SHA above before regenerating anything:

- `junction_relation_plan_train_val.csv`
- `junction_candidate_endpoints_train_val.csv`
- `junction_selection_audit_train_val.csv`
- `junction_ct_alignment_train_val.csv`
- exact generator / CT / baseline configs

Do not replace these with a newly regenerated version merely because row
counts match. Preserve provenance first.


## Regeneration audit

The supplied current `build_junction_plan.py` was independently rerun against
the supplied source pack with the stated development settings:

- target cut = 2.0 mm;
- max arm fraction = 0.9;
- max span = 16.0 mm;
- decoy step = 1.5 mm;
- endpoint margin = 2.0 mm;
- max one-arm negatives = 12;
- max two-arm negatives = 6.

It reproduced:

- 1,809 plan rows;
- 143 positives;
- 1,666 negatives;
- the same candidate-ID set and order as the committed feature table;
- the same source-junction/group labels;
- geometry values matching the feature table to numerical precision.

However, the regenerated candidate-plan file SHA256 was:

`29892466991c0af8a2acb7df6f18c1f977f433ed7d6b24c5a533d245a8874b26`

while the baseline metrics record the exact training-run candidate-plan hash as:

`2266609c3b14b3510b9707c7fb0282cbc93682cbde577c2951578d4466a002d4`

Therefore the regenerated plan is **not promoted as the exact canonical input
file**. The original plan/config from Varvara's successful run should be copied
into this artifact pack first. This avoids silently replacing provenance with a
numerically equivalent regeneration.
