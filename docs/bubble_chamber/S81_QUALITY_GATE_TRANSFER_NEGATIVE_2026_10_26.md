# S81 — image-quality-gated two-scale candidates: CAL rejected / new held-out study

**26 October 2026 · study ID `ANZA-LIRA-BC-S81-20261026`.** Preregistered in `S81_FROZEN_PIXEL_QUALITY_GATED_MULTISCALE_2026_10_26.md` before any S81 CAL or TEST. Historical S80 test motivated the question but was **not** used for gate thresholds, seeds, or CAL selection.

## Objective and frozen conditions

S80 two-scale candidate detection increased synthetic recall under blur but harmed clean-scene false-join counts. S81 adds an **image-only blur/texture quality gate**, switching the optional scale-specific candidate detector on only when the per-image fine/coarse spatial-frequency ratio is sufficiently low. Two variants use source-photo CLEAN CAL feature 20th and 40th percentiles as numerical thresholds; S56 geometry scorer, source-CAL threshold, original S72 per-direction base, S59 pair gate and S69 corrected orientation-invariant synthetic evaluator are unchanged. The gate never sees synthetic labels, true fragment IDs or candidate link correctness.

S81 CAL uses 16 fresh scenes per source direction, seeds `20268101/20268111`, conditions clean and artificial Gaussian blur σ=1.1. S81 TEST uses **different 96 artificial scenes**, seeds `20268103/04/13/14`, the opposite genuine H50Z background, and clean / σ=1.1 / σ=1.6 conditions. Both H50Z photo sets are views of **one physical run**; visualMetro film 070 source geometries belong to a different event. There is no physical true-track reference.

## CAL decision, frozen before TEST

| Source direction | Frozen image-only quality threshold q20 | q40 | CAL selected method |
|---|---:|---:|---|
| 005_to_006 | 0.508615 | 0.513986 | baseline |
| 006_to_005 | 0.593243 | 0.661858 | baseline |

**Neither direction met the preregistered safety gate.** In source 005 CAL, the gated blurred correct-join gain was +1 (minimum +2); in source 006 CAL, the gated methods gained +1 true blur join but increased noncorrect blurred joins by +4. The original S72 remained the selected method in both directions before TEST. Thresholds and CAL selection were not revised after TEST.

## New TEST results (96 artificial scenes per input condition)

All true/wrong/error terms are **relative to synthetically injected lines**, NOT independently confirmed physical particles. "Other joins" includes incorrect and non-verifiable artificial-proxy joins.

| Condition | Method | Synthetic-correct joins | Other selected joins | Whole synthetic 3-piece paths | Matched synthetic fragments by shape | Quality gate activations /96 |
|---|---|---:|---:|---:|---:|---:|
| No blur | Frozen S72 | 48 | 29 | 15 | 246 | 0 |
| No blur | S81 gated q20 | 51 | 30 | 16 | 252 | 39 |
| No blur | S81 gated q40 | 51 | 31 | 16 | 255 | 47 |
| Blur σ=1.1 | Frozen S72 | 19 | 22 | 6 | 138 | 0 |
| Blur σ=1.1 | S81 gated q20 | 23 | 22 | 7 | 156 | 96 |
| Blur σ=1.1 | S81 gated q40 | 23 | 22 | 7 | 156 | 96 |
| Blur σ=1.6 | Frozen S72 | 13 | 17 | 3 | 117 | 0 |
| Blur σ=1.6 | S81 gated q20 | 14 | 22 | 3 | 127 | 96 |
| Blur σ=1.6 | S81 gated q40 | 14 | 22 | 3 | 127 | 96 |

## Interpretation and deployment rule

At σ=1.1, both gated arms add four synthetic-correct joins (19→23), do not increase noncorrect joins (22→22), and recover one additional full artificial path (6→7). **But** these experimental variants automatically turn on in **all 96 blurred TEST images**, and in 39/96 or 47/96 **clean** TEST images, producing additional noncorrect clean proposals (29→30 or 31). At the stronger σ=1.6 blur, both produce five additional noncorrect joins (17→22) for only +1 correct (13→14). Thus the global texture ratio is insufficiently specific for safe use; this finite result is not evidence of a robust calibrated blur classifier.

**No promotion.** Both source-CAL directions chose baseline. No post-TEST switching, no threshold tweaking from the S81 held-out scenes. Any next quality-gating experiment must predeclare new seeds and source-CAL-only selection. All real-photo H50Z track identities remain UNKNOWN.

## Genuine native 39-tile read-only machine audit

One predetermined 192×192 center crop from each original 512×512 H50Z tile; no artificial tracks and no reference physical event labels.

| Method | Native candidate lines (UNKNOWN) | Graph advisory joins (UNKNOWN) | Extra scale-specific candidates | Quality gate activations /39 |
|---|---:|---:|---:|---:|
| Frozen S72 | 83 | 3 | 0 | 0 |
| S81 q20 | 93 | 4 | 10 | 25 |
| S81 q40 | 93 | 4 | 10 | 26 |

These are **not** true track detections; no native sensitivity, precision or mass/trajectory is known. Two H50Z source photos come from the same physical run, so domain generalization cannot be validated from them alone.

## Reproducibility and integrity

- Frozen CAL and TEST JSON public outputs have SHA-256 hashes stored in `EXPERIMENT_IDS_MACHINE_ONLY.json`.
- New TEST re-run is byte-identical, digest `d13d96d32b3ed3dcfc2d364ba9fb2fae54ada19d1d4308a8f6352effa9553854`.
- Frozen source scorer, S72 direction branch, S80 line predictor and S69 evaluator are not altered. S81 adds only a pixel-statistic gate and its source-CAL selection.
- Scene-level paired 1200-bootstrap uncertainty per direction/seed is recorded in the TEST JSON; do not treat individual pixels or photographed crops as independent physical runs.
- No original photos, measured source coordinates, source-point IDs or model weights in public source.

```bash
python -m pytest -q tests/test_s81_*.py
python scripts/research/bubble_chamber/s81_calibrate_quality_gate.py cal \
 --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/tiles \
 --out /PRIVATE/S81_FROZEN_CAL_PRIVATE.json
python scripts/research/bubble_chamber/s81_calibrate_quality_gate.py test \
 --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/tiles \
 --calibration /PRIVATE/S81_FROZEN_CAL_PRIVATE.json \
 --out /PRIVATE/S81_FRESH_TEST_PRIVATE.json
python scripts/research/bubble_chamber/s81_native_quality_gate_readonly.py \
 --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/tiles \
 --calibration /PRIVATE/S81_FROZEN_CAL_PRIVATE.json \
 --out /PRIVATE/S81_NATIVE_UNKNOWN_PRIVATE.json
```
