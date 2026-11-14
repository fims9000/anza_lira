# Tver V27 — real predicted ANZA–SBPP–LIRA, executed negative result (2026-11-14)

**CI SUCCESS:** https://github.com/fims9000/anza_lira/actions/runs/43596275581 ; artifact ID **14114995869**, ZIP sha256 **3df05feb2ca5e050a682b843994a11a3c54c77af4030665561be636e668dbb31**. Separate independent audit V27_INDEPENDENT_REAL_PREDICTED_E2E_COUNT_AUDIT_PASS. Historical checkpoints **not recovered**; both ANZA-v1 and U-Net newly trained on genuine CRACKS v2 F3 images.

## Cohort and leak boundaries
12 of 40 expert sections have a sufficiently long unbranched BLUE trace (>=80px). Availability-conditioned split, chosen after V26 revealed no BLUE in heldout bands: TRAIN sections 10,13,17,22,24,35,41; CAL 43,67; BUFFER 106; TEST 108 and 124. Only **two** controlled missing-segmentation gap interventions in TEST, plus two artificial candidate-absent NO_LINK worlds. Expert schedule selects rectangle and posthoc truth only; source port and candidate come from predicted segmentation and original seismic pixels, never supplied expert endpoints. This is exploratory selection, not confirmation.

## Trained REAL segmentation and same-model pipeline
Models from actual repository models.segmentation_v2: genuine AZConv ANZA-v1 14,194 parameters CAL threshold 0.6, train last30 loss 0.61808; matched U-Net 14,481 parameters CAL threshold 0.5, last30 0.55175. Both 120 patches/steps, 64x64, same samples/optimizer and admitted expert BLUE/ORANGE; distant WHITE is **assumed pseudo-negative for TRAIN only**, never verified geology. Case evaluator executes actual SBPP hard/soft extraction and cluster/port proposals using predicted probability, newly fitted logistic LIRA on predicted TRAIN candidates only, CAL-only refusal, repository maximin path plus straight baseline.

| Heldout stage, two positive gaps | ANZA AZConv | U-Net |
|---|---:|---:|
| Predicted source available | 0/2 | 2/2 |
| >=1 SBPP predicted candidate | 0/2 | 2/2 |
| >=1 correct proposed candidate | 0/2 | 1/2 |
| Accepted correct candidate by geometry CAL | 0/2 | 0/2 |
| Accepted correct by learned LIRA CAL | 0/2 | 0/2 |
| Correct maximin recovered path | 0/2 | 0/2 |
| False accepted synthetic NO_LINK | 0/2 | 0/2 |

**ANZA failure: over-dense probability mask**: both heldout interventions returned DEGENERATE_DENSE_PREDICTED_MASK before SBPP extraction. ANZA TRAIN predicted LIRA pairs had 0 positive and 48 negative, so **no ANZA LIRA pair head was trained**: explicit STOP_INSUFFICIENT_REAL_PREDICTED_PAIRS. Its risk is fail-closed and zero correctly completed paths, not successful E2E. U-Net TRAIN supplied 21 true positive predicted pairs and 546 false; its LIRA was actually fitted but CAL locked **1.01 never-link** (0 accepted correct among only 4 positive and 4 synthetic-negative CAL worlds). U-Net had 10/14 hard branches and 45–48 proposals/case in heldout image, but accepted no valid link. No model bridges were hallucinated.

CAL pixel diagnostics use uncertain pseudo negatives and are NOT physical precision: ANZA threshold .6 positive recall .828, pseudo-precision .413 (F1 .551); U-Net threshold .5 positive recall .672, pseudo-precision .545 (F1 .602). ANZA threshold .5 predicts nearly every masked CAL pixel positive, showing diffuse mask, not a working sparse fault skeleton. **No superiority over U-Net, no natural fault instance identity risk, no published-segmentation model comparison.** Future re-training must be explicitly posthoc because 108/124 TEST have been opened.

## Source artifacts
- Draft PR #32: https://github.com/fims9000/anza_lira/pull/32
- CI: https://github.com/fims9000/anza_lira/actions/runs/43596275581
- Actual newly trained .pt ANZA/U-Net, fitted U-Net LIRA .joblib, frozen source/CAL SHA receipts, four test cases per method and independent per-case auditor in Actions artifact 14114995869.

**Science:** this is a real-pixel controlled missing-segmentation TEST of genuinely predicted candidates, not historical ANZA→SBPP→LIRA with original unavailable weights, and not natural-gap E2E in an unseen 3D field. Even the best front-end did not recover any of two heldout gaps.