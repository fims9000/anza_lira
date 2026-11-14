# Tver V30: real CRACKS predicted ANZA→SBPP→new original P0 architecture, **executed negative result**

**14 November 2026. Independent Github Actions [run 43598608411](https://github.com/fims9000/anza_lira/actions/runs/43598608411) SUCCESS.** Draft [PR #34](https://github.com/fims9000/anza_lira/pull/34). Audited artifact ID **14114584485**, SHA256 ZIP **cd461ded45fae273a649c208cbed30b87c274a0797ea92c27e9fe7d0a5654fe7**. Runner \`scripts/verify_tver_v30_real_p0.py\` returned \`V30_INDEPENDENT_NATIVE_P0_REAL_PREDICTED_CASE_AUDIT_PASS\`: all 40 posthoc TEST case/method rows and their exact candidate-score-derived CAL decisions, four original-repository P0 model \`.pt\` SHA-256 and two distinct training initializations were recomputed.

## Why V30 is architecturally necessary

V27/V28's \`new_lira\` had been a **12-feature LogisticRegression**, not the historical 5-convolution P0 contextual CNN architecture. V30 newly trains \`path_completion.pair_classifier.EndpointPairClassifier\` itself: five convolutions, original 6×33×49 oriented context of **real observed CRACKS RGB pixels**, visible model-predicted segmentation and two coordinate/corridor maps from predicted SBPP source + candidate endpoints. No hidden expert GT ends/centerlines/labels enter inference. **Historical pretrained ANZA or P0 weights remain UNAVAILABLE**: this study evaluates a *newly learned instantiation of the original architecture*, not replaying the old reported P0 outcomes.

The study's image evidence is real CRACKS F3, but all gaps are created by expert-selected artificial missing-segmentation rectangles and all negative worlds by suppressing the synthetic destination-side candidate. They do not certify natural geological fault-instance identity or natural missing-gap reconstruction. V30 TEST F3 sections **108 and 124 were already inspected in V27 and V28**: this is explicitly *posthoc*.

## Strict real model, source and provenance

- Physical F3 source eligibility was selected by BLUE expert-positive trace availability after V26 revealed empty candidate TEST. TRAIN: 10,13,17,22,24,35,41 (7 sections). CAL: 43,67 (2 sections, four positive controlled gaps + four synthetic candidate-absent NO_LINK scenes). BUFFER: 106. Reused previously inspected TEST: 108,124 (two positive gaps and two synthetic negatives).
- **Exact immutable real-trained V28 front-end weights restored**, no approximate fresh re-training: U-Net 14,481 params checkpoint SHA \`908861df22265db189b4d18f53ca587634673b3c39a96c3f3752458e8516dc4b\`; ANZA-v1 14,194 params SHA \`14ecda83fbf327085390f1cc5665d30730e250fc4b2192f68ac1d25fa751e300\`. Registered original V28 TRAIN/CAL topology thresholds U-Net **.60**, ANZA **.90**. Both provide true predicted SBPP source and at least one correct proposed continuation in **2/2** TEST gaps, after actual hard+soft clustering.
- **Pre-model technical safeguard:** initial V30 run \`43598051251\` STOPPED before P0 training because one-thread re-training of AZConv/U-Net resulted in different checkpoint hashes and CAL topology thresholds (.85 ANZA/.75 U-Net), violating the intended P0-only experiment. Fix committed **before any V30 P0/test outcome**: download exact successful V28 Actions model artifact and verify original SHA, copy into V30. Not a new method/result threshold.
- Trained actual P0 native 5-convolution CNN separately for each frontend with genuine seeds 11,23, 30 fixed TRAIN-only BCE epochs, AdamW and at most one geometrically close predicted TRAIN negative per predicted TRAIN positive: U-Net **27 positive +27 negative**, ANZA **35+35**, two architectures × two seeds = **four new actual \`.pt\` weights** (134,817 parameters each); separate SHA/distinct seeds verified.
- Compare geometry-only, original full-TRAIN 12-feature LogisticRegression, **matched 1:1 TRAIN** 12-feature LogisticRegression, two P0 CNNs for each segmentation frontend. P0 matched and matched-Logistic share precisely same labeled examples; full-data logistic uses more negative TRAIN pairs and is marked as a different information budget.
- CAL-select per-policy threshold from {0,.05,...,1.0,1.01}, maximizing correctly accepted CAL positives subject to ZERO wrong accept among 4 positive CAL and ZERO false accept among 4 synthetic NO_LINK CAL. If no feasible correct CAL link, freeze never-link threshold \`1.01\`. TEST is opened within this V30 run only after every fit/threshold is frozen; source data had been examined in previous V27/V28 research.

## Actual heldout (already seen earlier) TEST performance

| Predicted front-end / relation head | Source ports | Correct candidate offered | Ungated correct top-1 /2 | CAL-selected acceptance threshold | Correct accepted /2 | Correct widest-path /2 |
|---|---:|---:|---:|---:|---:|---:|
| ANZA-v1 / geometry | 2/2 | 2/2 | 0/2 | 1.01 | 0/2 | 0/2 |
| ANZA-v1 / full 12-feature logistic | 2/2 | 2/2 | **2/2** | 1.01 | 0/2 | 0/2 |
| ANZA-v1 / matched 12-feature logistic | 2/2 | 2/2 | **2/2** | 1.01 | 0/2 | 0/2 |
| ANZA-v1 / genuine P0 seed 11 | 2/2 | 2/2 | 1/2 | 1.01 | 0/2 | 0/2 |
| ANZA-v1 / genuine P0 seed 23 | 2/2 | 2/2 | 1/2 | 1.01 | 0/2 | 0/2 |
| U-Net / geometry | 2/2 | 2/2 | 0/2 | 1.01 | 0/2 | 0/2 |
| U-Net / full 12-feature logistic | 2/2 | 2/2 | **2/2** | 1.01 | 0/2 | 0/2 |
| U-Net / matched 12-feature logistic | 2/2 | 2/2 | **2/2** | 1.01 | 0/2 | 0/2 |
| U-Net / genuine P0 seed 11 | 2/2 | 2/2 | 0/2 | 1.01 | 0/2 | 0/2 |
| U-Net / genuine P0 seed 23 | 2/2 | 2/2 | 0/2 | 1.01 | 0/2 | 0/2 |

**No accepted wrong positive pairs, no false accepted synthetic negative**, because **all decisions refuse**, not because a detector has proven low geological false-bridge risk. This is a useful engineering safety stop but a negative recovery outcome. The original P0 classifier architecture **does NOT outperform the equal-positive-example-size logistic matched-feature baseline**, even on descriptive ungated top-1 in these two already inspected sections. Two TEST scenes cannot establish significance; 134k-parameter CNNs trained on 54 or 70 pair crops for 30 epochs are very small-data experiments. They do not represent a state-of-the-art trained fault segmentation backbone, and their poor result cannot be generalized to deep CNNs.

For source traceability, U-Net P0 TRAIN BCE epoch1→30 was 0.686→0.017 (seed11), 0.737→0.035 (seed23); ANZA P0 BCE 0.763→0.308 (seed11), 0.759→0.254 (seed23). Falling TRAIN BCE is **not generalization evidence**. Without enough independent CAL/TEST section labels, overfitting is plausible, not individually proven.

## Publication-safe consequence

The original claim in the 12-page November Tver manuscript must **NOT** be silently revised to imply full-pipeline real-image success. Prior controlled synthetic candidate recall (2611/2688), contextual AUROC (.9923), and 1/128 negative accept refer to separate historical experiments; weights not recovered. New V27–V30 are genuinely executed research on real F3 source pixels, but the image-to-predicted-port-to-context-decision-to-path stages have **0/2** accepted/recovered positive gaps across both frontends. The strong and reproducible partial observation is that CAL-selected topology threshold can make correct predicted SBPP candidate available in 2/2 heldout *controlled synthetic interventions* for both frontends; no independent clinical/geological/SOTA inference. Original P0 architectural reproduction is technically now verified but not performance-validated. New external geological volumes with line-instance labels and independent calibration sections are required.

**Final research decision:** V30 success means *the actual P0 CNN is trained, receives observed 6-channel context, and produces inspectable predictions*, not that LIRA is superior to logistic, or that it passes calibrated deployment. Retain the full negative evidence and keep PR Draft. Model weights and anonymized case-level synthetic-gap labels, metrics, and state checksums are in successful CI archive.
