# Tver ANZA–SBPP–LIRA: real F3 controlled-gap audit, V27 vs V28
**14 November 2026; independent archive comparison. Research on CRACKS real image pixels with artificially erased predicted-segmentation gaps. Not natural fault-instance identification.**

## What was actually retrained
Historical ANZA and P0 LIRA checkpoint weights are not in Git. V27 trained original AZConv encoder (ANZA-v1, 14,194 parameters) and matching U-Net (14,481 parameters) from scratch on original CRACKS F3 image pixels: 120 steps, identical 64×64 patch inputs, same optimizer and the same incomplete expert supervision with WHITE pseudo-negative mining. Their checkpoint SHA256 files were found **identical across V27 and V28**:
- ANZA: \`14ecda83fbf327085390f1cc5665d30730e250fc4b2192f68ac1d25fa751e300\`.
- U-Net: \`908861df22265db189b4d18f53ca587634673b3c39a96c3f3752458e8516dc4b\`.
- The real-data relation head in V27/V28 is **LogisticRegression on 12 numeric candidate descriptors**, not the original native P0 five-convolution LIRA. V30 separately tests the real P0 architecture.

## Data availability and exploration limits
Original expert CRACKS v2: 40 F3 sections with an expert mask; 12 have at least one confident unbranched BLUE fault of >=80px. After V26 discovered initial selected TEST source region lacked positive expert masks, V27 selected TRAIN 10,13,17,22,24,35,41; CAL 43,67; BUFFER 106; TEST 108,124. This label-availability-based cohort is exploratory, not a new untouched external split. V28 reuses the already inspected **same 2 TEST sections** (2 positive expert-selected artificial gaps plus 2 candidate-removed synthetic negatives) and is explicitly **posthoc**.

## Isolating the V27 failure
With CAL pixel-F1 thresholds ANZA=.60 and U-Net=.50, ANZA predicted segmentations were too dense; on its two real-pixel TEST intervention crops, SBPP returned no source ports and no proposals. ANZA's 0 positive/48 negative predicted TRAIN relations therefore correctly caused \`STOP_INSUFFICIENT_REAL_PREDICTED_PAIRS\`, and its new pair head was never trained. U-Net gave 2/2 source ports and 1/2 correct proposed continuation, but its CAL-only logistic relation gate refused all proposals. Both frontends: 0/2 accepted correct relations and 0/2 paths recovered.

## V28: original weights unchanged; topology-oriented CAL threshold only
V28 independently enumerated fixed segmentation thresholds .35 to .90 on real TRAIN/CAL artificial gap scenes. Requirements: >=4 correct predicted TRAIN pairs, >=4 false predicted TRAIN pairs, >=1 CAL gap with a correct predicted candidate. Then maximize correct CAL gap coverage, fewer candidates and other predeclared tie breakers. Selected:
- ANZA-v1: threshold **.90**, not pixel-F1 .60; TRAIN 35 positive vs 485 negative predicted candidates; CAL 3 of 4 correct candidate gaps; real TEST 2/2 source, 2/2 proposals, 2/2 true proposal available.
- U-Net: threshold **.60**, not pixel-F1 .50; TRAIN 27 positive vs 530 negative predicted candidates; CAL 3 of 4 correct candidate gaps; real TEST 2/2 source, 2/2 proposals, 2/2 true proposal available.
- Both new logistic pair heads now actually trained on predicted TRAIN candidate features. All CAL-only link gates selected threshold **1.01 (never link)** for both P0-substitute logistic classifiers; the geometry baseline was also fully rejected at its CAL threshold .45.

| Heldout TEST stage | V27 ANZA .60 | V28 ANZA .90 | V27 U-Net .50 | V28 U-Net .60 |
|---|---:|---:|---:|---:|
| Predicted source port | 0/2 | **2/2** | 2/2 | 2/2 |
| >=1 correct predicted SBPP candidate | 0/2 | **2/2** | 1/2 | **2/2** |
| Correctly accepted relation | 0/2 | **0/2** | 0/2 | **0/2** |
| Correctly recovered path | 0/2 | **0/2** | 0/2 | **0/2** |

Crucial observation: the first bottleneck (dense ANZA segmentation) has a plausible posthoc engineering remedy through topology-aware threshold selection. The full pipeline remains **0/2** on both original F3 TEST sections, because ranking/refusal of alternatives remains unsolved. High raw LINK score (e.g., ANZA .953 and .970) cannot be deployed merely because it exceeds some TEST-chosen threshold: synthetic negative scores reach .916 and .817; CAL had only four positive and four synthetic negative scenes and conservatively froze never-link.

## Independent reproducibility controls
Original V27 official artifact [CI 43596275581](https://github.com/fims9000/anza_lira/actions/runs/43596275581) includes genuine real-trained ANZA/U-Net checkpoints and one fitted U-Net logistic scorer. Original V28 [CI 43596665171](https://github.com/fims9000/anza_lira/actions/runs/43596665171) succeeded, with two trained logistic scorers and unchanged original segmentation checkpoint SHA. Separate local archive-only audit \`Tver_V28_independent_archive_audit_2026-11-14.json\` independently checked **all 16 case-method rows**, every source/candidate/accepted/path denominator, exact old/new model checkpoint hashes, selected topology threshold and original 2 TEST section IDs. It did not re-train the segmenters or establish new independent TEST evidence.

## Next architecture experiment and truth boundary
V30 research draft [PR #34](https://github.com/fims9000/anza_lira/pull/34) executes the actual five-convolution \`EndpointPairClassifier\` P0 architecture **newly trained** on real observed original RGB crop, predicted source/candidate points and original SBPP mask, with two independent seeds. Controls include matched-feature LogisticRegression and original V28 full-data logistic plus geometry. Strict CAL-only refusal stays unchanged, TEST 108/124 already seen before V30 and therefore cannot confirm statistical superiority. Even a positive P0 ranking result would not imply naturally labeled fault identity or genuine 3D volumetric fault reconstruction.

**Publication-safe conclusion:** ANZA-v1 has not surpassed U-Net; topology thresholding makes its predicted segmentation usable by the proposed SBPP stage in these two artificially defined gaps. The original architecture pair identity decision and risk-selected full-pipeline recovery require further verification, and original historical weights remain absent.
