# Tver V25: fair hard-negative trained HGB and actual image CNNs — executed negative result
**Executed 14 November 2026. Entirely controlled synthetic oracle-endpoint 2.5D linkage task; NOT historical ANZA–LIRA E2E, NOT real seismic fault-instance recognition.**

- Frozen-before-outcome protocol: \`docs/research/tver2026/V25_EQUAL_HARD_NEGATIVE_TRAINING_FROZEN_PROTOCOL_20261114.md\` (commit \`8271c9023f8c8459f0567d33d1b3b3d12d52cac4\`).
- Research [draft PR #31](https://github.com/fims9000/anza_lira/pull/31).
- Independent GitHub Actions **SUCCESS**: [run 43593052911](https://github.com/fims9000/anza_lira/actions/runs/43593052911); full aggregate-and-model artifact \`tver-v25-fair-hard-negative-trained-hgb-cnn\` ID \`14112834610\` SHA256 \`2f9bfc2804d19ff940e0d4d1384da830bd2b2cbecf6b1fbe880ab62be338385b\`.
- Independent \`scripts/verify_tver_v25_case_evidence.py\` **PASS**; separately recomputed 480 TEST decisions, frozen CAL thresholds, all method confusion metrics, exact counterfactual central and one-neighbor negative SHA constraints, nuisance-base cluster CIs. The four actual trained CNN \`.pt\` weights are archived along with complete case-level synthetic evidence.

## Scientific reason for V25
V24 appeared to beat 2.5D HistGradientBoosting on a new **switching-neighbor hard NO_LINK** regime (+13.54pp three-way accuracy). But the HGB baseline had been fitted to the *simpler V23 negatives*. This different training distribution undermined an architecture-superiority inference. V25 answers the stronger question: **what if trained methods see precisely the same hard negative construction during TRAIN?**

## Frozen method and dataset
One nuisance base defines three paired synthetic worlds: A-link, B-link, hard NO_LINK. The central 64×64 image, source/endpoint oracle proposals, and candidate presentation order are identical in all three worlds. The negative world has **one actual adjacent image identical to the A positive and the other identical to B**, so one-neighbor evidence alone is insufficient. Original V24 renderer and source-conditioned V23/V24 prototypes unchanged except explicit disjoint base seeds.

V25 splits: **200 entirely new bases/600 worlds TRAIN** at seed 202683010, **100 bases/300 worlds CAL** at seed 202693010, **160 bases/480 worlds TEST** at seed 202703010. All base groups differ from V23 and V24 by seed. 2D and 2.5D HGB are trained with exactly the same 600 labels, 2D and 2.5D tiny real PyTorch CNNs each 12 epochs, 3 conv blocks [16,32,48] + 96-unit head, AdamW lr=0.001, batch48, two genuine initializations 11/23. CNN sees three 64×64 image planes, plus the same oracle source/endpoints rasterized as visible Gaussian coordinates; 2D CNN repeats central plane. No historical pretrained CNN or ANZA/LIRA checkpoints loaded. All model weights and link-acceptance thresholds fixed by TRAIN/CAL **before generating TEST**. Each method chooses acceptance and margin thresholds from only 100 CAL nuisance groups subject to observed false-acceptance ≤2% and wrong-branch acceptance ≤1%, then opens fresh TEST once.

## Held-out TEST, independent case-level audit

| Method | Correct accepted links /320 | Wrong accepted branch /320 | False accepted NO_LINK /160 | 3-class overall accuracy |
|---|---:|---:|---:|---:|
| Original source-conditioned V23 mean | **320** | **0** | **8** | **98.333%** |
| New V24 source-conditioned two-neighbor minimum consensus | **317** | **0** | **7** | **97.917%** |
| **HGB trained on identical V25 hard negatives and same 2.5D feature vector** | **320** | **0** | **4** | **99.167%** |
| 2D HGB, same TRAIN labels, identical central pixels | 0 | 0 | 0 | 33.333% |
| 2D CNN seed11 /seed23, central-only information | 4 /1 | 4 /1 | 4 /1 | 33.333% each |
| Real image+oracle-marker **2.5D CNN seed11** | **105–106** | **5–7** | **0** | **55.2–55.4%** |
| Real image+oracle-marker **2.5D CNN seed23** | **0** | **5** | **0** | **33.333%** |

A separately successful initial runner \`43593001860\` produced 105 accepted positive links for 2.5D CNN seed11 and 7 wrong, while independently audited run \`43593052911\` produced 106 accepted and 5 wrong. This is numerical/learning-run sensitivity; the two different initializations already show marked failure of this *small, fixed 12-epoch CNN* to reliably learn the hard relation. In both runs, **V23 mean, V24 minimum and hard-trained HGB results were identical**, as were V25 predeclared source/method training populations. Do not misreport this weak CNN as a representative large 3D CNN or CNNs in general.

Consensus-minus-hard-trained HGB three-way accuracy: **−1.25 percentage points**, paired 95% base-cluster bootstrap CI **[−2.50, 0.00] p.p.**, 3000 resamples over 160 independent scene bases rather than 480 dependent counterfactuals. **No superiority of V24 over this equal-hard-trained baseline.** Although HGB has better point estimates (99.17% vs 97.92%), the CI touches zero and does not prove HGB is universally superior either.

**Predeclared synthetic error-risk budget FAILED for all top three methods:** V23 mean 8/160=5.00% false links, V24 min 7/160=4.375%, equal-hard HGB 4/160=2.50%, each **exceeding the fixed 2% false-NO_LINK ceiling** on TEST. All had zero wrong accepted branch events. A CAL-only empirical 2% cap did **not** reliably transfer to new TEST groups. The overall superiority gate is therefore **FALSE**; no claims of risk-controlled generalization. Small calibration negative group count 100 makes guarantees particularly weak.

## Why V25 changes the Tver story
V23 and V24 demonstrated that *designed cross-slice continuity is useful* in a synthetic nonidentifiability setting, especially when testing a fixed model under new fake links. But V25 shows a supervised HGB **can learn an even stronger distinction from the same visible features** when difficult negatives are included in TRAIN. The prior V24 superiority point estimate was **training-distribution-dependent**. V25 further shows that empirical accept/reject thresholds calibrated with only 100 negative base worlds do not guarantee even an empirical 2% false-link rate on fresh 160 groups.

The research contribution should thus be framed as a **controlled protocol and interpretable structural consistency prior**, with explicit abstention and patient-like scene-group evaluation, not a winning standalone ANZA–LIRA architecture. A plausible new contribution would be out-of-distribution risk calibration with much larger, genuinely independent calibration sets, or a differentiable consistency constraint for a matched-information CNN/graph model; both need a new protocol and fresh heldout tests.

## What the results do NOT validate
No actual pretrained ANZA/SBPP/LIRA weights, learned ANZA segmentation, predicted endpoints, naturally expert-verified geological fault identities, or real CRACKS full-volume E2E evaluation are present. The actual real F3 CRACKS evidence remains a **separate diagnostic** (V22 raw central gradient 38/40 vs 17/40 geometry tie), not proof a trained network performed at that accuracy. Published FaultSeg3D, STA-Fault3D and other 3D segmentation systems solve fundamentally different tasks and were **not** run head-to-head. To claim their superiority is disproved, or that the full ANZA-LIRA is state of the art, would be scientifically false.

**Scientific judgment:** retain V23/V24 as useful mechanism studies, but V25 is the most important comparative result for a Tver manuscript precisely because it removes the unfair easy-vs-hard training asymmetry and reports the negative outcome. Maintain PR in Draft.
