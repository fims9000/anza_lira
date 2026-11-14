# Tver V28 — real image predicted ANZA/SBPP/LIRA posthoc topology calibration (14 November 2026)

**Executed and independently audited.** [CI 43597110508](https://github.com/fims9000/anza_lira/actions/runs/43597110508) **SUCCESS**, [draft PR #33](https://github.com/fims9000/anza_lira/pull/33). Immutable artifact `tver-v28-posthoc-topology-predicted-sbpp` ID **14114698150** ZIP SHA256 **af3a35464e3373b286af54252c6d72dc7119ebfa5a04ff13e27e93be325df938**. Independent evidence receipt: **V28_INDEPENDENT_POSTHOC_TOPOLOGY_REAL_E2E_CASE_AUDIT_PASS**. No historical ANZA/LIRA checkpoints recovered.

## Chronology and scientific boundary
The first V27 real-CRACKS predicted-port pipeline ([CI 43596275581](https://github.com/fims9000/anza_lira/actions/runs/43596275581)) yielded **0/2** restored true paths for both ANZA and U-Net. ANZA's 0.6 CAL pixel-F1 threshold produced **DEGENERATE_DENSE_PREDICTED_MASK** in both heldout test scenes (0 real SBPP source ports and 0 candidate pairs). It therefore had **0 correct TRAIN pairs** and could not fit its own LIRA. U-Net found some true candidate geometry but its trained pair head was refused on CAL and no link passed. V28 was designed **AFTER** those two TEST sections 108,124 were inspected. It is a **posthoc engineering sensitivity**, never new confirmatory evaluation.

V28 kept *bit-for-bit identical original V27 newly trained* AZConv ANZA-v1 and matched U-Net state_dict checkpoints (120 TRAIN steps, 7 real expert sections) and changed **only** their segmentation-mask threshold-selection rule: evaluate the real predicted SBPP hard/soft-port chain on TRAIN 14 and CAL 4 annotated controlled intervention gap scenes, sweep the frozen 12 thresholds .35 through .90 at .05 increments. Admit a threshold only if TRAIN has >=4 correct and >=4 wrong predicted candidate pairs and CAL has >=1 gap with a correct proposed source/target. Select by CAL count of correct-candidate gaps, then fewer CAL proposed pairs, source coverage, density and larger threshold, without decoding TEST. The same 2 source-eligible TEST 108,124, held separate from V28 model fitting but **already exposed during V27**, were used *only after* freezing new TRAIN/CAL threshold and LIRA CAL gates.

## Outcomes and stage attrition

| Field | ANZA V27 | ANZA V28 | U-Net V27 | U-Net V28 |
|---|---:|---:|---:|---:|
| CAL mask threshold | 0.60 | **0.90** | 0.50 | **0.60** |
| True / false predicted TRAIN candidate pairs | 0 / 48 | **35 / 967** | 21 / 546 | **27 / 530** |
| Actually fitted pair-head | **NO / STOP** | **YES** | YES | YES |
| Predicted source available in heldout gaps | 0/2 | **2/2** | 2/2 | 2/2 |
| At least one correct predicted candidate | 0/2 | **2/2** | 1/2 | **2/2** |
| Correct accepted candidate after CAL safety gate | 0/2 | **0/2** | 0/2 | **0/2** |
| Correct maximin recovered path | 0/2 | **0/2** | 0/2 | **0/2** |
| False accept synthetic candidate-absent NO_LINK | 0/2 | **0/2** | 0/2 | **0/2** |

V28 is the first run in this real-data stream where both genuine (newly trained) ANZA/U-Net segmenters reached real SBPP source/target candidate extraction in **all two real TEST scenes** and each supported its own newly trained LIRA: ANZA head fitted from **35 true, 967 false** TRAIN pairs; U-Net head from **27 true, 530 false** TRAIN pairs. The rest of the ANZA→SBPP→LIRA→maximin executable software chain ran with no oracle ports and SHA-locked model files. The correct target is demonstrably **among** predictions in 2/2 test gaps, **but was not accepted**.

### What still failed
The new CAL selective relation head **chose accept threshold 1.01** (never-link) on each model; each model had only **4** CAL positive and **4** synthetic candidate-absent negative worlds. A high 0.95/0.97 score on a positive TEST pair is not permission to loosen that calibrated decision rule: synthetic negatives also obtained high maximum scores. Zero false links during zero acceptance is an abstention result, *not safety evidence*. Neither geometric baseline nor learned LIRA successfully recovered a gap. The pair ranking cannot yet be called validated physical fault identity.

## Independent scientific audit
[GitHub Actions 43597110508](https://github.com/fims9000/anza_lira/actions/runs/43597110508) separately checks all source/CAL receipt and per-case report SHA256 hashes, original and changed segmentation weights, independently recomputes every TRAIN/CAL 12-threshold feasibility and selected winner, validates both newly trained `.joblib` LIRA heads and every heldout source/candidate/decision/path count. It asserts V28 was designed after viewing V27 TEST, and no empirical false-link rate or published segmentation superiority is inferred.

## Publication-safe conclusion
A native ANZA-v1 operator can produce useful observed source/candidate proposals from **actual CRACKS F3 real seismic images** when its foreground mask is calibrated for **topological fragment extraction**, not merely pixel F1. The new LIRA layer can be **trained** on resulting detected pairs, with strict refusal when no defensible CAL threshold exists. **The experiment demonstrates feasibility of the executable stages and identifies a new relation-calibration bottleneck, not successful restored paths and not superiority over U-Net or 3D FaultSeg3D/transformers.**

Official CRACKS expert connected components do not establish natural geological fault-instance identity for disconnected sections; the synthetic NO_LINK intervention deletes a candidate. There are only **two** TEST source sections and they share one F3 volume. The source BLUE-eligible cohort was also selected after the original V26 zero-BLUE test issue. To claim broad model superiority requires previously unused expert-labelled seismic volumes with natural LINK/REJECT/AMBIGUOUS identity, externally evaluated E2E predicted segmentation+ports+relations, plus comparable published 3D neural baselines. Do not retrofit synthetic V23–V25 performance into real model quality.
