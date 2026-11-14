# Tver V29 executed: close rivals, calibration and the decisive geometry baseline

**Status: executed new synthetic generator; negative result for generic V25 region-hybrid superiority.** The complete generator, exact methods, frozen conditions and reject-threshold procedure were committed **before opening V29 TEST** in `V29_FROZEN_SYNTHETIC_NEAR_RIVAL_SELECTIVE_LINKS_20261114.md`. [GitHub Actions SUCCESS #43591845741](https://github.com/fims9000/anza_lira/actions/runs/43591845741), **10 unit tests PASSED**. Complete artifact **14112975931** `tver-v29-synthetic-risk-near-rival-procedural-probability` includes 9,000 per-condition/method/policy rows, 80 aggregate rows, initial and calibrated rules, source SHA and descriptive paired-group bootstrap receipts.

## What has and has not been evaluated

Methods are **the actual repository** `scripts.cracks_real_auto_core_v23.run_pipeline` geometry/nearest/context and `scripts.cracks_real_region_v25.run_regions` region_hybrid, with a real `NO_LINK` and fixed morphological-close comparison. They receive exactly the **same procedurally generated soft probability map**, which is NOT a synthetic seismogram passed through a trained AZConv detector. No full historical ANZA/LIRA checkpoints have been evaluated; neither `AZConv2d vs U-Net` nor trained LIRA novelty is established here.

Four cases per nuisance/background group: `easy_positive` continuous hidden curved trace, `near_rival_positive` hidden true continuation with close near-parallel distinct distractor, `indistinguishable_negative` same **bit-for-bit** soft image as easy_positive but hidden latent identity says disconnected, and `distinct_negative` nearby unrelated visible fragments with different orientations. There are 75 CAL groups×4=300 scenes, heldout 150 TEST groups×4=600 scenes. Gap lengths 12, 22 or 32px in equal fractions. No oracle endpoints/physical identities passed into inference.

For each scored link algorithm, apply a **CAL-only additional reject threshold** to existing accepted links: 21 CAL score quantiles and reject-all/accept-all sentinels, maximize CAL correctly connected positives subject to ≤1/75 false bridges on **distinguishable** CAL negatives. The observation-identical counterfactual negatives are deliberately **separate**, since no algorithm can infer the latent labels and imposing risk on them would force rejection of the identical positives. Test did not influence thresholds.

## New heldout synthetic TEST, 150 examples per class

| Actual existing linking policy | Easy positive correct | Near-rival positive correct / wrong branch | Visible-orientation distinct-negative false links | Observation-identical negative false links |
|---|---:|---:|---:|---:|
| V23 nearest, original | 50/150 | 50/150; wrong0 | 0/150 | 50/150 |
| V23 context, CAL reject only | 46/150 | 46/150; wrong0 | 0/150 | 46/150 |
| **V23 geometry, CAL reject only** | **150/150** | **148/150; wrong2** | **1/150** | 150/150 |
| V25 region_hybrid, original | 150/150 | 96/150; wrong54 | 0/150 | 150/150 |
| **V25 region_hybrid, CAL reject only** | **150/150** | **138/150; wrong12** | **0/150** | 150/150 |
| NO_LINK | 0 | 0; wrong0 | 0 | 0 |
| Binary morphology closing radius4 | 0 | 0; wrong140 (parallel branches merged) | 0 | 0 |

**Paired scene-group descriptive bootstrap** near-rival correctly joined (V25 calibrated minus V23 geometry calibrated) = **−10/150 = −6.67 percentage points**, 95% descriptive **[−11.33; −2.00] p.p.**, conditional on this generated line family. The region version had one fewer observed false bridge among distinguishable negatives but strictly worse near-rival accuracy, so neither method universally dominates every error criterion.

Before additional CAL rejection, V23 geometry had **21/150** distinct-negative false bridges. With CAL choice, **1/150** on TEST; V25 region_hybrid had **0/150** before and after CAL rejection, but its near-rival correct recovery improved from **96/150** to **138/150** because the added acceptance filter removed some extra wrong cross-component links and left the correct one. This is a meaningful *engineering* control of graph over-linking, not proof of physical no-false-link risk. CAL-only budgets and TEST empirical false fractions do not provide field safety guarantees.

## Per gap-band failure diagnosis, new heldout TEST

| 150 near-rival TEST scenes, by gap | V23 geometry CAL rejected | V25 region_hybrid CAL rejected |
|---|---:|---:|
| 12px gap, n50 | **50/50** | 44/50 |
| 22px gap, n50 | **50/50** | 44/50 |
| 32px gap, n50 | 48/50 | **50/50** |

There is a CONDITIONAL small long-gap advantage for region candidates at 32px, but it does not overcome the extra wrong short/intermediate competitor links (6+6 wrong over 100 shorter-gap scenes). Gap stratification was frozen by group index modulo 3 before TEST, not cherry picked.

## The counterfactual 2D identifiability negative control

Every `easy_positive` scene is paired with a `indistinguishable_negative` scene having **exactly the same observed floating-point probability array and exactly the same visible source and destination pixels**; only a latent source-instance identity flag differs. Tests verify that every deterministic algorithm produces **byte-identical completed masks and the same link decisions** for paired observations. On the new TEST the V23 geometry and V25 region policies accepted 150/150 real-positive links AND 150/150 identical-observation negative false links. This is an intentional mathematically unavoidable trade-off under this artificial observational equivalence, **not a new physical seismic false-positive estimate**.

## Strongest publication conclusion

1. **AZConv vs Conv** V28 has a real CRACKS benefit on expert anchor connectivity conditional on more ORANGE false pixels. HGB is a strong original alternative; no unconditional superiority.
2. **Link decision** V29 shows that on new *idealized curved-line probability scenes*, simple geometry can outperform the more complex V25 region-hybrid by a nontrivial margin on close rival branches under CAL-only reject thresholds. Extra component ports and two-link degree allowance are possible over-connection mechanisms, but a controlled ablation is still needed to attribute causality.
3. **Identifiability** cannot be decided solely from the central 2D observation where two latent worlds produce the same input. Additional *physically valid* adjacent-slice evidence or external semantic constraints are needed; the V15 synthetic neighbor-slice gains cannot be promoted to real 2.5D without their V16 negative shift results.
4. For the Tver manuscript add a compact honest comparison as a stress test, not a sprawling new model version or a claim that all analogues were defeated. No trained AZConv prediction, Frangi, clDice-trained U-Net, or externally independent geological instance ground truth was evaluated by V29. Synthetic generator is axis-preferential and does NOT approximate realistic F3 seismic dynamics.

All results, including negative controls and failures, should be retained. Historical main accepted work and master remain unchanged.
