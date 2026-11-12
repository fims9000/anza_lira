# ANZA–LIRA V13/V14: OOD synthetic controlled continuation (12 Nov 2026)

**Status:** reproducible *synthetic* negative-result audit. This text is a research receipt, not a claim of real CRACKS performance. The full model/CSV/code ZIP is supplied separately in the research conversation and is **not** attached to this repository commit.

## Scientific question and fixed decision budget

Given two generator-supplied candidate endpoints and a source endpoint, select **candidate 0, candidate 1, or NO_LINK** while controlling empirical false bridges (negative-scene rate <=2%) and wrong-branch selections (positive-scene rate <=1%). Original base HGB+CNN scores and weights frozen. No historical real-domain ANZA/LIRA checkpoints; these are V8 synthetic models.

In-family V12 had a positive result on its fresh 1600-scene test (35/510 difficult correct with extra geometry vs 11/510 from original strict consensus, at 9/617 vs 3/617 false bridges). This is **not** evidence that the same improvement persists under synthetic distribution shift.

## V13 first held-out OOD test (1,200 scenes; two new generator families)

393 hard true, 325 easy true, 482 negative; joint three-way selector trained on older V10 scenes, threshold calibrated on older V11 scenes, no test tuning. Background groups of five scenes.

| Method | Correct hard | False bridges | Wrong branch / 718 positive |
|---|---:|---:|---:|
| Original HGB+CNN consensus | 2/393 | 2/482 | 1 |
| Frozen V12 score-only | 5/393 | 9/482 | 3 |
| New permutation-equivariant joint score-only | 6/393 | 10/482 | 3 |
| New joint with geometry | 8/393 | 7/482 | 3 |

Against frozen V12 score-only, hard recall difference +0.76 percentage points, paired group-bootstrap 95% CI **[0, +1.76]** (includes zero). Prespecified improvement criterion **FAILED**. In this difficult positive subset, correct conditional candidate *ranking* was ~61.3%, but correct accepted links were only 8/393: low coverage of hard scenes under strict acceptance risk.

Pre-test protocol SHA256 `70c783b95bb8df9d7d4dac3909bd2c17b9047a7d463e9f042ad3ea690f31c6fe`. A superseded preparatory lock was replaced before opening test outcomes for an input-column technical correction; both receipts are retained in the separate study ZIP.

## V14 first held-out OOD test (1,200 new scenes)

A fresh 1,200-scene CAL cohort calibrated new models; V14 TEST unused until locking. Two newly trained selectors each received the **same 2,800 TRAIN labels** from V10+already-opened V13. Compare matched-label score-only vs a geometry-rich model with observed tangents/curvatures from both endpoints.

394 hard true, 332 easy true, 474 negative.

| Model | Correct hard | False bridges | Wrong branch / 726 positive |
|---|---:|---:|---:|
| Frozen older V12 score-only (not label-matched) | 4/394 | 4/474 | 3 |
| Frozen older V13 joint geometry (not label-matched) | 3/394 | 3/474 | 1 |
| Newly fit score-only, 2800 labels | 1/394 | 4/474 | 1 |
| Newly fit bi-endpoint geometry, 2800 labels | 1/394 | 2/474 | 1 |

Matched-label hard-recall difference **0.00 p.p.**, paired group-bootstrap 95% CI **[-0.76, +0.75]**. Improvement criterion **FAILED**. Lower observed false bridges 2/474 vs 4/474 is not proof of a universal safety improvement.

Protocol SHA256 `be1c16dcfac3c8094500540821cca017adf4d469599404245537aa2ea832c44d`.

## Post-test geometry ambiguity audit (NOT prospective evaluation)

Construct C1 cubic Hermite curves between visible source and both candidate endpoints with tangents estimated solely from the visible portions. At exploratory maximum curvature <=0.04 pixel^-1, **both alternatives** were geometrically feasible in **250/393 V13** hard positives and **258/394 V14** hard positives. This is a diagnostic *geometric* ambiguity only: it neither proves two geologically valid faults nor a nonzero Bayes error under a real geological distribution. In observationally identical hypothetical worlds carrying different hidden labels, a deterministic 2D decision cannot be correct for both; this is an identifiability limitation, not a novel theorem.

## Next falsifiable step

Stop retuning classifiers on opened V13/V14 tests. Instead, independently evaluate whether **actual adjacent seismic inlines** from the F3 volume improve hard continuation ranking under a fixed 2D false-link budget. CRACKS contains 400 position-indexed seismic sections but only 40 expert-annotated sections; all from one 3D source. Preserve spatial blocks (and a gap) across TRAIN/CAL/TEST; do not use hidden expert centerline or endpoint labels for candidate generation. Negative real pairs need instance-identity ground truth or should remain ambiguous. Report predicted-ANZA candidate recall separately from pair selection and reconstructed path success. Existing 49/49 oracle-endpoint bridges on CRACKS are NOT a full model result.

Background: CRACKS v2 DOI 10.5281/zenodo.13926822; selective classification is established (El-Yaniv & Wiener, JMLR 2010) and not a claimed new invention.

**Repository caveat:** This record is scientific context on PR #25. The primary source scripts, models, frozen CSVs, and digest-manifest currently exist in a separately delivered ~5.7 MB ZIP; do not claim this PR alone can reproduce V13/V14 until the corresponding source archive is incorporated or cited in a persistent public repository.
