# Resume point: CCTA Graph-LIRA Safe Repair

Date: 2026-09-20

Active branch: `research/ccta-graph-lira-safe-repair`

## Canonical benchmark

Primary structural benchmark is now the 800-case ImageCAS-X bundle with official 560 / 80 / 160 patient split.

The old 921/953 two-patient experiments are historical controlled evidence, not the headline benchmark.

Current frozen task:

```text
candidate generation
-> pair / junction local ranking
-> relation existence (PAIR / JUNCTION / NONE)
-> joint Graph-LIRA
-> perturbation consistency
-> repair or abstain
-> downstream max-min path
```

## Main diagnosis

Candidate generation is not the bottleneck:

- true pair / junction candidate recall is 100% under the frozen stress protocol;
- test30 true pair top-3 is 99.15%;
- test30 true junction top-3 is 92.97%.

The dominant unsolved problem is **relation existence / branch identity**.

## Canonical broad-dataset numbers

Geometry-only relation-type system:

- test30 exact 55.46%, false 10.92%, incomplete 33.62%;
- test45 exact 51.37%, false 8.05%, incomplete 40.58%.

Perturbation-consistency gate selected on validation at 0.60:

- test30 coverage 74.34%, false among accepted 0.823%;
- test45 coverage 81.87%, false among accepted 0.863%.

Do not describe those selective results as high repair recall. Repair-needed accepted strata remain mostly incomplete.

## Latest no-CT result: binary-lumen shape-context proxy

Because matched CCTA intensities are unavailable, a controlled proxy experiment used the **broken binary vessel mask** as spatial context. It uses no anatomical segment labels as input and must not be called CT evidence.

Pair-presence ranking, geometry -> geometry+mask:

- test30 AUROC 0.8543 -> 0.9288;
- test30 AUPRC 0.6604 -> 0.8253;
- test45 AUROC 0.8166 -> 0.9001;
- test45 AUPRC 0.6104 -> 0.7837.

Junction-presence ranking:

- test30 AUROC 0.9100 -> 0.9314;
- test30 AUPRC 0.8628 -> 0.8949;
- test45 AUROC 0.8661 -> 0.8946;
- test45 AUPRC 0.8015 -> 0.8497.

At a validation-defined <=5% structural false budget:

Geometry+mask:

- test30 exact 56.50%, false 6.50%, incomplete 37.01%;
- test45 exact 51.60%, false 5.98%, incomplete 42.42%.

Versus the canonical four-class geometry relation head, patient-cluster bootstrap gives:

- test30 exact +1.04 pp, CI [-0.52,+2.58];
- test30 false **-4.43 pp**, CI **[-5.68,-3.20]**;
- test45 exact +0.24 pp, CI [-1.03,+1.47];
- test45 false **-2.07 pp**, CI **[-3.12,-1.03]**.

Interpretation: spatial vessel context gives a clear safety benefit for relation-existence decisions, but not a clear exact-rate gain. The cost is more incomplete / abstaining decisions.

Critical failure: degree-4 junctions remain bad. At test30, the 30 degree-4 scenes have 0% exact and 63.33% false under this proxy operating point.

## Next executable step without CT

The current test set is frozen. Do not retune thresholds on it.

A defensible additional no-CT experiment is robustness to **predicted-mask-like corruption** of the binary-lumen context:

- erosion / local thinning;
- dilation / boundary uncertainty;
- spurs / small false branches;
- isolated false positives;
- small holes / missing local vessel voxels.

Train / tune corruption parameters and any operating point only on train + validation, then report the already-frozen test once descriptively.

The question is not whether augmentation boosts test accuracy. It is:

> does the mask-context safety gain survive when the clean anatomical mask is made more like a realistic predicted segmentation?

This experiment is optional proxy evidence. It cannot replace matched CCTA.

## Primary blocked step

Obtain a true original ImageCAS CT corresponding to an ImageCAS-X patient ID. Then:

1. verify CT / mask / centerline geometry;
2. freeze matched cohort;
3. compare geometry, radial 2.5-D, 3-D tube, full cross-section CNN sequence;
4. test ANZA as an incremental local encoder;
5. keep the same Graph-LIRA and perturbation-consistency layers;
6. evaluate whether image evidence reduces confident wrong-branch / high-degree failures.

## Current documentation

- `LARGE_SCALE_800_CASE_REPORT.md`
- `MASK_SHAPE_CONTEXT_PROXY.md`
- `CHECKPOINT.md`
- `DATA_SOURCES.md`
- `FOUR_CASE_IMAGECAS_PILOT.md`

Machine artifacts live under:

`results/ccta_graph_lira_safe_repair/2026-09-20/`

Reproducibility scripts live under:

`scripts/research/ccta_graph_lira_safe_repair/`
