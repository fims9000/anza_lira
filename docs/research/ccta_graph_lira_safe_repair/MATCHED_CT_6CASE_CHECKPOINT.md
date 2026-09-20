# Matched ImageCAS CT + ImageCAS-X six-case checkpoint

Date: 2026-09-21
Branch: `research/ccta-graph-lira-safe-repair`

## External blocker resolved for a six-case pilot

Six original ImageCAS CCTA volumes were extracted from the official Kaggle multipart archive and matched by the original ImageCAS IDs to ImageCAS-X annotations:

- train: 953, 964
- validation: 957, 966
- test: 980, 984

Raw CT is not committed. The exact CT SHA256 values are stored in the alignment artifact.

## Alignment audit

All six cases pass the same physical-geometry checks.

For every case:

- CT shape equals ImageCAS-X segmentation shape;
- voxel spacing is identical;
- CT and mask sform affine matrices are exactly equal (`max abs error = 0`);
- ImageCAS-X VTK centerlines map with the expected LPS -> RAS x/y sign conversion;
- 100% of centerline points are in bounds;
- 100% land inside a non-zero vessel voxel at nearest-voxel sampling;
- 100% match the anatomical segment label stored on the VTK point;
- CT SHA256 matches the extraction manifest.

| ID | split | shape | spacing mm | CT SHA256 |
|---|---|---|---|---|
| 953 | train | 512x512x223 | 0.318359375 x 0.318359375 x 0.5 | `ee9f43b79309950fb89420de5511d64364a9399032b7e3cf236895e0187cb2de` |
| 964 | train | 512x512x275 | 0.390625 x 0.390625 x 0.5 | `4c59eb24829bcb02544e548a6483de03a5967bdfa1b15b70701bd3c6e51ea8a5` |
| 957 | val | 512x512x231 | 0.341796875 x 0.341796875 x 0.5 | `5c402b3d73ac7e1cf2ba79080c6a9e8292f47541947b9ffbd5045b3f01d428d6` |
| 966 | val | 512x512x275 | 0.330078125 x 0.330078125 x 0.5 | `e251066e96567bec755dfb52022499ff74c008566b435a6ebfd5e2fe08c9e5c3` |
| 980 | test | 512x512x233 | 0.357421875 x 0.357421875 x 0.5 | `9aa45cd125ac65b269b4db6887a8c86eff1eb9d011474cbf0f1388206d205261` |
| 984 | test | 512x512x275 | 0.44921875 x 0.44921875 x 0.5 | `719fab0e73472330f383b1f2999ad0bd4509ff753c441e1470de985c73bc420e` |

This closes the earlier provenance problem caused by the row-indexed BDMAP mirror. These are true original ImageCAS IDs and exact ImageCAS-X physical grids.

## Pilot A: candidate-pair ranking on real CT

A first controlled candidate-pair pilot used 4 mm gaps, local wrong-branch decoys, radial CT corridor summaries, and the official patient split above. The same `StandardScaler + LogisticRegression(C=1, class_weight=balanced)` model was used for each ablation. Thresholds were selected on validation only under FPR <= 5%.

Test result:

| representation | AUROC | AUPRC | true-pair top-1 | recall at val-selected threshold | FPR |
|---|---:|---:|---:|---:|---:|
| geometry | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0455 |
| CT only | 0.8892 | 0.8245 | 0.8750 | 0.0000 | 0.0000 |
| geometry + CT | 0.9858 | 0.9851 | 1.0000 | 0.4375 | 0.0000 |

Interpretation: this easy candidate-ranking pilot is already saturated by geometry. It is not evidence that CT improves candidate ranking. It is useful as a negative control and is consistent with the 800-case conclusion that candidate ranking is not the main bottleneck.

## Pilot B: geometry-matched wrong-branch relation-presence stress

A harder controlled stress set was therefore created. Positive examples are true 4 mm within-branch gaps. Negative examples are nearby pairs from different VTK polylines and different anatomical segments. Each hard negative is matched to a unique positive using only fixed-scaled geometric features before any CT model is fit. Masks and segment identities are used only for controlled ground-truth construction; they are not model features.

Balanced examples:

- train: 66 positive + 66 hard negative;
- validation: 59 + 59;
- test: 47 + 47.

The decision threshold is frozen from validation by maximizing recall subject to FPR <= 5%.

Validation:

| representation | AUROC | recall | FPR |
|---|---:|---:|---:|
| geometry | 0.9210 | 0.1186 | 0.0339 |
| CT only | 0.9560 | 0.5085 | 0.0169 |
| geometry + CT | 0.9687 | 0.7458 | 0.0339 |

Held-out test:

| representation | AUROC | AUPRC | recall | FPR | precision |
|---|---:|---:|---:|---:|---:|
| geometry | 0.9873 | 0.9881 | 0.0213 (1/47) | 0/47 | 1.000 |
| CT only | 0.9357 | 0.9502 | 0.4681 (22/47) | 0/47 | 1.000 |
| geometry + CT | 0.9746 | 0.9760 | 0.6170 (29/47) | 0/47 | 1.000 |

Per held-out patient for geometry + CT:

- scan 980: recall 3/16 = 18.75%, false 0/16;
- scan 984: recall 26/31 = 83.87%, false 0/31.

The key pilot observation is operating-point behavior, not test AUROC: under the validation-frozen low-false policy, real CT evidence recovers many more true repair relations than geometry alone while producing no observed false repair in these 47 held-out negative examples.

This is still a very small six-patient controlled pilot. `0/47` is an observed count, not evidence of zero or sub-1% population risk. The large difference between test patients 980 and 984 also shows that image-conditioned generalization is not solved.

## Scientific status

Supported now:

1. the six CCTA volumes are correctly matched to ImageCAS-X anatomical labels;
2. real CCTA intensity carries usable continuation evidence beyond broken-mask occupancy;
3. on a geometry-matched relation-presence stress set, geometry + CT substantially improves recall at the validation-defined low-false operating point;
4. easy pair ranking remains geometry-saturated.

Not supported yet:

- population-level risk claims;
- natural clinical-gap performance;
- a final CT-conditioned Graph-LIRA improvement on the complete 800-case test set;
- benefit from ANZA, CNN sequence encoding, Transformer, or Mamba;
- a claim that CT improves average AUROC in every regime.

## Next frozen experiment

Do not retune the canonical geometry or uncertainty policy on these held-out test cases.

Use this exact matched-CT split to add image evidence to the same PAIR / JUNCTION / NO-REPAIR relation layer, then feed it into the frozen joint Graph-LIRA optimizer and the already frozen `tau=0.85`, consistency `0.60` selective policy. Compare radial 2.5-D, candidate-aligned 3-D tube, and full cross-section encoder representations. ANZA remains an explicit later ablation only.
