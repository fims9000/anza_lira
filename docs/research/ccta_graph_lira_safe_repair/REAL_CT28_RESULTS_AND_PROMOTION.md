# Real 28-patient matched-CCTA result and promotion checkpoint

Date: 2026-09-21  
Branch: `research/ccta-graph-lira-safe-repair`

## 1. Raw-data recovery

The user uploaded `801-1000.z04` split into ten Google Drive parts.

The parts were materialized and concatenated into:

`801-1000.z04`

Reconstructed size:

`4,293,918,720 bytes`

SHA256:

`e113e44e4984e383da13637ef6ce18511b8509c8c19f9b30cfe0c216ad10d3f2`

The final split-ZIP member was not available in the execution environment. This was not treated as a blocker because all required ImageCAS CT entries 953–984 are fully contained inside z04.

A local-header fallback was added to the extractor. It scans valid ZIP local headers directly in z04, caches entry metadata, verifies compressed size / uncompressed size / CRC, and then reads the nested `.img.nii.gz` bytes exactly.

The scan found 48 valid ImageCAS CT entries in z04. All 28 frozen study patients were found and extracted successfully.

## 2. Alignment gate

All 28 patients passed the frozen ImageCAS / ImageCAS-X geometry checks:

- expected patient ID;
- shape;
- voxel spacing;
- full sform affine;
- raw ZIP CRC;
- NIfTI validity.

No resampling was used to force a match.

Result:

`28 / 28 PASS`

## 3. Cohort and frozen split

Official ImageCAS-X patient split remained unchanged:

- train: 17 patients;
- validation: 5 patients;
- test: 6 patients.

Frozen controlled relation examples:

- train: 379 positive + 379 hard negative;
- validation: 134 + 134;
- test: 167 + 167;
- total: 1,360 rows.

Thresholds were chosen on validation only. No held-out test threshold tuning was performed.

## 4. Held-out relation-presence result

Threshold rule:

`maximize recall subject to validation FPR <= 5%`

Held-out test:

| method | AUROC | recall | FPR | precision | TP | FP |
|---|---:|---:|---:|---:|---:|---:|
| nearest endpoint | 0.9189 | 16.17% | 4.19% | 79.41% | 27 | 7 |
| OGMC-style geometry | 0.8148 | 39.52% | 3.59% | 91.67% | 66 | 6 |
| geometry HGB | 0.9685 | 37.72% | 1.20% | 96.92% | 63 | 2 |
| radial 2.5-D CT | 0.9440 | 68.86% | 4.19% | 94.26% | 115 | 7 |
| **geometry + radial 2.5-D CT** | **0.9847** | **82.63%** | **1.80%** | **97.87%** | **138** | **3** |

The strongest comparison is geometry+CT against the stronger geometry HGB baseline, not against simple logistic geometry.

Observed geometry+CT delta versus geometry HGB:

- recall: **+44.91 percentage points**;
- FPR: **+0.60 pp**;
- precision: **+0.95 pp**;
- AUROC: **+0.0162**.

## 5. Patient-cluster paired bootstrap

Bootstrap unit = patient, using the six held-out test patients.

Geometry + radial CT minus geometry HGB:

- recall delta median: **+44.71 pp**;
- 95% interval: **[+32.62, +60.00] pp**;
- FPR delta median: **+0.61 pp**;
- 95% interval: **[-1.29, +2.56] pp**;
- AUROC delta median: **+0.0162**;
- 95% interval: **[-0.0058, +0.0400]**.

The recall improvement is consistently positive across the patient-cluster bootstrap. The FPR difference is small and uncertain around zero.

This is pilot evidence on six held-out patients, not a clinical population-risk guarantee.

## 6. Risk / coverage behavior

Held-out test under validation-selected budgets:

### Geometry HGB

- val FPR budget 0% -> test recall 15.57%, test FPR 0%;
- 1% -> recall 15.57%, FPR 0%;
- 2% -> recall 20.96%, FPR 0.60%;
- 5% -> recall 37.72%, FPR 1.20%;
- 10% -> recall 81.44%, FPR 4.79%.

### Geometry + radial CT

- val FPR budget 0% -> test recall 23.95%, test FPR 0%;
- 1% -> recall 57.49%, FPR 0.60%;
- 2% -> recall 58.08%, FPR 0.60%;
- 5% -> recall 82.63%, FPR 1.80%;
- 10% -> recall 97.60%, FPR 8.38%.

The CT-conditioned model therefore changes the risk/coverage frontier rather than merely increasing a ranking metric.

## 7. Pre-registered hard anatomy

Before the CT result was opened, the hard positive set was frozen as:

`OM1, IM, D2, LAD, OM2, R-PLA`

Geometry HGB -> geometry+radial CT held-out acceptance:

- OM1: 11.11% -> **100%**;
- IM: 18.18% -> **81.82%**;
- D2: 25.00% -> **100%**;
- LAD: 27.27% -> **63.64%**;
- OM2: 30.77% -> **100%**;
- R-PLA: 33.33% -> **80.56%**.

This is consistent with the hypothesis that CCTA intensity helps distinguish a real contrast-enhanced continuation from a geometrically plausible wrong branch.

### False-link sentinels

Geometry HGB observed false groups:

- LCX|OM2: 1/8;
- R-PDA|RCA: 1/16.

Geometry+radial CT reduced both of those to 0 in this test set, but produced three other observed false links:

- D2|LAD: 1/20;
- LCX|OM1: 1/24;
- R-PLA|RCA: 1/21.

Therefore the correct conclusion is not "CT removes false links". The supported conclusion is that it substantially increases true relation acceptance while pooled false relation risk remains low at the validation-selected operating point.

## 8. Promotion gate

Frozen internal pilot gate:

1. lower 95% patient-cluster interval for recall improvement > 0;
2. upper 95% interval for absolute FPR increase <= +5 pp.

Result:

**PASS**

The radial CCTA relation evidence is promoted to evaluation inside the frozen Graph-LIRA layer.

This does not authorize any test retuning of:

- relation confidence `tau = 0.85`;
- perturbation consistency `0.60`.

## 9. Scientific implication

The earlier six-case result was not a cohort accident in the direction that mattered most.

With 28 matched ImageCAS / ImageCAS-X patients, a simple image representation already provides strong patient-general relation evidence. The immediate bottleneck is no longer "does CT contain signal?"

The next question is:

> does the local CT gain survive global Graph-LIRA compatibility and the already-frozen selective policy?

Only after that question is answered should model capacity be increased.

## 10. Next frozen experiment

Immediate sequence:

1. keep candidate generation and geometry identity ranking frozen;
2. use the 28-patient image-conditioned relation evidence as the CT presence / continuation term;
3. insert it into the existing PAIR / JUNCTION / BOTH / NONE relation layer;
4. run the same global Graph-LIRA optimizer;
5. apply `tau=0.85`;
6. apply perturbation consistency `>=0.60`;
7. report:
   - repair-needed exact;
   - false structural repair;
   - incomplete / abstain;
   - coverage;
   - false among accepted;
   - exact among accepted;
   - patient-cluster uncertainty;
   - LAD / LCX / high-degree junction strata.

Only if this stage preserves the low-false objective should the project proceed to:

`radial CT vs compact CNN vs compact ANZA`

under the exact same patient split and graph policy.

## 11. Reproducibility

Expanded radial result bundle SHA256:

`bb4cee1c9c70b06541ef9b0c296c430f8ab2ae49d09811071f0b828874078ba3`

Resumable/local-header extractor source SHA256:

`8e858b5e291a709a797ec5e2d80612168dbbf4f14a3a56b15fa5bb4b518420af`

Postprocessor source SHA256:

`914191f4e483110cfbb2ac3adbabd9ddba773054c84814a44f0872e3d6be1131`

Raw medical images are not committed to Git.
