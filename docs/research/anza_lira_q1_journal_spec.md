# ANZA-LIRA Q1 Journal Protocol

Status: pre-registered research plan for the next ANZA-LIRA journal line.

This protocol is deliberately separate from the historical `docs/research/anza_v2_master_spec.md` and from every frozen STOP/PASS cycle already recorded under `.codex/notes/`. It does **not** reopen ANZA-HS, ANZA-FS, ANZA-EK, ANZA-KS/KIR, SurfTrack S0, TraceGraph P0 Endgame V1, LIRA Final F1, Intervention, Graph-Cut V2, H1, or Structural Stability V1.1.

The scientific center of the journal paper is no longer "another anisotropic convolution". The target problem is **risk-controlled structural continuation**: decide whether two observed fault fragments belong to the same continuation before any path is drawn.

## 1. Central paper claim

The paper will test the following bounded claim:

> Explicit fragment-pair reasoning, combined with soft candidate generation and a calibrated LINK / REJECT / REVIEW decision, can recover more valid structural continuations at a fixed false-link budget than geometry-only or ordinary 2-D context baselines. Cross-slice context is evaluated as an additional source of evidence, not as evidence for Anosov-specific dynamics.

The final pipeline is:

```text
seismic evidence
    -> ANZA/local geometry (published component, not new novelty)
    -> SBPP candidate generation
    -> contextual pair model (LIRA)
    -> calibrated LINK / REJECT / REVIEW decision
    -> maximin path only after LINK
```

The paper must keep the four error layers separate:

1. visible-structure segmentation failure;
2. candidate-generation failure;
3. wrong structural-identity decision / false link;
4. path-construction failure after a correct link decision.

Dice alone is not a sufficient endpoint for this study.

## 2. Frozen evidence that must not be re-tuned

The following results are historical/frozen evidence and are used only to motivate the new protocol.

### 2.1 SBPP candidate generation

Frozen V3-B development evidence:

- BranchCandidateRecall@12 = `2611 / 2688 = 0.971354`;
- median candidate count = `1`;
- p95 candidate count = `5`;
- `weak_branch_continue = 117 / 192 = 0.609375`.

Interpretation: candidate generation is strong overall but weak branches remain a named failure stratum.

### 2.2 Historical / controlled LIRA evidence

The historical controlled continuation experiment reported:

- AUROC = `0.9923`;
- automatic recovery = `86 / 128 = 67.2%`;
- false links = `0.78%`;
- maximin path success with the correct pair supplied = `100%` in that protocol.

These numbers are not natural-gap external validation and must remain labeled as controlled historical evidence.

### 2.3 Fresh TraceGraph P0 Endgame evidence

On the frozen fresh-development cycle:

- AUROC = `0.978948`;
- top-1 = `0.978503`;
- safe-threshold recovery = `0.447173`;
- false bridges = `0.016493`;
- wrong-branch rate = `0.002687`;
- NONE recall = `0.983507`.

At a recovery target near 0.87 the observed false-bridge rate would have risen to about `0.155382`. Therefore the ranking result is strong but the operating-point problem is real and must be treated as a first-class scientific question.

### 2.4 Natural-gap blocker

The frozen real-gap audit found only `3` positive natural gaps in LIRA calibration and `1` in LIRA development. Across all already-opened non-confirm sections there were `76` positives, which is insufficient for independent calibration/development/confirm cohorts under the frozen protocol.

CRACKS provides raster semantic annotations, not geological fault instance identifiers. Local skeleton trace IDs are defensible local objects, but they are not global fault identities through crossings or disconnected components.

### 2.5 Cross-slice causal evidence

SurfTrack S0 established that the controlled task was not identifiable from the center slice alone (center-only AUROC about `0.4972`) while an ideal adjacent-history oracle achieved Top1 `1.0`. The fitted ANZA-Cocycle selected `lambda = 0`; the Anosov-specific effect failed the predeclared practical gates.

Interpretation: cross-slice information is worth testing; a new Anosov mechanism is not.

## 3. Explicit no-go rules

The journal line must not:

- create another Anosov / Koopman / entropy / cocycle rescue architecture;
- present determinant-one or hyperbolic parameterization as the main novelty;
- treat CRACKS raster connected components as ground-truth geological fault instances;
- tune thresholds, candidate geometry, splits, or hard-pair definitions after confirm/test labels are opened;
- claim real natural-gap repair from the historical controlled hidden-gap experiment;
- compare LIRA to segmentation networks only by raw Dice and call that a structural-continuation baseline;
- demand an arbitrary `10-15%` improvement. The primary comparison is recovery at a fixed false-link risk.

## 4. New data target: pair identity, not another binary mask

The largest current scientific blocker is ground truth for the relation:

```text
SAME CONTINUATION
DIFFERENT FAULT / DIFFERENT BRANCH
UNCERTAIN -> REVIEW
```

### 4.1 Primary real-data annotation unit

Each annotation item must contain:

- source fragment and candidate destination fragment;
- the central seismic section;
- a small fixed stack of neighboring sections (2.5-D context);
- the local dense fault evidence;
- no model score or model recommendation shown to the annotator;
- a label from `{SAME, DIFFERENT, UNCERTAIN}`;
- optional reason code: crossing, near-collinear independent fault, weak continuation, branching/Y, low signal, other.

The pair set must be sampled before labels are collected and stratified to include easy negatives and hard near-collinear/crossing cases. Model failures must not be used to define the confirm set.

### 4.2 Human annotation protocol

Preferred protocol:

- at least two independent geophysical reviewers for the confirm subset;
- disagreements are preserved, not silently overwritten;
- adjudication is separate from first-pass labels;
- report agreement (Cohen kappa for two raters or Krippendorff alpha for more/general missingness);
- `UNCERTAIN` is a legitimate outcome and maps naturally to REVIEW.

If only one expert is available, the paper must describe the labels as expert interpretation, not objective geological truth.

### 4.3 Sample-size freeze

Do not choose the confirm size from model results. Run an annotation-only pilot to estimate class prevalence and rater disagreement, then freeze a confirm size based on confidence-interval precision for false-link risk and recovery.

As a practical lower bound, aim for at least `300` confirm negatives if the paper reports a false-link operating point around 1-3%, and at least `300` confirm positives for recovery. Larger grouped samples are preferred. Statistical uncertainty must be aggregated by seismic section / structural group rather than by pixels.

## 5. External datasets and their roles

### 5.1 CRACKS / Netherlands F3

Use for the existing crowd-to-expert setting and for the new pair-annotation study. Preserve partial-label semantics: unlabeled raster regions are not automatically confident background.

### 5.2 Thebe

Use as an external real 3-D field-seismic segmentation/domain-shift dataset and, only if a defensible pair-identity annotation can be constructed or manually adjudicated, as an external continuation set.

Do not infer instance identity from a binary fault mask without an explicit rule and validation.

### 5.3 FaultSeg3D / controlled synthetic instance data

Use for controlled branch/instance identity where ground truth is known. This is useful for ablation, failure analysis, and stress tests but is not a substitute for real field validation.

## 6. Baselines required on the same pair task

All pair models receive the same frozen candidate set for the primary relation comparison.

### B0: geometry-only

Features may include distance, endpoint tangents, angle compatibility, ANZA local axis/support, gap length, and maximin pre-path support. No image/context encoder.

Purpose: quantify how much of the result is explainable by local geometry alone.

### B1: historical P0

Reuse `path_completion.pair_classifier.EndpointPairClassifier` and the frozen feature interface where possible. Do not silently change the historical baseline.

### B2: ordinary 2-D context

A small Siamese/CNN pair classifier using the same central-section context as LIRA but without ANZA-specific geometry.

### B3: strong 2-D semantic backbone

Use a standard modern backbone/feature extractor consistent with the available large fault-segmentation benchmark (e.g. UNet/UNet++, DeepLabV3+, SegFormer front-end). Compare on the pair task, not only on segmentation Dice.

### B4: 2.5-D context baseline

Stack neighboring sections with a simple non-specialized fusion rule (channel stack, late fusion, or shallow 3-D stem). This is the critical control for any proposed 2.5-D LIRA.

### B5: LIRA 2.5-D

LIRA pair reasoning with the same context extent and candidate set as B4, plus the declared ANZA/local structural features.

No transformer or architectural escalation is authorized unless B4/B5 expose a specific representational failure.

## 7. 2.5-D LIRA experiment

The first new model experiment must test a minimal question:

> Does neighboring-section information improve hard pair identity at the same false-link budget compared with a matched center-only model?

Freeze before development/confirm:

- slice offsets, e.g. `[-2,-1,0,+1,+2]` or another train-only selected set;
- crop size and physical normalization;
- candidate generator and K;
- architecture family;
- parameter/computation reporting;
- calibration method;
- hard-stratum definitions.

Primary strata:

- independent near-collinear faults;
- crossing/X configurations;
- Y/branching configurations;
- weak continuation;
- long gaps;
- low-confidence seismic evidence.

## 8. Decision theory: LINK / REJECT / REVIEW

Replace an informal "safe threshold" discussion with an explicit three-action decision rule.

Let `p = P(SAME | evidence)` and define losses:

- `C_FL`: cost of LINK when the pair is different;
- `C_MISS`: cost of REJECT when the pair is the same;
- `C_REVIEW`: cost of sending the pair to manual review.

Expected conditional risks are:

```text
R_LINK(p)   = C_FL * (1 - p)
R_REJECT(p) = C_MISS * p
R_REVIEW(p) = C_REVIEW
```

The Bayes action is the action with minimum expected risk. This produces a principled review region whenever review is cheaper than either automatic error near the decision boundary.

This is the preferred theoretical extension for the journal paper. A generic VC/Rademacher bound for LIRA is not a priority because it is weakly connected to the actual safety/operating-point problem.

## 9. Calibration and selective-prediction evaluation

The paper must report ranking, calibration, and operating behavior separately.

Required metrics:

- CandidateRecall@K before LIRA;
- AUROC and AUPRC for pair ranking;
- Brier score and a reliability diagram / calibration error;
- automatic recovery / true-link coverage;
- false auto-link rate;
- wrong-branch rate;
- REVIEW fraction;
- risk-coverage curve;
- recovery at common fixed false-link budgets, at least `1%` and `3%` when statistically estimable;
- group/bootstrap confidence intervals by seismic section or structural group;
- path success conditioned on a correct accepted pair.

Threshold/calibration selection occurs only on calibration data. Confirm/test is one-shot.

## 10. Statistical primary comparison

Primary comparison:

```text
2.5-D LIRA vs strongest matched non-LIRA context baseline
```

at the same frozen candidate set and the same false-link budget.

Primary effect:

```text
Delta recovery at fixed false-link risk
```

with grouped confidence interval.

A Q1-facing positive result does not require an arbitrary percentage gain. It requires that the claimed incremental effect survive a matched baseline, independent confirm data, and uncertainty analysis.

If the incremental LIRA feature contribution is negligible but 2.5-D context is strong, report that honestly and recenter the paper on explicit pair identity + selective decision rather than ANZA-specific superiority.

## 11. Reproducibility contract

Every journal-phase run must save:

- immutable split / annotation manifest with hashes;
- exact config;
- code commit SHA;
- candidate manifest;
- calibration receipt;
- `metrics.json` with denominators;
- per-section / per-group prediction table;
- model checkpoint where licensing permits;
- figure/table generation script.

External data and large checkpoints stay out of Git. Dataset provenance and licenses must be documented.

## 12. Phase order

### J0 - protocol and data audit

1. freeze this protocol and claim boundary;
2. map existing CRACKS/F3 data to 2.5-D section stacks;
3. audit Thebe/FaultSeg3D availability and label semantics;
4. build pair-annotation export format and blind review UI/data bundle;
5. run annotation-only pilot and freeze confirm sample size.

No new model selection is allowed before J0 is frozen.

### J1 - baseline harness

Implement B0-B4 on one common pair dataset and one candidate manifest. Reuse historical P0 code rather than reimplementing it.

### J2 - minimal 2.5-D LIRA

Implement B5 with the same context and compute budget reported next to B4. One architecture family first; no architecture search storm.

### J3 - calibration/selective decision

Fit the declared calibrator and LINK/REJECT/REVIEW loss policy on calibration only. Produce risk-coverage and false-link-constrained recovery tables.

### J4 - one-shot confirm

Open confirm once after code/config/model hashes are frozen. Preserve negative result if the primary incremental gate fails.

### J5 - external / controlled triangulation

Use Thebe for real 3-D/domain-shift evidence where labels support the claim; use controlled synthetic instance data for ground-truth branch identity and stress analysis. Do not use either to retroactively retune the primary confirm decision.

## 13. Journal narrative

The paper should be written around this sequence:

1. semantic segmentation gives local fault evidence but not explicit identity of competing fragments;
2. SBPP converts uncertain endpoints into a small high-recall candidate set;
3. pair reasoning decides whether a candidate continuation is structurally admissible;
4. cross-slice context is tested with matched controls because prior causal work showed the information exists outside the center slice;
5. calibrated three-way decision controls false links and makes abstention/review part of the method;
6. maximin path construction is downstream and only runs after pair acceptance.

ANZA is cited as the published local operator. Its old mathematical and hyperbolic branches are background/ablation material, not the new headline contribution.

## 14. Completion criteria for the research phase

The research phase is complete only when all of the following are available:

- frozen real pair-identity confirm set or a documented external blocker;
- common B0-B5 candidate manifest and evaluation harness;
- matched 2-D vs 2.5-D context comparison;
- calibrated LINK/REJECT/REVIEW analysis;
- grouped uncertainty intervals and hard-stratum table;
- external/controlled triangulation with claim boundaries;
- reproducibility bundle mapping every paper number to a machine artifact.

If real pair-identity labels cannot be obtained, stop the strong real-natural-gap claim. Do not manufacture instance identity from raster connectivity. The remaining controlled/semantic paper can still be published, but it must be positioned accordingly.
