# S85 — TRAIN-only learned extra-line filtering, frozen research protocol

**Experiment ID:** `ANZA-LIRA-BC-S85-20261027`  
**Protocol date:** 2026-10-27. Written **before any S85 private-data TRAIN, CAL, or held-out TEST execution**.  
**Branch:** `research/bubble-chamber-graph-lira`. S72 remains the default. NO claims of physical tracking quality.

## Motivation and hypothesis

S80–S84 showed that photometrically strong new line instances sometimes restore synthetic links but often increase selected noncorrect/unknown joins; hand-written graph gates can reject both decoys and useful fragments. S85 tests whether a **small regularized supervised classifier of individual S80 extra-line hypotheses**, trained from **unambiguous synthetic examples only**, improves *complete three-piece artificial paths* without increasing wrong/unverifiable selected joins in new domains. Earlier S80–S84 TEST labels and metrics are motivation ONLY and never model training or threshold selection data.

## Fixed source boundaries

- Frozen S72 source-direction selected image-to-graph extractor, S56 fitted source-photo geometry scorer and original CAL probability threshold, S59 `strict_s39` pair gate, S69 orientation-invariant graph evaluator, S76 shape/endpoints auditor and S80 two-scale-consensus `+<=6` candidate generator are **unchanged**.
- Preserve all original S72 polyline objects, vertex order, and identity. S85 may append at most six **extra** S80 curves, never replace original lines or directly force links. S56 global graph solver chooses final joins with unchanged 0.04 stability margin.
- All model input features are derived solely from 192×192 image pixels and frozen S56 per-candidate contacts (13 named features). **No GT/source track IDs/scene source names/physical metadata** in inference. Curves reversed in vertex order must receive the same feature vector within float tolerance.
- Frozen model: sklearn `StandardScaler` + `LogisticRegression(C=0.25, solver=liblinear, max_iter=1000, tol=1e-9)`, deterministic seed 85, class-frequency reweighting and downweighting of scenes with many duplicated nuisance views. Numerical means, scales, coefficients and intercept stored in a **private** immutable `S85_TRAIN_MODEL_PRIVATE.json`, never in public Git. Public receipts contain model SHA256 and aggregate label counts only.
- Training labels are used exclusively in the TRAIN evaluator **after** blind pixel inference on a whole scene. Positive=extra instance confidently matched to one of six injected synthetic *true* fragments by corrected S76 shape matching; negative=extra instance confidently matched to the **injected synthetic decoy only**. Ambiguous, partly overlapping or unmatched lines are **excluded** rather than silently labeled wrong; unmatched native H50Z lines may be real physical tracks and MUST NOT be treated as training negatives. A source-direction model requires ≥12 positives, ≥24 negatives from ≥4 distinct scene groups in each class. If unmet, that direction **fails closed** (no added lines).

## Fixed 3-stage data isolation

Measured source templates from visualMetro film **070** only, separated by whole frame **79 TRAIN / 30 CAL / 36 TEST**, split SHA `448f21ca3a709010ed972a596ed046cadff0643112905fd1d029c91f8b14faef`. The H50Z photo backgrounds 005 (21 private tiles) and 006 (18) are **two photographs of one physical run** and have no expert-verified physical track GT. Film 070 geometry is not H50Z truth. Each artificial scene contains six three-piece target fragments and one decoy.

1. **TRAIN:** source-photo `005` seed `20268500`, source-photo `006` seed `20268510`; 64 synthetic scenes per source using only film-070 TRAIN frames; paired clean, Gaussian blur σ=1.1, and **synthetic nuisance scratch+grain**. Collect candidates by image-only S72→S80, then compute artificial reference associations in a separate TRAIN evaluator. Fit frozen separate models by source direction, record counts/abstentions and source/model hashes. No trained weights in public report.
2. **CAL:** source-photo `005` seed `20268501`, `006` seed `20268511`; 24 **different** scenes per direction from only film-070 CAL frames, paired `clean`, `blur_sigma_11`, `scratch_grain`. Evaluate `baseline`, `s85_learned_p85`, `s85_learned_p95` with fixed model-probability thresholds `.85` and `.95`. TRAIN model and code hashes must match. Use corrected S69 synthetic link evaluation and S76 shape/endpoint/full-path metrics **only after inference**. Source-CAL arm selection **before TEST**.
3. **HELD-OUT TEST:** source `005→006` seeds `20268503, 20268504`, source `006→005` seeds `20268513, 20268514`, 24 new scenes per seed → 96 total per condition. Film-070 TEST frames only, opposite H50Z photo backgrounds. Paired conditions `clean`, `blur_sigma_11`, `blur_sigma_16`, `scratch_grain`. No threshold/feature/model/arm changes after TRAIN/CAL. Replay identical TEST bytes on repeat. Report 1200 paired scene-bootstrap 95% intervals per direction and seed, NOT pixel-level or independent physical-run intervals.

Synthetic scratch+grain condition is a frozen pixel transformation: on each generated 192×192 scene add two thin diagonal dark scratches at random locations/angles using only a noise seed independent of synthetic GT, plus additive zero-mean Gaussian noise with standard deviation 9/255. No simulator coordinates may be consulted to place or suppress scratches. Never present this condition as genuine physical camera variation.

## CAL-only deployment gate (identical for both learned arms)

On SOURCE CAL `blur_sigma_11`: correct synthetic selected links ≥ baseline+2; wrong or unconfirmed synthetic links ≤ baseline; synthetic full three-piece paths ≥ baseline; shape-matched synthetic true fragments ≥ baseline+2. On clean and `scratch_grain`: noncorrect selected links ≤ baseline, full synthetic paths ≥ baseline, correct links not below baseline−1. All solver runs must succeed; TRAIN model must be available. Choose highest CAL blur correct, then fewer blur noncorrect, then fewer extras, then stricter p95. Otherwise select **baseline**. TEST cannot change the selection, even if S85 appears better post hoc. Beyond CAL the native physical auto-join gate is permanently FALSE pending expert reference data from independent runs.

## Model selection and evaluation caveats

A classifier's artificial precision, ROC/AUROC and AUPRC are **diagnostics**, not deployment criteria; primary efficacy is the **end-to-end graph count of complete correctly joined synthetic chains**, with a no-worse-error requirement. A small or one-class TRAIN set produces abstention, never a fabricated model. The opposite H50Z backgrounds are still the same physical event, so the TEST evaluates conditional synthetic transfer only, not independent physics domain generalization. No 3D identities, momenta or particle species are inferred.

## Public/private partition and status

The public repository may receive this frozen protocol, source-only modules, tests, CI, and public **aggregate-only** future run outputs. Source photographs, film-070 coordinates, model coefficients/checkpoints, per-example predictions, experiment-specific pixel arrays and private viewer ZIPs must stay private. No S85 numerical scientific outcome exists until real private data have been run through all three isolated phases. Every published result must carry SHA-256 hashes of protocol/operator/frozen model/TRAIN/CAL/TEST, exact seeds, counts and no-physics-GT status.
