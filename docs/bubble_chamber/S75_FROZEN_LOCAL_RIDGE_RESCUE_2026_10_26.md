# S75 preregistration: local, pixel-only enhancement of weak ridge instances

**Frozen on 26 October 2026 BEFORE S75 calibration or TEST.** Research branch: `research/bubble-chamber-graph-lira`. Experiment ID `ANZA-LIRA-BC-S75-20261026`. Historical S73 and S74 TEST results already known; they motivate the new algorithm family but do not choose S75 parameters.

## Research question

In S73, deterministic Gaussian blur (sigma=1.1 px) dramatically reduced artificial track recovery on authentic H50Z image backgrounds. S74 global unsharp enhanced some correct joins but also increased false synthetic joins. Does a *local, pixel-only ridge support mask* improve end-to-end artificial 2-D path recovery without increasing wrong graph connections, using identical S69 orientation-invariant synthetic evaluation?

## Frozen operators / no labels in inference

All operators have the same `preprocess(gray: uint8[192,192], arm)` interface, never see generator geometry or synthetic labels. They are deterministic and apply *identically* to clean and blurred images, with no pre-flag indicating whether the image was blurred.

* `identity`: exact original bytes.
* `coherent_dog_080`: compute multiscale dark-blob/high-pass response `max(Gaussian(gray, sigma=2.2) - Gaussian(gray, sigma=.65),0)`; compute local gradient *structure-tensor coherence* on Gaussian(gray, sigma=.9), tensor smoothed sigma=1.4; subtract up to 0.80 times the ridge response, weighted by `clip((coherence-.40)/.40,0,1)` and a local signal floor `clip((ridge_response-1.5)/4,0,1)`. Cap the enhancement magnitude at 20 gray levels. No global sharpening, no new connected components inserted by source geometry.
* `local_blackhat_070`: black-hat response on original uint8 gray using a 5x5 elliptical structuring element, subtract 0.70 times `clip((blackhat-2)/8,0,1)*blackhat`, capped at 16 gray levels, but only where a 3x3 directional gradient coherence is at least 0.5. This highlights local dark ridges but can also emphasize artifacts; no physical GT enters its calculation.

These are *fixed*, not tuned on S75 TEST. Existing S56 geometry-only scorer, original source-photo train/cal seeds and logistic model, exact S72 SHA-locked source files, S72 direction-specific CAL detector (`005 -> 006`: `s68_branch_endpoint_refine`, `006 -> 005`: `s39_original`), S59 strict pair generation, S56 cutoff and MILP margin .04 stay unchanged. S69 corrected endpoint evaluator used only AFTER predictions for every arm.

## Source and experiment split

VisualMetro film 070 provides 145 audited, expert-measured *2D source-frame* templates (train/CAL/TEST 79/30/36 by whole frame), NOT native H50Z physical annotations. Genuine H50Z 512-px backgrounds consist of 39 tiles from 2 photos of one physical run (source 005:21, 006:18). Generated synthetic 3-piece tracks and synthetic decoy are injected on these real-photo backgrounds. H50Z physical correspondence and real native particle IDs remain UNKNOWN.

CAL on *source-photo* backgrounds, visualMetro 30-frame CAL only: `005=20267501`, `006=20267511`, 24 complete synthetic scenes in each direction. Each scene is evaluated clean and with the S73 `blur_sigma_11` synthetic blur; every arm sees precisely the same input pixels for that condition.

Fresh held-out TEST on opposite-photo backgrounds, visualMetro 36-frame TEST only: `005 -> 006: 20267503, 20267504`, `006 -> 005: 20267513, 20267514`; 24 synthetic scenes/seed, total 96 new scenes. Evaluations on paired **clean**, `blur_sigma_11`, and *TEST-only descriptive* `blur_sigma_16` (Gaussian sigma=1.6 px). The `blur_sigma_16` distribution is **not** used for CAL selection or later hyperparameter tuning.

## Frozen CAL-only selection rule

For a non-identity arm to qualify against identity on the 24 CAL scenes of its source direction:
1. At least **+2 synthetic-correct global joins** on sigma=1.1 blurred inputs;
2. **No more** synthetic-noncorrect selected joins and no fewer complete 3-piece artificial paths on blurred inputs;
3. On clean inputs: no more than **2 lost correct joins**, no more than **2 added synthetic-noncorrect joins**, no more than **1 lost full path**;
4. Zero global graph MILP failures on either condition.

Of eligible arms choose highest blurred synthetic-correct joins, then fewer blurred wrong joins, then fewer clean wrong joins, then lexicographic arm name. If none qualify, **freeze identity**. Record CAL file SHA-256 and locked S72 calibration SHA before running TEST. No TEST-based reselection. All arms tested and disclosed, including negative outcomes. Passing synthetic metrics cannot authorize real H50Z joining.

## Evaluation/reporting

Count synthetic-correct selected joins, synthetic-noncorrect/unmatched/wrong ports, full artificial 3-fragment paths, candidate segments, solver failures, scene error incidence, for CAL and TEST by direction and condition. 1200 paired bootstrap resamples at the whole-*scene* level; all different conditions for one scene are correlated. Report both absolute and per-direction changes. The held-out native-physics GT count must remain ZERO; no learned physical mass or 3D reconstruction. Original S73/74 historical reports remain immutable.

## Integrity requirements

Fail if immutable S72 CAL digest/locked source hashes, photo-bank count, source-frame split hash, frozen S75 protocol/operator SHA, CAL method or scorer cutoff drift. Do not save raw private images, per-scene coordinates, frame IDs, pseudo-label pixels, model weights or hidden reviewer answers to public GitHub. Persist only safe source, tests, aggregate JSON, experiment run ID and SHA checksums.
