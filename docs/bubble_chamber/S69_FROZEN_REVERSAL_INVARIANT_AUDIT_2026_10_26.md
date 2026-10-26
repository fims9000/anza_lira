# S69 — frozen correction of synthetic endpoint evaluation (2026-10-26)

**Frozen before the S69 corrected CAL and all fresh S69 TEST scenes.** Historical S64–S68 inference code, source S56 scorer and thresholds, S68 CAL/Test scenario seeds, and S39/S67/S68 methods are unchanged. S69 changes the *post-hoc evaluation* only.

## Proven evaluation defect

`make_three_piece_scene()` reverses the stored source points of each of seven **synthetic** fragments with independent probability 0.5. Historical `s64_pixel_to_path.audit_synthetic_scene` assumes the fragment arrays have forward orientation (`source[0]` right endpoint -1, `source[1]` left endpoint 0, etc.). Reversing the storage order flips some correct synthetic joins into `selected_synthetic_wrong_endpoint`. Unit regression: two exact joins, 0 correct under S64 after reversing source[1], 2 correct under S69. Fixed-orientation source still yields the original 2 correct.

S69 evaluator uses only known artificial true pair IDs and the nearest pair of source **endpoints**, independent of each source curve's array orientation. This reference information is used strictly *after image-only prediction and global graph selection*. Both original and corrected diagnostics must be returned for the **same** candidate graph. Any wrong endpoints, wrong identity pairs, and unmatched candidate curves remain wrong/unknown. Do not change original S64 evaluation module or overwrite its historical JSON reports; label earlier numerical values as potentially biased downwards.

## Methodology / immutable controls

- All 145 film-070 visualMetro point-only frames, original whole-frame 79/30/36 train/CAL/TEST split and digest. Film 070 geometry is **not** pixel GT for H50Z; 39 private H50Z 512px tiles are same-run backgrounds only, synthetically injected lines carry known source identities. No expert verified physical identities.
- Frozen S56 geometry-only classifier and CAL negative threshold per source photo; no refitting/tuning against S69 heldout results.
- Four frozen arms: S39, S67, S68 branch split, S68 bounded endpoint refinement. Exact predictor hash and S68 protocol hash must agree with S68 CAL saved after bug correction A.
- CAL reanalysis of S68's **source-photo** seeds 20266801 (005), 20266811 (006), 16 scenes each; selected arm re-evaluated with orientation-invariant metrics and original S68 strict promotion gate (+2 correct synthetic joins; no increase incorrect links, bad scenes, no decrease full chains). This selection is frozen before fresh TEST.
- Fresh withheld destination-photo tests: **005→006** seeds `(20266903,20266904)`, **006→005** seeds `(20266913,20266914)`, 24 scenes per seed; all fresh, **96 scenes, 384 synthetic truth links, 192 full path opportunities**. Generated with unchanged S56 artificial shapes over native H50Z photographic backgrounds. No method or threshold tuning after first TEST execution.
- Once independent S69 TEST is complete, replay the historical S68 TEST seeds `(20266803,20266804)` and `(20266813,20266814)` solely to quantify evaluation bias and reproduce previous old-count aggregates. This retrospective replay must not influence calibration/selection.
- Paired within-scene bootstrap (1,200 resamples) conditional on the simulated seed bank, no pixel- or crop-level pseudo replication. Report absolute sums and per-scene delta 95% CI for S39 vs each candidate; also old evaluator vs corrected same predictions. Tests remain synthetic-injection-only and no physically verified native joins.
- Store public-safe experiment ID, code/protocol/calibration/source split SHA-256, seeds, aggregate metrics, test count, CI and negative results. Do not upload H50Z pixels, per-scene generator coordinates, frame IDs, weights or private annotation files.

## Acceptance gate

Do not reinterpret corrected metrics as measured physics. Do not promote new S68 arm unless strict corrected CAL gate passes and fresh heldout synthetic TEST has no increase in noncorrect joins and bad-link scenes plus more correct links/full chains. Disclose old vs new evaluation mismatch. If recalculation changes conclusions, revise report transparently rather than silently replacing old output. Keep native graph joins disabled.
