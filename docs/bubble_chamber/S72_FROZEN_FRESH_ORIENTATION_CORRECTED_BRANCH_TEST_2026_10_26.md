# S72 preregistration — prospective synthetic-on-H50Z pixel-to-graph study, orientation-corrected at inception

**Frozen before any S72 CAL or new TEST is generated.** Date 2026-10-26; branch `research/bubble-chamber-graph-lira`. Research question: with S69's corrected endpoint evaluation applied before calibration, do the existing, completely frozen four S68 2-D pixel-to-graph algorithms achieve a safer whole-chain result on **new** synthetic scene seeds?

## Fixed predictor, dataset split and safety

- No new model training and no new detector tuning. Reuse the pre-S71 S68 image predictor unchanged (`bubble_chamber/s68_branch_preserving_instances.py` SHA-256 `dc16718c9bcac613eacc38771b2fa9a840266d9f7da377cc7538ae136bc4a91a`) and preexisting benchmark S68 (`scripts/research/bubble_chamber/s68_benchmark_branch_preservation.py` SHA-256 `f8a841eb7c6fc8e5074e7d88c667891a36f95be67d5a7eb86d145cc9c3fc3e71`). The unchanged S56 geometry scorer source SHA-256 `088024077e5f86b5f4308c5d23e72643c8b261a355cfb12f2bbc8c1d3d87355b` fits 64 TRAIN scenes on source-photo backgrounds, CAL negative-FPR threshold on original 64 source-CAL scenes; geometry global stability margin .04; S59 strict pair gate.
- Four arms remain S39, S67 crossing continuation, S68 branch split, S68 branch split+endpoint refinement; no test-time modification of one arm.
- Evaluator is **only** S69 orientation-invariant synthetic audit (`bubble_chamber/s69_orientation_invariant_audit.py`, SHA-256 `3e4ad80c0fc544b7b231b6a7b7dcdc3501484830a98d6d19c133dc0eb1c5b495`). Artificial identities are not passed to model inference. The old orientation-dependent metric is not used for CAL selection.
- Reuse S37 frame-disjoint source visualMetro 070 data (145 complete frames, 79 TRAIN/30 CAL/36 TEST) and 39 authentic H50Z 512px crops as background. Both H50Z photos belong to the SAME camera run; visualMetro film-070 source trajectories are not matching physical GT. No independently labelled real H50Z tracks.

## New scene IDs and selection policy

- CAL source H50Z 005: seed `20267201`, source 006 `20267211`, **32 scenes per source**, scenario order frozen to existing `SCENARIOS` round-robin. Source synthetic curves from 30 CAL source frames only. H50Z background **same as source** in CAL.
- Fresh TEST 005→006: seeds `20267203`, `20267204`; 006→005: `20267213`, `20267214`; 24 scenes each (96 total). Source synthetic curves exclusively from 36 TEST source frames; H50Z opposite photo backgrounds. These synthetic seeds have never been used in S64–S71.
- Original frozen S68 `choose_calibration_only` rule: candidate must improve synthetic-correct links by >=2 vs original S39, not increase synthetic-noncorrect links or scenes with errors, and not decrease full synthetic three-piece paths on CAL. If none eligible, choose original S39. **Persist immutable CAL JSON before TEST**, including full source hash and selected arm per direction. TEST loads saved CAL, checks exact SHA/protocol and original scorer cutoff; no method changes after opening test.
- TEST all four original arms for transparent negative/positive comparisons. Per-seed paired bootstrap with 1200 resamples *of complete artificial scenes*, not fragments/pixels. Report correct/other selected links, complete three-piece paths, any-error scenes, and method-vs-S39 CIs. Examine both photo-transfer directions separately. A test-positive conditional improvement is not physical robustness or independent real-run verification.

## Deployment and privacy

Never replace default native S39 automatically from S72, even if one synthetic seed looks favorable; there is zero native H50Z verified pixel-level identity. No 3D reconstruction, track mass, or automatic physical glue. Only source Python, unit tests, frozen protocols, public-safe aggregate counters, experiment ID, source/calibration/TEST SHA-256 fingerprints may be shared. Private BMP/PNG/7z, operator measured source coordinates, per-scene trajectories, frame-specific filenames, weights and individual source point IDs stay private.
