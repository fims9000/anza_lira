# S67 — frozen pixel-only instance continuity experiment (2026-10-26)

**Status at authoring:** frozen before S67 CAL and TEST. Prior S64–S66 TEST results have been read and are exploratory motivation, not tuning targets. New S67 seeds are reserved and distinct from all previous S56–S66 seeds. No physically verified H50Z dense track identities exist.

## Scientific question

Can pixel-only, orientation-consistent local stitching across skeleton junctions recover more complete *artificial* three-piece chains without increasing erroneous global links, versus the unchanged S39 extractor?

## Predictors, no GT in inference

- `s39_original`: original `ridge_evidence` + `extract_ridges` (unchanged, no GT).
- `s67_crossing_orientation_v1`: same ridge mask, skeletonize, detect junctions degree >=4, cut 3×3 junction neighborhood, trace branch components >=18 pixels, pair only nearly collinear oppositely oriented tangents. Fixed maximum 25° deflection, endpoint-to-junction radius 8.5 px, margin 0.08 vs rival. Fail closed for ambiguous pairings and cycles. Reconstructed centerline continuity is an *algorithmic hypothesis*.
- Both feed unchanged `pair_rows`, `filter_rows(...,'strict_s39')`, original S56 geometry-only logistic scorer, S56 CAL-only 5% negative-pair threshold, and global MILP stability margin 0.04. No AZUNet or U-Net training.
- Inference sees only a uint8 192×192 image, the prefit scorer, threshold, and frozen algorithm name. Injected artificial fragment coordinates, true identity, source frame IDs, simulated gaps, camera IDs, and centerline labels are passed ONLY into post-hoc evaluation.

## Fixed data and IDs

- Source: 145 audited expert *point-only* film-070 frames, split 79/30/36 by whole frame according to `split_source_frames`. They do not match the H50Z image events.
- Real H50Z: exactly 39 unlabeled private 512×512 tiles (21 for image `005`, 18 for `006`) from the same run. Brightness and textures are genuinely photographic; embedded strokes and their GT are artificial.
- S56 scorer `train_scenes` and `cal_scenes` source geometry as frozen in S56; refit S56 geometrical scorer only on source photo and TRAIN frames; threshold fit only on source CAL.
- **S67 source-photo CAL** (never used for model selection of synthetic lineage): source 005 seed 20266701, source 006 seed 20266711, 16 scenes each, scenarios rotate `parallel`, `near_parallel`, `crossing`, `scratched`, `empty_gap`.
- **Fresh destination-photo TEST**: source 005 → target 006 seeds 20266703 and 20266704, source 006 → target 005 seeds 20266713 and 20266714; 24 scenes/seed = 96 total, 384 true artificial links / 192 artificial three-piece chains. No reuse from S64–S66.
- CAL decision: choose S67 only if its CAL count of **selected correct artificial links improves by >=2** versus S39 and its count of selected *noncorrect* links and scene-level false-link incidence do **not increase**. Otherwise select S39. This decision is frozen before TEST; all TEST arms are still reported.
- TEST nonpromotion gate: S67 can only replace S39 in a *future* synthetic demonstration after independent data if it achieves more correct links, no more noncorrect links, no more false-link scenes, and at least as many full chains on TEST. TEST is never used for additional threshold tuning.
- Report per-scene paired bootstrap 95% confidence intervals grouped by complete synthetic scenes, **not by pixels/crops/track points**. Use 1200 paired bootstrap replicates with frozen seeds (20266789 and 20266799).

## Identity/provenance policy for Git

Persist public-safe `experiment_id`, `calibration_sha256`, `protocol_sha256`, `code_sha256`, `S56_model_code_sha256`, `source_split_sha256`, `git_base_sha` and aggregate metrics. Keep exact per-scene photography, frame folders/IDs, pixels, generated point coordinates, raw ZIP/7z, weights and detailed reviewer JSON private. Do NOT encode frame IDs in publicly reversible hashes. Git identifiers in the run registry are opaque **experiment/protocol identifiers**, not physical particle IDs. Native H50Z particle correctness remains UNKNOWN.

## Fixed failure criteria

No causal explanation of real physics from artificial accuracy. Do not turn overlap, crossings, textures, or paired images into automatic physical track identities. No increase in physical automation authority. An unexpected split, missing source crop, changed frozen source/algorithm SHA, inconsistent CAL, solver failure, or leakage must fail closed and be reported.
