# S64 preregistration: pixel-to-polyline-to-graph (2026-10-26)

**Branch:** `research/bubble-chamber-graph-lira`. **Status:** frozen BEFORE executing the held-out S64 TEST. This experiment uses zero H50Z native physics labels.

## Scientific claim boundary

The H50Z photo bank contains two 512x512 photographs of one physical run, cropped into 39 (non-independent) tiles. VisualMetro measured curves are from a DIFFERENT film (070). A generated overlay is artificial. Therefore numbers assess only synthetic injection on real background, never real particle accuracy, 3D reconstruction, mass, or human-verified labels.

## Frozen split and training

- Reuse the S37 visualMetro **whole-frame** 79/30/36 train/CAL/TEST source split, and S56 *unchanged* geometry logistic regression (`GEOMETRY_FIELDS`, class weights, scale, regularization and seed), trained on 64 S56 three-piece scenes on the SOURCE H50Z photograph with S56 source-train seed, threshold from 64 independent CAL scenes on SOURCE photograph using <=5% empirical simulated pair-negative FPR, S56 global-forest counterfactual stability margin .04. No pixel model trained on TEST.
- TEST uses 36 held-out source frames and the OPPOSITE H50Z photographic background, as in previous transfer tests. New seeds are `005 -> 006: 20266403, 20266404; 006 -> 005: 20266413, 20266414`, 24 scenes each = 96 total. SCENARIOS cycle in frozen order. No TEST-derived threshold tuning or test reruns with different detector settings.
- The image-only inference function accepts **only the 192x192 uint8 image**, an S56 geometry-only scorer fitted before TEST and CAL threshold. It outputs polylines, filtered candidate pairs, scored edges, and global abstention decisions. It must not access injected source nodes, generator curve identities, labels or oracle endpoints.
- Detector arms: (A) original S39 image-derived ridge polylines (conservative default), (B) S63 CAL-selected `junction_less` image-derived ridge polylines (experimental). Both use S59 `strict_s39` pair gate and the SAME frozen S56 graph scorer/threshold/margin. Neither is an ANZA trained relation head; label the work **machine geometry graph**, not physical-identity prediction.
- Separate oracle diagnostic: apply SAME S56 fitted scorer and global policy to the **seven generator-supplied noisy polylines** only to measure the non-image bottleneck. Never supply those oracle curves to the image-only inference function.

## Post hoc synthetic identity audit (not part of the algorithm)

- Distinct inferred lines are mapped ONE-TO-ONE to the seven generator fragments by Hungarian assignment AFTER image-only inference. An inferred polyline can match a synthetic fragment only if at least 65% of inferred samples are within 3.5 px of the fragment, 50% of synthetic samples within 4.5 px of that SINGLE inferred polyline, and BOTH corresponding endpoints are within 8 px under one orientation. If a line matches multiple similarly supported generator fragments (within .10 joint coverage) its identity is marked ambiguous, not truth. All unmatched and ambiguous proposals remain unverified.
- A suggested join is counted `correct_synthetic` only if both inferred line IDs map to a pair of different generator IDs in `((0,1),(1,2),(3,4),(4,5))` AND the chosen endpoint ports correspond (within 8 px) to the generator gap-facing ports. Every other proposal is NOT verified synthetic-correct and is split into `wrong_identified`, `unknown_unmatched`, and `wrong_endpoint` without disguising unmatched proposals as true negatives.
- Count whole synthetic 3-piece paths only if the two correct bonds occur between three distinct inferred lines and none of those lines has a conflicting suggested bond. Report detected synthetic decoys, extractor candidates, pair-gate sizes, solver failures, selected true/false/unknown joins, any-error scenes and full paths, and detection/association losses separately. Summarize by scene, not pixel, with paired **scene-bootstrap** uncertainty; there is no independent physics-run CI.
- `proxy_extracted_piece_support` from S62 is an optimistic UNION-of-detections upper bound, **not** an instance-to-instance identity match. Never conflate it with end-to-end joint recovery.

## Prespecified safety gate

The S63 experimental extraction is **not promoted** unless the held-out end-to-end test shows more synthetic correct joins/whole paths AND no increase in synthetic non-correct proposals or false-positive scenes relative to S39. Passing this one finite synthetic benchmark would NOT authorize joining real particle tracks. Every native output retains `UNKNOWN`, `physics_auto_joins_authorized=false`.

## Privacy and integrity

Only aggregate metrics, reproducible code, and test assertions may be committed to this branch. All H50Z image pixels, measured source coordinates, generated scene coordinates, predicted polylines indexed by photo, model weights and scene previews stay in private local data. No files are modified in other git branches.
