# S68 — frozen branch-preserving pixel-to-graph protocol

**Frozen before S68 CAL or TEST; date 2026-10-26.** Scope: machine-only 2-D fragment hypotheses; never certified bubble-chamber particle identity. Existing S39/S67 results motivate this test but provide no tuning information for new TEST seeds.

## Question / irreversible controls

Does separating connected skeleton branches based on topological 8-neighbor arm transitions improve *injected synthetic* correct links/full three-piece paths versus S39 without increasing noncorrect global links? Is bounded (<2.5 px) endpoint continuation towards the same image-only junction helpful? S68 never merges components across crossings and never relabels native H50Z as physical tracks.

Four fixed arms:
1. `s39_original`: unchanged ridge extraction.
2. `s67_crossing_orientation_v1`: unchanged S67 junction-stitching comparator.
3. `s68_branch_split`: identical S39 Hessian ridge and activation budget, 3x3 closing, 8-neighbor Moore arm-count >=3 defines branch cores, remove 3x3 core, extract bounded individual components; no crossing joins.
4. `s68_branch_endpoint_refine`: same as arm 3, add optional <=2.5-px extension toward detected junction if end tangent alignment >0.8, junction distance 0.7–4 px and nonzero score support; independent arms are NOT fused.

Fixed extraction limits: 192x192 uint8 input; 32 candidates max; same 3.5% pixel activation budget; component >=18 skeleton pixels and >=17px span. Common unchanged S56 geometry-only logistic scorer, S56 5% CAL negative-pair cutoff, `strict_s39` gate, global MILP stability margin 0.04; no trained segmentation modifications.

## Private sources and split

Film-070 visualMetro measurement geometry: 145 valid complete frames, whole-frame 79 train / 30 calibration / 36 heldout; this is **not** H50Z photo-specific truth. Real H50Z photo backgrounds: 39 private 512px tiles, 21 from image 005 and 18 from 006, all from ONE physical run; injected tracks/scratches and their identities are synthetic. Training source-photo versus testing other-photo, not independent film holdout. Do not publish pixel data, image crop metadata, frame IDs or point coordinates.

S56 pre-fitted scoring lineage: unchanged original S56 train and CAL scenes/seeds (not refitted using S68 TEST).

**S68 new CAL seeds** 005=20266801, 006=20266811; 16 scenes per direction, source-photo tiles and CAL-source geometry only. Scenarios cycle `parallel,near_parallel,crossing,scratched,empty_gap`.

**S68 new HELDOUT TEST seeds** 005->006 `(20266803,20266804)`, 006->005 `(20266813,20266814)`; 24 scenes each, 96 scenes, 384 correct synthetic links, 192 complete chains potential. Frozen test generator is unchanged S56 three-piece scene. TEST used once, no threshold retuning.

CAL selection: promote an S68 arm to *synthetic experimental* primary ONLY if +2 or more CAL correct joins, zero increases in incorrect joins and scene-level any-incorrect count and no decrease in complete three-piece paths relative to S39. Prefer fewer incorrect then more correct. Otherwise keep S39. S67 is only a reference, not eligible for promotion. TEST evaluates all arms for transparent negative controls; no post-TEST tuning or implicit replacement. TEST deployment gate requires nondecreasing correctly joined links/full paths and no additional wrong links/scene-level risks; otherwise reject.

Paired scene bootstrap: 1200 resamples per direction/test seed, grouped by complete simulated scene, not pixels, crops or tracked clicks. Report correct joins, noncorrect proposals, any-noncorrect scenes, full chains, and model failures; only aggregate outcomes are public.

All outputs record schema, code/protocol/S56 scorer SHA-256, frozen frame split digest, CAL digest, CAL-selected arm, seeds, candidate/solver counts. Never store native source frame identifiers, imaging pixels, private trajectories, models or reviewed annotations in public Git. Physical auto-joins remain disabled. Report unknown native correctness. Strictly reject changed CAL protocol/predictor/S56 source hash/split digest during TEST.


## Pre-result code correction A (2026-10-26)

The first held-out TEST command failed before emitting any metrics: NumPy array comparison when two junction centers were equidistant inside optional endpoint refinement. After freezing/calibrating initial source SHA, change ONLY deterministic tie sorting using `(distance, center_index)`; add an exact equidistance unit regression. Retain original CAL artifact privately as **obsolete**; rerun complete CAL after correcting code and re-lock new predictor/protocol SHA. None of the withheld TEST aggregate scores were seen at correction time. The originally frozen TEST seed IDs and selection gates remain unchanged. No hyperparameter tuning from TEST.
