# S82 | Frozen local pixel-support gate for additional multiscale 2D line instances

**2026-10-26, before S82 calibration and held-out TEST.** Branch: `research/bubble-chamber-graph-lira`. ID: `ANZA-LIRA-BC-S82-20261026`. Main S72 algorithm is NOT replaced.

## Question

S80 consensus multiscale candidates add synthetic true links on blurred photographs, but also can add unsupported links on sharp photographs. S81's *whole-image* fine/coarse gate misfires on clean backgrounds. Does evaluating dark-stroke support **along each proposed extra line** improve the complete pixel-to-graph trade-off, without dropping original S72 lines?

## Frozen pixel-only arms / calculations

All variants start from precisely the unchanged S72 per-direction graph arm, then the S80 `s80_consensus_add6` candidate generator. Original source polylines are retained *as identical objects* in their original order. Only the up-to-six **extra** S80-consensus polylines may be selected for additional graph proposals; no raw object is altered or relabeled. Competing hypotheses remain UNKNOWN; never assert physical 2D or 3D particle identity.

Line-local support (pixel-only) is sampled at **31 approximately arclength-uniform positions** along a 2D line, excluding both endpoint samples. Directional tangent is a centered difference; both normal sides are averaged at offsets **3 and 5 pixels**. Grayscale normalized to [0,1], sampled bilinearly. At position `s` define `d(s) = mean(gray(s+normal*{3,5}), gray(s-normal*{3,5})) - gray(s)` (positive for a dark, locally thin streak on brighter sides). Score `0.55 * mean(d>0.008) + 0.45 * mean(clamp(d / 0.045,0,1))`; no label, no GT, no track ID, no optical flow, no scene ID, and no pixel outside 192x192. Reject curves whose sample positions or sidebands fall outside the image. Score must be finite and in [0,1]. Reversing a line must return exactly the same score within numeric tolerance.

- `baseline`: original S72, no extra lines.
- `s82_local_q25`: append an S80 consensus extra **only if** its pixel-only score is >= source-photo CAL 25th percentile of the pooled extra-line scores from clean+blur(σ=1.1) synthetic-CAL **pixels**.
- `s82_local_q50`: same at source-photo CAL 50th percentile (stricter).

The two numeric thresholds are calibrated **per source direction** from S82 CAL **pixel-only extra lines**, no generator annotations or correct-link labels. If no extras exist for source CAL, both thresholds are 1.0 (fail-closed). Equality qualifies. Duplicate suppression and base-line cap stay those of S80. Recompute original S59 `strict_s39` gate, original S56 logistic score, original 0.04 margin. All added candidates remain advisory and suppress none of the source instances.

## Frozen source split, scenario IDs, independent held-out scenes

The measured geometric templates are visualMetro **film 070**, not truth for actual H50Z. 145 audited frames are split by whole frame 79 TRAIN / 30 CAL / 36 TEST (SHA `448f21ca3a709010ed972a596ed046cadff0643112905fd1d029c91f8b14faef`). H50Z real 512x512 crop backgrounds are 21 from photo `005` and 18 from photo `006`, from one physical run, not independent camera-run or physics validation. Each 192x192 generated scene has two artificial three-piece tracks and an artificial decoy.

Original frozen S56 TRAIN/CAL geometry scorer and score threshold are fitted on SOURCE photo; original S72 CAL-selected base graph arm is immutable (`S72_PUBLIC_FROZEN_CAL_2026_10_26.json` and S72 code locks). S69 orientation-invariant correctness evaluator and S76 endpoint-vs-shape diagnostics are unchanged.

- S82 SOURCE CAL: source photo 005 seed **20268201**, source photo 006 seed **20268211**; 20 synthetic scenes per direction, conditions `clean`, `blur_sigma_11`. First pass collects **only pixel-derived extra-line support scores** and fixes q25/q50; second pass measures CAL arm outcomes with generator labels **only after** all blind inference is complete.
- S82 NEW held-out TEST: source 005 -> target 006 seeds **20268203, 20268204**, source 006 -> target 005 seeds **20268213, 20268214**, 24 scenes each (96 TOTAL scenes), conditions `clean`, `blur_sigma_11`, `blur_sigma_16`. Seeds are fresh, disjoint from S77-S81. No TEST thresholds/arms can be tuned after TEST.

## Frozen safety gate (SOURCE CAL only)

Select a local gate only if on CAL blur σ=1.1: synthetic-correct graph links **>= baseline+2**, synthetic noncorrect proposals **<= baseline**, full three-piece synthetic paths **>= baseline**, shape-matched artificial fragments **>= baseline+2**; and on clean source-CAL: synthetic-correct links **>= baseline-1**, synthetic noncorrect proposals **<= baseline**, complete artificial paths **>= baseline**; solver successes required in both. If both pass, choose greatest CAL blurred correct links, then fewest CAL blurred noncorrect, then fewer extra lines, then q50 (conservative). If neither passes, `baseline` is frozen. Held-out test reports **all arms** but cannot change this choice. 1200 paired bootstrap draws by scene per seed, NOT by pixels or fragments. In no case authorize automatic joins of physical H50Z tracks.

## Measurables and integrity

Number of source vs extra candidates, accepted/rejected extras, candidate pair count, synthetic shape-matched fragments, endpoints >8px from artificial source, S69-correct selected joins, noncorrect selected joins, incorrect-proposal scenes, complete synthetic three-piece paths, solver failures, 95% scene-paired bootstrap deltas. Record protocol SHA-256, predictor SHA-256, S80 SHA, S72 CAL SHA, source split SHA, frozen CAL bytes SHA, TEST seed sets, run/experiment ID in public JSON. No raw H50Z pixels, source measured coordinates, per-scene point clouds, trained weights, or native physical identifiers in Git.
