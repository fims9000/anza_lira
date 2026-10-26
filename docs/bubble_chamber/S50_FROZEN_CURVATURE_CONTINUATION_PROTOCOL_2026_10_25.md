# S50 preregistration — source-disjoint controlled 2D continuation

Status: FROZEN BEFORE S50 benchmark. Active branch: `research/bubble-chamber-graph-lira` only.

## Question

Does **local osculating-curvature continuation** add independent evidence over the
S35 six-feature geometric baseline, when correct and decoy fragments have
matched sample counts? This is a constrained 2D geometry hypothesis, **not**
Belyaev's special stopping-track hadroide, a magnetic-field fit or physical PID.

## Data and locked split

- Use the 145 complete film-070 visualMetro *measured coordinate* frames, never
  claim they are photos corresponding to H50Z. Use unmodified S37 79/30/36
  SHA256 frame split, with all source template identities disjoint.
- Photographic backdrops: native H50Z 005 and 006 crop banks; train/calibrate
  on one image; test on the other, and reverse. They are **one physical run**.
- Synthetic truth belongs only to artificially injected paths. Native
  backgrounds, unlabeled bubbles, scratches, cross-photo repeats and AZUNet
  pseudo-labels NEVER serve as truth.
- Use S35 pre-existing 4 scenarios; within each scene truncate every `left`
  candidate to last 31 points and every `right` candidate to first 31 points;
  this prevents trivial decoy discrimination from 31-point versus long lines.
  These truncations are applied **only to candidate centerlines**, not to
  the rendered photographic injection. Do not alter generator or train/test
  split based on observed outcomes.
- Frozen source-independent RNG scenes: 80 train, 100 calibration and 100 test
  per seed, two independent test seeds per photo direction: 20265003/20265004
  for 005->006 and 20265013/20265014 for 006->005. Training seeds 20265001,
  20265011; calibration seeds 20265002, 20265012.

## Three fixed comparisons

1. `geometry`: S35 `GEOMETRY_FIELDS` (6), logistic regression with z-scaling.
2. `geometry+curvature`: the identical 6 plus a frozen three-feature local
   osculating model: signed-curvature reversal discrepancy times gap, predicted
   half-gap closure relative to gap, predicted mid-gap tangent mismatch. It
   fits degree-2 local polynomials to independently smoothed endpoint fragments.
3. `geometry+shuffled_curvature`: SAME numeric feature family as (2), but new
   curvature columns are shuffled across TRAIN scenes; calibration/test use
   permuted columns separately only as a *negative control*, never used as
   a model-selection criterion.

Same training scenes, fixed logistic `C=.3`, balanced class weights, fixed seed,
fixed 2% empirical negative CAL cutoff and 0.04 mutual-best margin for all.
No additional cutoffs, source-specific tuning or test-guided selection.

## Primary and safety metrics

Full heldout synthetic AUROC/AUPRC; scene-level correct links, false joins,
scenes with >=1 false link, suggestion coverage and abstentions; scenario strata
and paired scene bootstrap uncertainty on *difference of AUCs*. Bootstrap
conditional on simulated generator, **not** independent physical-trial CI.

Do not advertise any improvement unless the same direction and other direction
both support the effect, while per-scene false joins are not materially worse.
All native H50Z graph edges retain UNKNOWN physical identity and zero automatic
joining; independent real GT count stays zero. Irrespective of outcome,
negative findings are reported and model source remains unchanged.

## Scope/QA

Freeze core source before running heldout metrics; implement orientation
reversal invariance, finite-feature and fail-closed malformed-source tests.
Any native ranking, if generated, is only a reversible advisory on existing
182 edges. Keep all private pixel positions and per-scene source frame IDs
outside the public repository. Do not change original ANZA/U-Net models.
