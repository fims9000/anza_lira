# S65 preregistration — CAL-only endpoint-termination guard on pixel-derived graph (2026-10-26)

**Frozen BEFORE CAL tuning and any new S65 TEST.** This is a new synthetic-only hypothesis motivated by the *previously disclosed* S64 failure, NOT a revision to the locked S64 TEST. Original S39 pixel extractor, S56 geometry scorer, S56 CAL pair cutoff, S56 MILP margin, S37 whole-frame source split remain unchanged.

## Hypothesis

S64 pixel-only path proposals disproportionately fail because the inferred curve endpoint is an artificial break in a still-visible line, rather than a true end of the injected fragment. A pixel-only endpoint-termination indicator compares local darkness along the extrapolated outgoing tangent 2–6 px BEYOND the inferred endpoint against darkness 2–6 px INSIDE the inferred line. If outwards stays dark like inside, the endpoint is suspicious; a proposed connection using it may be rejected. This is NOT physically diagnostic on genuine particle photographs: an observed track can overlap/continue across uncertain image segments.

## Locked policy

- Feature calculated ONLY from the 192x192 photographic image and image-extracted polyline endpoints; never sees generator identities or truth. Frozen outgoing distance samples at [2,3,4,5,6] px, local darkness `max(GaussianBlur(gray/255, sigma=7)-gray/255,0)`, 75th percentile, denominator floor `0.010`, and `max(ratio_endpoint_A,ratio_endpoint_B)` per selected S56 edge. It cannot create edges, only withhold them.
- Baseline `no filter` and frozen admissible ratio thresholds `[0.25,0.40,0.55,0.70,0.85,1.00,1.25]` tested on SOURCE photo calibration only. The predeclared selection rule picks the threshold giving the **fewest CAL non-correct selected links** while reducing noncorrect links by at least `max(3,ceil(0.25*baseline_noncorrect))` and retaining at least `ceil(0.60*baseline_correct)` synthetic-correct CAL links. Tie-break: most CAL-correct links, then smallest threshold. If no threshold satisfies both, stay at original S64 with no filter. Any test-derived adaptation is prohibited.
- CAL: film 070 whole-frame 30-source-frame CAL fold; H50Z photo same as training source; seeds `20266501` and `20266511`, 32 new scenes/direction. The S56 scorer is independently trained on the 79-frame TRAIN fold, 64 original S56 train scenes, and original S56 64 CAL scene cutoff; frozen before S65 CAL. Generator identities are used only for computing the CAL policy, NEVER passed to inference.
- TEST: 36-frame held-out source fold and the opposite H50Z photo background. Seeds `005 -> 006: 20266503/04`, `006 -> 005: 20266513/14`; 24 new scenes per seed, 96 TEST scenes. TEST phase MUST load an immutable saved CAL decision, check SHA and exact recomputed source scorer cutoff, and may not adjust the chosen endpoint ratio threshold after reading TEST scenes.
- Report per-arm scene-level synthetic true, noncorrect (unmatched / wrong port / wrong identity), any-error scenes, complete paths, abstained selected edges, matched source fragment count; paired scene bootstrap. No independent run-level uncertainty. Negative or degenerate results (zero output) must be reported as such, not hidden by precision.

## Explicit exclusions

No H50Z native physical ground truth; image bank belongs to one run. Neither true particle identity nor mass/3D track recovery may be claimed. All future native joins remain `UNKNOWN` and not physically authorized. The study evaluates a **subset** of baseline pixel-only S56 proposals; selection safety on synthetic CAL is not a guarantee on other data.
