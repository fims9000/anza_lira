# S48 preregistered two-image jackknife registration stress audit

Status before experiment: FROZEN. Work exclusively in research/bubble-chamber-graph-lira.

Scientific question: How do different plausible fiducial registrations affect S45
same-coordinate image-repeat warnings on original machine proposals? This does
NOT identify scratches or real physical particle tracks.

Inputs: private full BMP 005/006 and source checksums, six common f1-f6 marks,
S39 frozen machine candidates, S44 frozen original ridge controls, S40 fiducial
exclusions, S39/S40 native graph proposals. No new manual labels.

Procedure:
1. Fit the original restricted similarity model using all six fiducials.
2. Fit six restricted similarity models with one original fiducial omitted each
   time. Evaluate predicted held-out landmark distances. Never use H50Z pixel
   repeat results to choose model type, fit parameters or thresholds.
3. For every S39 proposed line, transform its full-image coordinates through
   each omitted-fiducial model. Map the resulting destination displacement into
   the source-grid coordinates. Keep the S44 source and displaced controls fixed.
4. Sample cross-photo ridge coverage on six predicted path positions. Keep S45
   thresholds frozen: support >= 0.70 in both views and counterpart advantage
   >= 0.20 over the displaced negative control. Known fiducial proximity
   suppresses the extra repeat cue and retains the separate S40 risk.
5. Categorize warnings: original S45 flag; stable across at least 4 of 6
   alternative registrations; unstable; newly possible only after a shift; no
   strong repeat evidence; incomplete near image boundary. All states are
   advisory, not truth. No real native track link is executed.
6. Annotate graph risks without deleting candidates or converting unknown
   labels into positive/negative labels. Private per-candidate image positions,
   transforms and overlays remain outside the public repository.
7. Separate synthetic controls use new RNG seed 20261048 on native crop
   backgrounds: source-only stroke, exact duplicate, duplicate offset by
   4 pixels along normal, duplicate offset by 8 pixels along normal.
   These are not new independent physical runs. Include a conceptual
   counterfactual where the same 4-pixel duplicate is a distinct nearby
   physical line: its pixels are identical and cannot be distinguished from
   a registration error by any pixel-only method.
8. Report every negative or ambiguous outcome, candidate and graph counts,
   input SHA verification, and exact training restrictions. No human GT.

Acceptance: tests validate registration, jackknife positions, fixed thresholds,
private provenance and graph consistency; real 39-crop run succeeds; synthetic
control demonstrates both utility and limitations; end-to-end rerun and archive
integrity checks succeed. Do not claim native accuracy or label separation.
