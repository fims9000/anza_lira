# S77 · preregistered pixel-only short-gap instance recovery · 2026-10-26

**Status: frozen BEFORE S77 CAL and any S77 TEST.** Existing S76 TEST metrics and S75C retrospective diagnoses motivate this approach; S77 NEVER uses those TEST labels for thresholds. Historical S39/S68/S69/S72 source and evaluator code are read-only. All performance claims are synthetic-on-H50Z, not physics GT.

## Research question

When S76 shape/endpoint diagnostics show that blur removes individually recoverable fragments, can tightly constrained **pixel-derived short-gap repairs** improve end-to-end S69-correct artificial joins / full 3-piece paths without raising spurious joins or losing clean-image fragments? Different from S66, we do NOT extend curve ends without another already detected corresponding fragment. We consider only joining two separate observed line instances across a tiny visible gap; ambiguous crossings and rival continuations abstain.

## Image-only algorithm, fixed arms

Use frozen direction-specific S72 `CAL_selected` extractor on each 192x192 uint8 image, same source-trained S56 graph scorer, same S56 CAL-only 5%-negative cutoff, same strict S39 candidate-pair gate and global forest abstention margin 0.04. Existing S68/S69/S56 sources are not edited. The S77 module receives only the raster and the **already predicted polylines**, never any generated coordinates or source identities.

- `baseline`: original image-only S72 prediction untouched.
- `short_gap_strict`: paired endpoints 1.6–5.5 px apart, outward tangents aligned with joining direction cosine >= cos(18 degrees); pixel darkness sampled at 25%, 50%, 75% of gap with minimum 0.006 in grayscale/255 units and median-to-interior evidence ratio >=0.32; no third predicted line within 2.5 px of gap midpoint; unique best continuation advantage >=0.13.
- `short_gap_relaxed`: endpoints 1.6–8.5 px apart, tangents cosine >= cos(25 degrees), minimum median gap darkness >=0.004, gap/interior >=0.24, third-line exclusion radius 2.5 px and advantage >=0.13.

Each candidate merges exactly two preexisting pixel-derived curves, using a short straight pixel-backed gap and **preserves all remaining original curve hypotheses**. Never bridge if a rival endpoint direction/score is within 0.13 of the best for either endpoint; never form cycles, re-use a port, or join a component into itself. At most two merges per original piece, bounded 32 input pieces, no physical identity inferred. No additions of unobserved free-standing lines or artificial GT-based postprocessing. For graph comparison, re-score the resulting modified list with unchanged S56 scorer, unlike purely diagnostic S76 shape metrics.

## Frozen data/splits and CAL selection

- Measured geometric templates are visualMetro film 070 whole-frame S37 split **79 train / 30 CAL / 36 TEST**, unrelated to physical H50Z ground truth. Private 39 genuine H50Z 512x512 tile backgrounds are two images from **one physical run** (21 crops photo 005 / 18 photo 006).
- Use S72 frozen source scorer and direction-specific graph extractor CAL record SHA-256 `1ed7e58667aaab1a7d5ba91252db64b38bf8d3da517396b9e18e9e6bb484a952`.
- NEW S77 source-photo CAL seeds 20267701 (photo 005) and 20267711 (006); **16 scenes per source**. Evaluate `clean` and `blur_sigma_11` on exactly paired scenes. Artificial segment identity is permitted only in the CAL evaluation after all arms predict. Baseline is CAL comparison control.
- Predeclared CAL promotion eligibility for a *synthetic experimental arm only*: >=2 additional artificial correct joins under blur, **zero increase** in blur noncorrect joins, **zero increase** in clean noncorrect joins, loss of <=1 correct clean joins, >= baseline number of full artificial three-part chains under blur and clean, zero solver failures. Tie-break: more blur correct, fewer blur noncorrect, strict before relaxed. Otherwise keep baseline unchanged.
- NEW S77 independent scene TEST seeds 20267703/20267704 (005 -> 006) and 20267713/20267714 (006 -> 005), 24 scenes/seed = **96 synthetic scenes**. Conditions: `clean`, `blur_sigma_11`, `blur_sigma_16`. All 3 arms always reported even if CAL rejects; no TEST retuning. Source train/CAL photos and test photos are different still photos from same physics run, NOT independent physics-run validation.
- Report corrected S69 artificial join correctness, other proposals, complete paths, per-scene any-error, predicted curve count, S76 shape-only matched fragments, S76 shape endpoint-far and endpoint-near errors, algorithmic merge counts, and solver failures. Paired bootstrap by whole synthetic scene (1200 draws; fixed seed 20267790 + direction/condition index); no pixel-level bootstrap, no statistical claim across independent physical runs.

## Privacy and acceptance

Only publish source/tests/frozen methods/SHA-256 references, aggregate per-seed CAL/TEST metrics and `ANZA-LIRA-BC-S77-20261026` experiment ID. Never upload H50Z photos, visualMetro per-event geometric measurements, generated image previews, raw 7z, checkpoints or per-scene coordinates to GitHub. All native H50Z identity remains `UNKNOWN`, physical-auto-join authorization is **false regardless of synthetic result**. Failure to meet CAL gate is a valid negative result and must not be concealed; TEST is not for optimization.
