# S74 preregistration — can pixel-only unsharp masking mitigate synthetic blur?

Frozen **before S74 CAL and TEST** on 2026-10-26. Experiment ID `ANZA-LIRA-BC-S74-20261026`. The S73 study, already opened and reported, motivates the hypothesis but its TEST scenes are NEVER reused for selecting S74 parameters.

## Hypothesis and fixed arms

S73 showed a major conditional loss of image-derived correct synthetic joins after σ=1.1px Gaussian blur. Try deterministic pixel-only unsharp masking before the original S72-selected full pixel-to-graph algorithm:

1. `identity`: pass raster unchanged.
2. `unsharp_065`: gray + 0.65*(gray - GaussianBlur(gray, σ=1.1px)), output clipped uint8.
3. `unsharp_130`: gray + 1.30*(gray - GaussianBlur(gray, σ=1.1px)), output clipped uint8.

All same candidate limits, S59 pair gate, original frozen S56 logistic model, score cutoff, stability margin .04, S68 predictor implementation, and **S69 corrected orientation-invariant evaluator**. S72 source-photo CAL selected method remains unchanged: 005→006 `s68_branch_endpoint_refine`; 006→005 `s39_original`. No native particle labels enter inference. An image is not tagged as blurred for preprocessor decisions: each arm preprocesses pixels identically regardless of condition. Both clean and synthetically blurred views are evaluated independently.

## Data and frozen experiments

- Source visualMetro film 070 145 complete frames, same audited 79/30/36 whole-frame split. TEST geometry only from 36 TEST frames; CAL geometry only from 30 CAL frames. Images are artificial 3-fragment traces over genuine H50Z private backgrounds 005 and 006 from the SAME physical run, **not authentic particle labels**.
- Train/threshold frozen original S56 source-photo scorer using its preexisting TRAIN/CAL seeds. Check byte-identical SHA256 source code locks from S72 and **frozen S72 CAL SHA** `1ed7e58667aaab1a7d5ba91252db64b38bf8d3da517396b9e18e9e6bb484a952` before any scenes.
- S74 source-photo CAL seeds: 005 `20267401`, 006 `20267411`, **24** synthetic scenes per direction. Each scene evaluated under `clean` and `blur_sigma_11`; apply all three preprocessors to each condition's EXACT SAME pixels.
- Fresh S74 opposite-photo TEST seeds: 005→006 `20267403`, `20267404`; 006→005 `20267413`, `20267414`, **24 scenes each**, 96 scenes overall. No S64–S73 scene RNG reused.

## CAL-only gate, frozen before observing TEST

For a method to qualify vs `identity`, require on CAL in *same source-photo direction*:

- on blurred views: at least **2 additional correct synthetic selected joins**, no more synthetic-noncorrect selected joins, no fewer full synthetic three-piece paths;
- on clean views: at most **2 fewer correct joins**, at most **2 more noncorrect joins**, no more than **1 fewer full path**;
- zero MILP solver failures in both conditions.

Among qualifiers choose maximum blurred synthetic-correct join count, then fewest blurred noncorrect joins, then less clean correct loss, then the milder method. If none qualify, freeze `identity`. Test the CAL-selected arm against unchanged identity on new TEST and publish all arms for research transparency. **Do not** select based on S74 test outcomes.

## Evaluation/limitations

For each direction and condition report correct synthetic selected joins, noncorrect selected joins (including unmatched/incorrect ports), complete synthetic paths, scenes with noncorrect joins, candidate segments, solver failures. Include 1200 paired scene-bootstrap samples for chosen-vs-identity and blurred-vs-clean, resampling entire synthetic scenes. Repeated stress conditions from a single scene are *not independent physical observations*. This is not validation on independent cameras, authentic bubbles, physical particle identities, 3D reconstructed tracks or mass. No automatic promotion to native H50Z is authorized by this test, even if it passes synthetic CAL and TEST constraints.

Only public text source, frozen protocol, ID+SHA hashes, code unit tests, and aggregated metrics may enter Git. Private film images, coordinates, specific frame identities, training weights or per-scene predictions remain local.
