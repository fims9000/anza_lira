# S81 — quality-gated S80 multiscale candidate activation (frozen before CAL/TEST)

**26 October 2026** · `research/bubble-chamber-graph-lira` · study ID `ANZA-LIRA-BC-S81-20261026`.

The S80 **TEST** has already been opened; it motivates this follow-up but provides **no threshold tuning, training scenes or seed reuse**. This S81 protocol and all quality-gating rules were frozen before any S81 CAL/TEST outcomes were examined.

## Purpose and bounds

S80 two-scale consensus extracted more synthetic artificial track pieces in blurred source photos but added extra incorrect graph edges on clean photographs, and therefore failed its preregistered CAL gate. S81 asks whether an *image-only quality indicator* can activate S80's consensus instance proposals only when the photograph is visibly low in fine spatial frequencies, avoiding clean-image damage.

No expert-labeled physical H50Z particle ground truth exists (native GT count **0**); the 39 private H50Z tiles originate from 005/006 of the SAME physical run and visualMetro film 070 source geometry is unrelated to those photographed physical events. All tracked success/wrong classifications are only for artificially injected lines, and native auto-physical-join authorization remains false. This experiment does not implement deblurring, introduce human marking, or assume a physics track identity.

## Frozen image-only gate and arms

Quality feature `fine_coarse_ratio(gray)` is a dimensionless, per-192x192-tile pixel statistic:

`mean(|gray - Gaussian(gray, σ=0.85)|) / (mean(|Gaussian(gray,σ=0.85) - Gaussian(gray,σ=2.6)|) + 0.0001)`

Gray is converted to float in [0,1]. Low ratio indicates relatively little high-spatial-frequency energy but is NOT a calibrated probability of blur or a physical track label. Uniform images return 0 and may be activated without yielding any line; metrics measure such cases. Thresholds are learned ONLY from clean S81 **source-CAL** photo-backed scenes: clean feature quantiles `q=0.20` and `q=0.40`, one threshold per source direction. These are specified **before** seeing CAL labels or TEST. The inference API receives only image pixels, frozen S56 scorer, cutoff, frozen graph arm, and a numerical quality threshold; no simulator geometry/labels/scene IDs.

All scene input polylines are first produced by the unchanged S72 source-photo CAL branch. Arms:
- `baseline`: original S72 with no additional candidates.
- `s81_gate_q20`: only if image quality score <= frozen S81 CAL q20 threshold, append S80 *two-scale-consensus* extra candidates (at most 6) and recompute strict-S59 candidate gate, original S56 graph scores and margin .04. Otherwise retain exact original S72 prediction.
- `s81_gate_q40`: same with CAL q40 threshold.

Never remove or alter original image polylines. Never use the S80 medium-only arm, which failed the previous CAL. No additional learning of physical identity and no physics claims.

## Frozen data protocol

Reuse the S37 visualMetro 79 TRAIN /30 CAL /36 TEST whole-frame split (SHA-256 `448f21ca3a709010ed972a596ed046cadff0643112905fd1d029c91f8b14faef`). Keep original S56 TRAIN fit, original source-CAL cutoff, original S72 direction-selected graph arm, exact S69 orientation-invariant evaluator and S76 shape/endpoint diagnostics unchanged.

- S81 CAL: source `005` seed `20268101`, source `006` seed `20268111`, **16 new artificial scenes per direction**, paired clean vs blur σ=1.1.
- S81 TEST: fresh opposite-photo backgrounds and 36 TEST-frame source geometry; `005->006` seeds `20268103/04`, `006->005` seeds `20268113/14`, **24 new scenes per seed = 96**, conditions `clean`, `blur_sigma_11`, `blur_sigma_16` paired within each synthetic scene.
- New seeds are disjoint from S75–S80. No reviewer-labeled source nodes or photo-specific detailed coordinates enter public files.
- CAL quality feature thresholds and synthetic performance gate are frozen to immutable JSON BEFORE opening TEST. TEST checks protocol/operator/source SHA and original S56 cutoff; no post-TEST switching.

## Prespecified CAL choice

For each direction, a gated arm qualifies ONLY if on CAL blur σ=1.1 its synthetic-correct selected graph joins improve by >=2, its synthetic noncorrect proposed links do not increase, its shape-matched synthetic fragments improve by >=2 and complete 3-piece paths do not decline; on CAL clean scenes its correct joins decline <=1, noncorrect joins do not increase and full paths do not decline; all solver calls succeed. If multiple eligible, choose most blurred correct joins, then fewest blurred noncorrect, then lower triggered CAL-clean scene count, then lexical name. If none, choose `baseline`. The new TEST compares ALL arms but cannot change the CAL selection or authorize native auto joins.

Record scene-level paired 1200-bootstrap confidence intervals per direction/seed. Separate quality-gate activation counts for clean, blur1.1 and blur1.6; these counts are an image-domain diagnostic, NOT evidence of particle correctness. No native physics accuracy measure exists. Privacy: never put original pixels, source point coordinates, private overlays, raw 7z or model weights into GitHub.
