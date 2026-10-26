# S83 | Preregistered non-saturating local pixel-line support and continuity

**Date 2026-10-26. Frozen BEFORE S83 CAL or TEST. ID `ANZA-LIRA-BC-S83-20261026`.** The observed saturation of S82's 25th/50th source-CAL quantiles at 1.0 motivates the next question, but no S82 TEST measurements are used for S83 selection, seeds, thresholds or model training. All TEST truth is exclusive to post-hoc assessment.

## Source and controls

Original S72 source-photo direction-selected graph policy, S56 geometry logistic model and CAL threshold, S59 strict candidate gate, S69 corrected orientation-invariant evaluator, S76 shape/endpoint diagnostic, S80 two-scale consensus extra instance generator and S37 film-070 whole-frame split 79 TRAIN/30 CAL/36 TEST all remain byte-identical. S82's arclength resampling of line-local 31 points is reused as immutable geometry code; the **new per-line feature is different**, and its code hash is locked in the CAL artifact. Only *extra* S80 polylines can be gated. The original S72 lines are always preserved as identical Python objects; no auto physics joins.

## Non-saturating feature (pixel-only, no GT)

At 31 approximately equally arclength-spaced, non-endpoint positions sample original normalized (0..1) uint8 image brightness on the candidate centerline and sidebands at normal offsets ±3 and ±5 px, using bilinear pixel sampling. Out-of-image candidate support fails closed. Dark stroke strength `s_i = max(0, mean(sidebands_i) - gray(center_i))` (NO clipping at a small intensity value). The *spatial consistency* factor is `q25(s) / (q75(s) + 0.005)`, in [0,1]. Final score:

`score = mean(s) * (0.35 + 0.65 * consistency)`.

No artificial 0.045 clipping as in S82, so multiple high-contrast lines can have different scores. Sampling both symmetric sides and using arclength means the score is unchanged under reversing source vertex order. A line has no class label or physical identity; scoring works only from pixels. For source CAL, pool scores on all S80 consensus extra curves across **clean and Gaussian-blur σ=1.1** SOURCE photo scenes (no GT), obtain 40th and 70th percentiles per SOURCE direction, and freeze them as numeric thresholds. If no source CAL extra curves exist, both thresholds are `1.0` (fail-closed). Two S83 arms:

- `baseline`: unmodified S72, no extra instances.
- `s83_local_q40`: add S80 extra lines with non-saturating score >= SOURCE-CAL 40th percentile.
- `s83_local_q70`: same with SOURCE-CAL 70th percentile (more conservative).

Both preserve original S72 polylines, S80 duplicate suppression, <=32 total lines, S59 strict pair gate, S56 global-MILP solver and 0.04 margin. Never use synthetic track nodes, source film IDs or label masks for inference.

## Frozen data and CAL/TEST

Film 070 audited 145 source frames and 79/30/36 whole-frame split, SHA-256 `448f21ca3a709010ed972a596ed046cadff0643112905fd1d029c91f8b14faef`. Original H50Z 005/006 are **two photographs from a single physical run**, 21+18 private 512px crops. VisualMetro 070 geometry is NOT H50Z physical truth. Only 192px synthetic artificial insertions have known labels, and are assessed against generator truth **after** blind pixel inference.

- SOURCE CAL: source 005 seed `20268301`, source 006 seed `20268311`, **20 new scenes/direction** (40 total), clean + blur σ=1.1 paired. Source CAL only for numeric per-line q40/q70 and synthetic safety selection.
- HELD-OUT TEST: source 005→006 seeds `20268303`, `20268304`; source 006→005 `20268313`, `20268314`, **24 new scenes per seed, 96 scenes/condition**, clean + blur σ=1.1 + blur σ=1.6. No TEST data used to alter thresholds or CAL-selected method. No reuse of S82 seed/sample.

## Preregistered CAL-only promotion conditions

To qualify, an arm must on CAL blur1.1 improve selected synthetic-correct links >= baseline+2, increase synthetic-noncorrect joins by at most 0, produce at least as many full artificial three-piece paths, and improve shape-matched true fragments >= baseline+2. On clean CAL it must not increase noncorrect joins, lose more than 1 correct join or lose a full artificial path. All solver calls must succeed. If both pass choose highest blur-correct links, then lowest blur-noncorrect, then fewer additional instances, then `s83_local_q70`. Otherwise `baseline` remains default. **TEST cannot change CAL choice**, even if post-hoc results look better. Never authorize native H50Z automatic physical joins; no native physics labels exist.

Record all three arms on TEST, per seed/direction/condition scene-paired bootstrap 95% intervals (1200 replicates, not pixel-based), synthetic S69 graph correctness, noncorrect proposals, S76 shape/endpoint diagnostics, extra line counts, solver failures and unknown-native summary. Persist immutable public aggregate-only JSON plus study ID/protocol/code/source/CAL hashes, no photographs, raw measured point arrays or model weights in Git.
