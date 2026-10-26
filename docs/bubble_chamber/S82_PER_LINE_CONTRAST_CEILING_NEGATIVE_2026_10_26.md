# S82 — Local pixel-support score saturated, CAL rejected, fresh TEST negative

**2026-10-26 · ID `ANZA-LIRA-BC-S82-20261026` · 96 fresh artificial scenes per condition.** Original S72 not replaced; no native H50Z physical GT.

## Goal

S80's multiscale line proposals have some synthetic recall benefit under image blur, while S81's *whole-image* blur gate triggers on many clean scenes. S82 tests a local pixel-only contrast indicator for each **extra** multiscale line, while always retaining existing S72 polylines in precisely the same order, with the same object identity. Inference receives only image pixels, frozen source-CAL geometry model and numeric per-line thresholds; reference fragment IDs, endpoints and synthetic correctness are read **only by a separate post-hoc evaluator**.

The line-local feature uses 31 equally arclength-spaced image positions, compares centerline pixel values with two symmetric sidebands (at 3 and 5 px) and combines the rate of sufficiently dark centerline points and clipped normalized contrast. Its exact frozen formula is in `S82_FROZEN_PER_LINE_PIXEL_SUPPORT_2026_10_26.md`; reversing the point order does not change it. The S69 orientation-correct evaluator and S76 shape/endpoint diagnostic are unchanged.

## CAL (40 fresh artificial scenes, before TEST)

CAL seeds: source 005 `20268201`, source 006 `20268211`; 20 scenes per source, paired clean and blur σ=1.1. Quantile thresholds are based only on candidate pixel-local feature values, pooled across the two CAL conditions; artificial truth is not used to set the numerical thresholds.

| Source direction | Pixel-only extra lines measured | CAL 25th percentile | CAL median | CAL-selected method |
|---|---:|---:|---:|---|
| 005 → 006 | 15 | **1.0** | **1.0** | **S72 baseline** |
| 006 → 005 | 21 | **1.0** | **1.0** | **S72 baseline** |

**Critical negative technical finding:** both thresholds saturated at `1.0`. The normalized local contrast is nearly maximized on multiscale ridge detections by construction; q25 and q50 cannot distinguish extra candidates. Both S82 arms therefore make identical choices and do not pass the frozen calibration gate. No threshold was retuned after seeing CAL/TEST; the ceiling effect is explicitly documented rather than hidden as a successful classifier.

## Fresh untouched TEST on opposite genuine H50Z photographic backgrounds

TEST seeds `20268203/04` for 005→006 and `20268213/14` for 006→005; 24 new synthetic scenes per seed, 96 scenes per image-quality condition. Source visualMetro film-070 curves from TEST frames are not matched physical H50Z tracks; both H50Z photo banks come from **one** physical run.

| Condition | Detector | Synthetic-correct joins | Noncorrect / nonverifiable joins (synthetic reference) | Full artificial 3-part paths | Shape-matched synthetic fragments |
|---|---|---:|---:|---:|---:|
| Clean | Original S72 | 40 | **36** | 10 | 234 |
| Clean | S82 q25 or q50 | 42 | 43 | 10 | 240 |
| Blur σ=1.1 | Original S72 | 9 | **17** | 2 | 121 |
| Blur σ=1.1 | S82 q25 or q50 | 11 | 23 | 2 | 129 |
| Blur σ=1.6 | Original S72 | **5** | **19** | 1 | 108 |
| Blur σ=1.6 | S82 q25 or q50 | 5 | 21 | 1 | 114 |

Each S82 arm made identical selected graph decisions on this study's TEST. Clean: +2 synthetic-correct links but +7 noncorrect; blur1.1: +2 correct but +6 noncorrect; blur1.6: no gain in correct links and +2 noncorrect. The strict predeclared CAL promotion gate failed both directions, **so the default remains the unmodified S72**. Per-seed scene-bootstrap 95% intervals, not pixelwise intervals, are stored in the immutable public TEST JSON.

## Native real-image UNKNOWN diagnostics

On the fixed central 192x192 region of the 39 original (not artificially injected) H50Z photo crops:

| Arm | Candidate machine lines | Machine advisory joins | Additional lines | Extra S80 candidates considered / rejected |
|---|---:|---:|---:|---:|
| Original S72 | 83 | 3 | 0 | 0 / 0 |
| S82 q25 | 92 | 5 | 9 | 12 / 3 |
| S82 q50 | 92 | 5 | 9 | 12 / 3 |

Those additions and links have status **UNKNOWN**, not true real physical particle connections; no physics validation or runtime auto-join was authorized. They are reversible hypotheses, not labels.

## Engineering / scientific implications

This study isolates a *feature saturation* failure in local pixel-quality measurement. A detector already conditioned on strong dark ridge evidence yields locally maximized clipped contrast, so another such threshold cannot discriminate false vs useful instances. A future S83 study, **on new independent synthetic CAL/TEST seeds**, should use a non-saturating, spatially varying feature of line morphology/contrast continuity rather than retune these two thresholds against S82 TEST.

All private originals, raw visualMetro point lists, synthetic pixel overlays, photo-identifying source locations and model weights stay OUTSIDE Git. The repo may store code, immutable aggregate-only JSON, source/protocol hashes and opaque experiment IDs.

## Reproduction (requires separately authorized private inputs)

```bash
python scripts/research/bubble_chamber/s82_calibrate_per_line_then_test.py cal \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/source_39_H50Z_crops \
  --out /PRIVATE/S82_FROZEN_CAL.json
python scripts/research/bubble_chamber/s82_calibrate_per_line_then_test.py test \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/source_39_H50Z_crops \
  --calibration /PRIVATE/S82_FROZEN_CAL.json --out /PRIVATE/S82_FRESH_TEST.json
python scripts/research/bubble_chamber/s82_native_readonly_audit.py \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/source_39_H50Z_crops \
  --calibration /PRIVATE/S82_FROZEN_CAL.json --out /PRIVATE/S82_NATIVE_UNKNOWN.json
python -m pytest -q tests
```
