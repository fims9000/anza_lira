# S65 — CAL-only image-endpoint abstention: negative practical trade-off

**2026-10-26 · `research/bubble-chamber-graph-lira` · no native physics ground truth.**

## Question and frozen protocol

S64 end-to-end path recovery found very few synthetic-correct connections from image-derived fragments, with many mismatched gap ports. S65 adds a strictly **pixel-only terminal darkness ratio** on the original conservative S39->S56 output, with fixed point probes 2–6 px outside vs inside each endpoint. An image-derived terminal whose outward direction stays dark is deemed more likely truncated (this is *not* a physical event label). The guard can only **remove** existing advisory edges; it cannot make new joins.

S65 preregistration `S65_FROZEN_PIXEL_ENDPOINT_ABSTENTION_2026_10_26.md` was saved before running the 32-scene-per-direction CAL and before opening the new 96 TEST scenes. A CAL-only threshold grid 0.25–1.25 was frozen, with empirical minimum reduction in synthetic-noncorrect joins and minimum retained correct links. Source geometry scorer, negative-FPR calibration, margin and strict graph generator remained unchanged from S56/S64. Source photos H50Z 005/006 from **one** run; curve geometry only from other film 070. Whole-frame 79/30/36 split, no test labels used for training or selecting the S65 threshold.

### CAL decisions, 64 synthetic scenes

| Train/CAL photo | Frozen selected terminal-ratio limit | Unfiltered CAL correct/noncorrect | CAL-selected correct/noncorrect |
|---|---:|---:|---:|
| 005 | **0.70** | 3 / 29 | 2 / 11 |
| 006 | **0.40** | 2 / 16 | 2 / 11 |

The saved CAL decision file SHA-256 is `b2c36d53e30ac9b03f540d603fe4ef0bd18aa57b7492171b0b48ec9d10637c93`, and the TEST output records this exact digest. A first attempted TEST invocation was stopped by a **case mismatch in the validation code** (`CAL` vs saved `cal`), before any TEST scenes were examined. The loader was corrected; CAL thresholds and frozen CAL artifact were not changed. This is a software fix, not a retuning.

### NEW untouched TEST seeds, 96 synthetic scenes

| Metric | Original S64 pixel-only proposals | S65 CAL-selected endpoint guard |
|---|---:|---:|
| Selected advisory links | 103 | 45 |
| Synthetic-correct links | **15** | 6 |
| Noncorrect w.r.t. artificial proxy | 88 | **39** |
| Scenes containing >=1 noncorrect proposal | 53 | 33 |
| Complete synthetic three-piece paths | 1 | 0 |
| Proposals abstained | 0 | 58 |

Fresh TEST seeds `20266503`, `20266504`, `20266513`, `20266514` (24 scenes each), unrelated to S64 TEST. Pixel background transfers both directions. Scene-level paired bootstrap per seed lives in `S65_PUBLIC_LOCKED_TEST_2026_10_26.json`; conditional uncertainty does not generalize across physical camera runs. The signal reduces noncorrect proposals by 49/88 = **55.7%** but loses 9/15 = **60%** of correct joins, including the sole complete synthetic path. The surviving 39 noncorrect proposals mean the guarded output is **not safe for physical automatic joins**.

## Verdict

**DO NOT PROMOTE the endpoint guard to primary 2D tracking.** At best it is an advisory risk indicator. The substantial decline in synthetic recall and whole-path reconstruction contradicts the primary project goal. S64 original S39 stays the conservative default; actual H50Z particle identities remain `UNKNOWN`, with zero certified native GT.

Do not conflate 15/103 or 6/45 synthetic correspondence with precision on real particles. This is a different unseen synthetic TEST from S64, and comparing values across S64 and S65 as though paired data would be invalid.

## Reproducibility

```bash
python -m pytest -q tests
python scripts/research/bubble_chamber/s65_calibrate_endpoint_abstention.py cal \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/source_39_H50Z_crops \
  --out /PRIVATE/S65_FROZEN_CAL.json
python scripts/research/bubble_chamber/s65_calibrate_endpoint_abstention.py test \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/source_39_H50Z_crops \
  --calibration /PRIVATE/S65_FROZEN_CAL.json --out /PRIVATE/S65_FRESH_TEST.json
```

All public outputs aggregate numbers only, no native photos, pixel coordinates, weights or per-scene private labels. No manual marking was needed.
