# S63 — CAL-selected pixel-only polyline extraction on new synthetic TEST

**26 Oct 2026 · research/bubble-chamber-graph-lira · 48 CAL scenes + 96 new TEST scenes.** No native H50Z physical track GT; no manual labels; no change to original AZUNet/U-Net, the S56/S60 graph policy, or S62 scores.

## Question and leak guard

S62 showed that the existing image-to-polylines extractor left only 73/384 artificial joins supportable and 19/192 entire synthetic three-piece paths. To improve this bottleneck without simulator truth leakage, four pixel-only algorithms were frozen **before** CAL and TEST in `S63_FROZEN_CAL_TO_NEW_TEST_PIXEL_CURVES_2026_10_26.md`. Each receives only a 192x192 pixel image and returns <=32 image-derived polylines. Generator curves are not passed to the extractor; they are used only for tolerant proxy evaluation after predictions. The control scene geometry came from visualMetro film 070; real photo backgrounds are H50Z 005/006 from one physical run.

CAL templates from 30 different visualMetro source frames; TEST from 36 source frames. Scene seeds CAL 20266301/02/11/12 (12 scenes each), TEST 20266303/04/13/14 (24 scenes each). All are different from S62 scene seeds. CAL selection required >=max(5,15% original true-join count) gain and no more than 2 additional synthetic decoy detections; no TEST feedback. Frozen selection: `junction_less`, allowing fewer skeleton junction cuts; compare with identical original S39 `extract_ridges`, darkness-only and ridge+darkness fusion. Test was not opened until CAL selection JSON was saved and subsequently cryptographically identified in TEST provenance.

## CAL aggregates, 48 scenes

| Metric | Original S39 | Junction-less selected |
|---|---:|---:|
| Artificial track fragments supported | 91 | 100 |
| Artificial correct join endpoints both supported | 32 | 37 |
| Full three-piece artificial paths supported | 8 | 10 |
| Artificial decoy fragments supported | 14 | 14 |
| Extracted candidate polylines total | 248 | 249 |

## NEW TEST aggregates, 96 scenes

| Metric | Original S39 | CAL-selected junction-less | Other candidate arm: darkness-only | Fusion |
|---|---:|---:|---:|---:|
| Artificial correct track fragments supported /576 | 198 | **208** | 163 | 203 |
| Correct joins conditionally supportable /384 | 76 | **85** | 55 | 80 |
| Entire three-piece paths conditionally supportable /192 | 22 | **28** | 15 | 25 |
| Artificial decoy fragments supported /96 | **28** | 34 | 28 | 34 |
| Image-derived candidate polylines (total across 96) | 447 | 455 | 375 | 437 |

Scene-paired bootstrap for the extra potentially linkable artificial true joins: +0.0104 to +0.1979 joins per synthetic scene (95% scene-bootstrap interval); by itself this is not a physical confidence interval, and each photo bank represents the same H50Z physical run.

## Gate verdict

**NO AUTOMATIC PROMOTION.** The CAL-selected variant returns **+9** potentially linkable correct artificial joins and **+6** fully supportable artificial paths but also **+6 artificial decoys** on untouched TEST. The predeclared CAL safety condition did not generalize to TEST, just as S60's CAL-only risk limit did not guarantee no false links. Moreover this metric is only a *fragment-availability proxy*: candidate polylines might belong to other overlapping strokes and real graph assembly may still fail. There is no measurement of confirmed H50Z physical particles or native false join rate.

The old S39 conservatively remains default. All candidate variants remain available for controlled investigation, without deleting lines or asserting physical joins. Future experiments should improve discriminability at crossings and link the pixel-only candidates directly into a full graph, always reporting decoy burden and scene-level wrong joins. No user or expert manual annotation is a prerequisite for that engineering stage.

## Reproduction

```bash
python -m pytest -q tests
python scripts/research/bubble_chamber/s63_calibrate_then_new_test_pixel_curves.py \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/source_39_H50Z_crops \
  --phase cal --out /PRIVATE/S63_FROZEN_CAL_PIXEL_CURVES_PUBLIC.json
python scripts/research/bubble_chamber/s63_calibrate_then_new_test_pixel_curves.py \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/source_39_H50Z_crops \
  --phase test --calibration /PRIVATE/S63_FROZEN_CAL_PIXEL_CURVES_PUBLIC.json \
  --out /PRIVATE/S63_FROZEN_TEST_PIXEL_CURVES_PUBLIC.json
```

Actual CAL and TEST outputs (aggregated public-only) are archived in `docs/bubble_chamber/`. The private viewer contains original 39 H50Z crops and four S62 injected-scene visual checks but these cannot enter public source/GitHub.
