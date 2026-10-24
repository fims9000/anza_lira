# Real input audit — H50Z photo/keypoints vs film 070 visualMetro measurements

Date: 2026-10-24. Scope: `research/bubble-chamber-graph-lira` only.

## Exact supplied data

**H50Z_030_01_006.7z** (film scans) has exactly these three members:

| Member | Bytes | Dimensions | Semantics |
|---|---:|---:|---|
| `H50Z_030_01_005.bmp` | 235883098 | 9020 x 26151 | full-resolution grayscale 8-bit BMP |
| `H50Z_030_01_006.bmp` | 234703910 | 9268 x 25324 | full-resolution grayscale 8-bit BMP |
| `labels.json` | 348015 | n/a | marked **fiducials** (`f1`...`f8`) and **start** points, not track masks |

Keypoint metadata lists **105** BMP names, totaling **1407** keypoints (570 `start`, remainder fiducials). **Only two of those BMPs were supplied**: image `005` has 13 keypoints (6 starts, 7 fiducials), `006` has 11 (3 starts, 8 fiducials). Do not treat 103 missing images as available.

Hashes for exact files:

- `H50Z_030_01_005.bmp`: `e78a90e72b80e4b166be2c0b2a53a9e70558c265c2a932fb1efc52783d5352e6`
- `H50Z_030_01_006.bmp`: `d1251c30b460e31f9b8b3e8fb4b9849b920d5210dcf94eb133fd814b1d8cbea3`
- `labels.json`: `df3fd9034d44ea4c27e79c50b5bc3c2de873071403740ab2c8d9f62004165ad7`

**070_008.7z** is a *different namespace* and contains manually measured visualMetro outputs for film 070, frames 004...010: seven frames, 42 identified tracks, 620 measured 3-D points and 1860 per-projection 2-D points plus fiducials. It contains **no** BMP images. Frame `070_008` itself includes 3 track identities with 38 3-D measured points. Nothing in the provided metadata proves that H50Z photos and film-070 track files refer to the same event. **Never join by coordinate proximity or label name.**

## Immediate usable data

The local H50Z pilot exports 11 **lossless native-resolution** 1024x1024 PNG tiles: nine point-of-interest `start` crops and two `f4` controls, with exact source image SHA-256 and crop translation. The ZIP is about **6.3 MB**, versus more than 470 MB for the two expanded original BMPs. These are **keypoint supervision only**: no entire-track ground-truth masks, no identified separate track instances, no background-negative labels.

Generate on a machine with extracted H50Z BMPs (and the repository checked out):

```bash
python scripts/research/bubble_chamber/export_small_bmp_tiles.py h50z \
  --image-dir /path/to/H50Z_images \
  --labels /path/to/labels.json \
  --out-dir /path/to/h50z_small_1024 --tile-size 1024
```

## Highest-priority data request for true image segmentation

Obtain **matching images from film 070**, preferably the **three independent BMP views of frame `070_008`**, *not more H50Z frames without pixel-level track labels*. `070_008_e1t01.txt` already supplies corresponding measured points in view order **1,3,2** (three groups of 15 observations). Crop all three views around the same indexed 3-D measurement (indices `0`, `7`, `14`, initially), keeping **original pixels** and origin `(x0,y0)`. This gives nine small PNGs (3 views x 3 points) with point correspondence and track ID.

```bash
python scripts/research/bubble_chamber/export_small_bmp_tiles.py visualmetro \
  --track-file /path/to/070_008/070_008_e1t01.txt \
  --view1 /exact/source/image-for-view1.bmp \
  --view2 /exact/source/image-for-view2.bmp \
  --view3 /exact/source/image-for-view3.bmp \
  --indices 0,7,14 --tile-size 1024 \
  --out-dir /path/to/070_008_small
```

This produces PNGs, `manifest.json` and a ZIP. File names for the three views must be provided explicitly; the program **does not guess** them. Verify the photos represent **the exact annotated frame and view** and supply view/run metadata (optical geometry, calibration, magnetic map), since pixel-bound checks alone cannot verify correspondence.

Even after the matched pixels arrive, clicks are **sparse positive landmarks, NOT full line masks**. To evaluate pixel-level segmentation, additionally request several carefully checked **dense masks / contours / polyline instance traces** on selected 1024x1024 image tiles, recording uncertain areas separately from true background. Start with e.g. 10–20 expert-checked tiles containing weak segments, gaps, crossings, stars and false-film scratches. Selection must cover more than one physical film/run before an unbiased holdout. Do not interpret unlabeled pixels as negatives.

## Model comparison protocol

Keep source pixels and 2-D/3-D correspondence intact. Train/validation/test partition by film/run, never split crops from the same physical film. Compare with fixed splits and preprocessing:

- Classical: adaptive contrast / Steger-Frangi ridge filter.
- CNN: capacity-accounted U-Net / attention U-Net.
- ANZA: same 2-D task with AZConv geometry, matched training budget and a generic anisotropic control.
- LIRA: predict source-to-destination trace-fragment relation, NONE/abstain and geometric or stereo veto; false physical track merges are more harmful than missed candidate continuations.

Metrics: centerline distance, precision/recall/clDice on *verified* masks, per-event false-bridge and wrong-track rates, disagreement near stars/intersections, runtime and operator review time. Do not claim ANZA or LIRA beats controls from H50Z keypoints alone or from same-film interpolation.

## Current reproducible truth

- `read_measurement_tree` verified all 42 film-070 named tracks and corresponding view indices; three-view sparse correspondences can be exported.
- `evaluate_geometry` computes descriptive withheld-coordinate interpolation error on film 070 only (not independent test, not image segmentation): mean px linear 39.7503, quadratic 36.0323, Hermite 36.2733; 126 track-view observations all from one film.
- Existing P0/P1/P2/S0/S1 synthetic model gains **do not** transfer into a real bubble-chamber accuracy claim.
- The two H50Z photos and all derived sample PNG tiles are **not committed** into the repository; confirm redistribution permission first.
