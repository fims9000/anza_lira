# S4: measured-track data acquisition and H50Z fiducial QA

**Date:** 2026-10-24  
**Branch:** `research/bubble-chamber-graph-lira` (no changes to Varvara branch).  
**Stage:** reproducible dataset engineering; **not** an ANZA superiority claim.

## Why these acquisitions

There are two sources, with distinct labels and **no established cross-source image alignment**:

- `070_008.7z`: 7 measured frames (`070_004` ... `070_010`), 42 track identities, 620 measured 3D points / 1,860 film pixel coordinates in three views. **No matching image files** supplied. These coordinates provide sparse *positive* clicks and inter-view/track identity, **not** dense masks or absence-of-track negatives.
- `H50Z_030_01_006.7z`: two real 8-bit source BMPs (`005`, `006`) and a separate `labels.json` covering 105 named frames (570 vertex-`start` keypoints). Only 9 of these `start` annotations correspond to supplied BMPs. More scan images exist in the label inventory but are not in this archive. Fiducial `f1`..`f8` marks are references, not particle tracks.

The first real-image/track-identity alignment should use **three original BMP views of frame `070_008`** (1, 2, 3) matched to its visualMetro files. The scanner filenames are not assumed. The next frame candidates for scene complexity are `070_004` and `070_009`, again all three camera views. All these frames share film `070`; they do *not* support cross-run independence.

If only new H50Z material is accessible, prioritize the following **metadata-confirmed** filenames for limited small crops:

| Order | Exact BMP filename | Annotated start points | Named scan group |
|---|---|---:|---|
| 1 | `H50Z_030_02_705.bmp` | 19 | `H50Z_030_02` |
| 2 | `H50Z_030_02_758.bmp` | 13 | `H50Z_030_02` |
| 3 | `H50Z_030_02_700.bmp` | 12 | `H50Z_030_02` |
| 4 | `H50Z_030_01_007.bmp` | 10 | `H50Z_030_01` |

Scan groups are identifiers based on filename only, **not certified distinct experimental media/runs**. The `007` keypoints include a duplicate `f5` requiring manual review; do not train fiducial localization from both without resolution.

## Fiducial audit

A deterministic RANSAC 2D affine agreement screen was applied separately within the two filename groups. This is *not* replacement for full optical camera calibration and is not enough to reconstruct momentum.

- **105 frame records**, **570** annotated `start` keypoints in source metadata.
- **2** records contain duplicated fiducial identifiers: `H50Z_030_01_007` (`f5`), `H50Z_030_02_747` (`f4`).
- **4** records have at least one outlying fiducial correspondence under the 60-px QA screen: `_01_005`, `_01_007`, `_02_747`, `_02_749`.
- For `H50Z_030_01_005`, six common labels fit `H50Z_030_01_006` with roughly 0.4–1.5 px residuals, whereas the correspondingly **named** `f8` differs by **3,773.397 px**. It may be a mistaken fiducial identifier; correction requires an expert. No automatic relabeling.

Full per-label residuals and warnings are in `fiducial_qc.json` and `fiducial_review_flags.csv` (generated, untracked outputs).

## Exact small-image export

The planner writes a CSV containing **51 requested 1024x1024 native-resolution crops**:

- 24 around selected `start` keypoints from the four next H50Z frames;
- 27 sparse measurement locations on three projections (3 points per frame x 3 selected measured frames x 3 views).

Each crop records its parent BMP, original pixel ROI, point coordinates, semantics and priority. Exporter never resizes, never fabricates masks, never guesses that two differently named sources are the same frame, and records original BMP and PNG SHA-256.

### Commands

```bash
python scripts/research/bubble_chamber/plan_real_data_requests.py \
  --h50z-labels /datasets/H50Z/labels.json \
  --visualmetro-root /datasets/070 \
  --available-bmps /datasets/H50Z \
  --out-dir results/bubble_chamber/s4_request

# Optional: if only H50Z files are available, creates partial bundle.
python scripts/research/bubble_chamber/build_small_data_bundle.py \
  --requests results/bubble_chamber/s4_request/requested_crops.csv \
  --h50z-bmp-dir /datasets/H50Z \
  --out /datasets/bubble-small-tiles.zip

# To add 070, create a CSV map with frame_id,view_id,file (see view_map_template.csv).
python scripts/research/bubble_chamber/build_small_data_bundle.py \
  --requests results/bubble_chamber/s4_request/requested_crops.csv \
  --h50z-bmp-dir /datasets/H50Z \
  --visualmetro-bmp-dir /datasets/070_BMP \
  --visualmetro-view-map /datasets/070_BMP/view_map.csv \
  --out /datasets/bubble-small-tiles.zip
```

The `view_map.csv` must contain actual local BMP paths, **not** guessed names. Exact view identity needs fiducial review and ideally optical calibration. A frame with missing images is not a dense segmentation sample. No source photo or large archive is committed.

## Tested

- 91 local Python tests passed: baseline tests plus 9 new S4 QA / request / crop tests.
- Four actual 1024x1024 PNG tiles were exported from the already available H50Z `005`/`006` BMPs with original-image coordinates and SHA manifests. Their pixel values match the original ROI exactly; this is an ingestion smoke test, not segmentation accuracy.
- Additional real BMP or expert line-mask annotations are still required for training and independent evaluation of authentic `AZUNet` vs `BaselineUNet` and Frangi. Avoid pseudo-labeling other unmeasured tracks as background.

## Scientific gating

1. Acquire 070 matched 3-view BMP crops and check fiducials + measured pixel correspondence.
2. Confirm a small set of foreground/negative track masks on H50Z. Unreviewed pixels remain UNKNOWN.
3. Compare standard CNN, original AZConv-based U-Net, and classical filters using **identical independent run/film-level data splits** and image-only inference. Compare Dice, clDice, centerline error, connected-track identity, false bridges per frame and runtime, with uncertainty.
4. Only then test ANZA + Graph-LIRA against identical frozen segmentation and candidate sets; do not compare against a weaker candidate pool.
5. No claim of ANZA superiority at the current stage.
