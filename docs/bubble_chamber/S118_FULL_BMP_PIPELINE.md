# S118: source-first tiled bubble-chamber tracing and global provisional IDs

Status: research/provisional, **not physical particle tracking, not ground truth**.

## What is implemented

A reproducible CPU-only command can read an original 8-bit BMP and run:

1. `s114_source_geometry.reconstruct`: source-image directional response, skeleton, ordered observed fragments;
2. `s115_junction_graph.associate_observed_junctions`: reconnect branches only along *existing* observed skeleton pixels;
3. `s117_gap_identity.associate_long_gaps`: propose visually supported, reversible same-track fragment groups with distinct inferred connectors;
4. `s118_tile_stitch.stitch_tiles`: deduplicate independent overlapping tile observations into provisional global groups, and export ambiguous cross-boundary alternatives.

At every stage the geometric A/B endpoints are not physical starting/ending points. IDs are hashes of source curve keys, not physical particle identities. No interpolated gap contributes pixels to observed masks.

## One-command BMP execution

Install dependencies:

```bash
python -m pip install -r requirements_s118.txt
```

Small real-region replay (original image and outputs remain private):

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python scripts/research/bubble_chamber/run_s118_full_bmp.py \
  --bmp /PRIVATE/H50Z_030_01_007.bmp --out /PRIVATE/S118_007 \
  --bbox 3600:6000:5400:7600 --tile-size 1200 --overlap 600 --max-tiles 3

# Resume and complete the same 1800 x 1600 region without overwriting old tiles:
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python scripts/research/bubble_chamber/run_s118_full_bmp.py \
  --bmp /PRIVATE/H50Z_030_01_007.bmp --out /PRIVATE/S118_007 \
  --bbox 3600:6000:5400:7600 --tile-size 1200 --overlap 600 --max-tiles 4 --resume
```

Omit `--bbox` and `--max-tiles` to process all tiles of the whole BMP (not yet performance-tested to completion on this image). The pipeline currently uses **CPU OpenCV**, not CUDA. Do not interpret the output as an ANZA GPU inference run.

The pipeline writes `RUN_CONFIG.json`, per-tile `S115_CURVES.json` and `S117_GROUPS.json`, `S118_MANIFEST_NNNNN.json`, `S118_STITCH_NNNNN/S118_STITCH_NOT_GT.json`, `S118_VECTOR_REVIEW.svg`, and a run report. BMP and human annotations never appear in these source-only outputs. On resume, input source SHA-256 and all stored local-curve provenance are checked. A partial tile checkpoint is rejected, not silently overwritten.

`run_s118_tile_stitch.py` can separately stitch any supplied tile manifest, and optionally `--verify-source-bmp` to check actual source BMP bytes without copying them.

## Real in-sample QA, 11 November 2026

- Source: original `H50Z_030_01_007.bmp` (9103 x 26317), with its matching private manual RLE annotation.
- Coverage: **only** rectangle `[3600, 6000, 5400, 7600]`, four 1200x1200 overlapping tiles covering 1800x1600 source pixels. It is NOT a completed run for the entire original BMP.
- 320 observed S115 curve instances; 246 within-tile S117 provisional groups; 133 after S118 cross-tile stitching.
- 113 overlap-alias union operations, 3 additional ambiguous aliases held for review.
- 244 source curves confidently associated to the incomplete manual annotation strokes; among alias proposals 204 consistent pairs, 39 without two confident human labels, zero conflicting confidently attributed human ID pairs; zero global groups with conflicting confident manual stroke IDs.
- This is *development-set* consistency on one partially labeled real photo, NOT particle-level precision/recall, and DOES NOT certify the 39 unknown cases.
- S117 on five other selected ROIs of that same frame had 75 automatic within-ROI links, 60 consistent with confidently matched annotation IDs and 15 unresolved. These are separate runs and should not be combined into one accuracy metric.

Reproducing the tested algorithms requires the actual private BMP and (for the manual QA only) corresponding manual JSON. Do not commit these images, crops, annotation arrays or derived machine previews to a public repository.

## Constraints / next independent acceptance

- A single grayscale frame and its manual drawing do not establish physical particle identities or semantic classes (`track` vs `cross` vs `spiral` etc.). No independent blind test was performed.
- Alias matching has safeguards for near-parallel lines, short overlap and competing endpoints, but can still make a mistake in unusually dense crossings or repeated texture; review-only alternatives are preserved.
- Long cross-tile gaps without image evidence are not automatically accepted by S118 (an expert may explicitly mark a verified candidate). Within a tile S117 uses directional-response/photo support to propose longer connections.
- This version is CPU source-first. GPU segmentation and learned multi-class interpretation are separate unverified work.
- Full-frame batch scale on the original 9103x26317 BMP has not been benchmarked, and the global ID over all tiles is not yet validated.
- Next acceptance should use an independently annotated held-out image: evaluate false splits, false merges, correct cross-tile continuity, long-gap link precision and branch/spiral preservation, including negative controls with parallel tracks and scanner defects.

## Tests

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_s114.py tests/test_s115.py \
  tests/test_s117_gap_identity.py tests/test_s117_runner.py \
  tests/test_s118_tile_stitch.py tests/test_s118_full_bmp.py
```
