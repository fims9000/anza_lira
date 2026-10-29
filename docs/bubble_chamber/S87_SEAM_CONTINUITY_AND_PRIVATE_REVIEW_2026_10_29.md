# S87: observation-only cross-tile trace continuity for bubble-chamber BMP segmentation

**Branch:** `research/bubble-chamber-graph-lira` · **Scope:** S86 full-resolution masks, source-only implementation on generated BMPs. **No real H50Z pixel validation or native physical ground truth.**

## Why

S86 segments an original high-resolution BMP into non-overlapping image cores with halo. Its skeleton polylines end at tile boundaries. A visually single continuous dark line may therefore be exported as three or more fragments even though the original observed pixels touch across seams. That creates preventable geometry discontinuity *before* Graph-LIRA. This S87 tool extracts only those geometrically contiguous image fragments for a private reviewer. It is **not** Graph-LIRA, does not fill gaps and does not estimate particle identity.

## Conservative rule

S87 only links endpoints from **different core tiles** when both are on their original tile edges, observed source pixels are immediate 8-neighbors (Chebyshev distance exactly one), each endpoint has only one cross-tile endpoint neighbor, and both outward local tangents are compatible with the one-pixel displacement. The default angle limit is 55 degrees. A nearby crossing with multiple competing endpoints, or tangent conflict, leads to an explicit **ABSTAIN**. Closed cycles are excluded from open-path stitching. We never modify the original S86 pixel mask, skeleton records or physical labels.

Every stitched output is `physical_identity=UNKNOWN`, `is_particle_track_gt=false`, and `source=OBSERVED_BINARY_SKELETON_ONLY_NO_INTERPOLATION`. `points_xy_source_px` contains only positions directly observed in S86 skeletons. The one-pixel seam connection is labeled an observed image adjacency, not a verified `same_particle` link.

## Run privately

First complete S86 on the desired original BMP, e.g.:

```bash
python scripts/research/bubble_chamber/physicists_full_bmp_segmentation.py \
  --bmp /PRIVATE/H50Z/H50Z_030_01_005.bmp \
  --private-out-dir /PRIVATE/results/S86_005 \
  --method frangi --core 384 --halo 48 --full-mask-npy
```

Then run the next, **separate** safe read-only stage:

```bash
python scripts/research/bubble_chamber/physicists_s87_seam_continuity.py \
  --s86-dir /PRIVATE/results/S86_005 \
  --bmp /PRIVATE/H50Z/H50Z_030_01_005.bmp \
  --private-out-dir /PRIVATE/results/S87_005 \
  --max-tangent-angle 55 --preview-limit 12
```

Open `/PRIVATE/results/S87_005/index.html` locally. It shows spatially distributed original 192x192 BMP crops and observed computer-generated polylines across tile seams. All pixel thumbnails and source coordinates remain in that **private** directory. Without `--bmp`, S87 still exports geometry and a plain offline summary, but no source-photo previews. The original BMP must exactly match the source SHA-256 and dimensions from S86 or the tool fails closed.

## Output contract

| File | Meaning |
|---|---|
| `seam_tracks.ndjson` | One row per reconstructed *image-contiguous group* (2+ S86 fragments); original pixel coordinates, ordered member fragment IDs, length and explicit one-pixel seam steps. **Not physical tracks.** |
| `seam_decisions.ndjson` | Every detected 8-neighbor cross-tile endpoint pair, with reason `unique_observed_pixel_continuity`, `multiple_adjacent_endpoints_ABSTAIN` or `tangent_incompatible_ABSTAIN` |
| `report.json` | Source manifest SHA, source BMP SHA, config and aggregate detection/abstention counts; `physical_GT_count=0`, `physical_auto_joins_authorized=false` |
| `index.html`, `preview/*.png` | Offline private review, when a verified original BMP is supplied |

Original S86 `records/`, `masks/`, `fragments.ndjson`, `full_candidate_mask.npy` and all S72/S85 frozen records remain unchanged. Re-running the same S87 configuration with identical S86 inputs is idempotent (bit-for-bit). Changing a recorded output or any S86 mask shard raises an error rather than quietly overwriting scientific evidence. Use a new output directory when changing parameters.

## Tests and current results

```bash
python -m pytest -q tests/test_full_bmp_segmentation.py tests/test_s87_seam_continuity.py
```

The synthetic test creates a 1-pixel dark line crossing two 128-pixel tile seams. S87 recovers at least one image-contiguous path made of at least three fragments, whose coordinates remain 8-neighbor-connected and contain **no interpolated missing pixels**. An adversarial four-endpoint seam crossing produces **zero accepted joins**. A geometrically incompatible right-angle continuation is also rejected. Additional tests verify source-mask tampering, private output restrictions, restart determinism, BMP SHA verification and offline gallery. These tests evaluate implementation contracts only; **do not report them as true bubble-chamber precision/recall**.

## Remaining scientific bottleneck

This feature fixes a *representation break* at tile boundaries; it **does not restore faint lines that the pixel detector failed to see**, solve occlusions, distinguish scratches from tracks, or establish particle-level relations. S86 Frangi remains tile-relative and can leave source-pixel response discontinuities. The next valid target is a frozen-seed pixel-only seam/weak-line benchmark using generated tracks over real H50Z backgrounds *if authorized private pixels are available*, then specialist confirmation of a small set of selected machine proposals on independent physical runs. Keep CAL-selected thresholds separate from TEST, and compare S86 alone vs S86+S87 with explicit abstention and false-continuation counts.
