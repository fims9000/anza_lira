# Private full-resolution bubble-chamber BMP segmentation (research tool)

**Branch:** `research/bubble-chamber-graph-lira` · **Status:** source-only engineering implementation, **not native H50Z accuracy validation**.

The `physicists_batch_preview.py` quicklook processes selected 192x192 patches using Hessian/LSD. This additional tool segments **every pixel** of one large 8-bit grayscale uncompressed BMP with bounded memory, keeping the original pixel coordinates and returning all hypotheses as `UNKNOWN`. No Graph-LIRA edges are accepted, no physical track is repaired, and no 3D particle identity is inferred.

## Usage

From the repository root, with access to a **PRIVATE** original photo:

```bash
python scripts/research/bubble_chamber/physicists_full_bmp_segmentation.py \
  --bmp /PRIVATE/H50Z/H50Z_030_01_005.bmp \
  --private-out-dir /PRIVATE/results/seg_005_frangi \
  --method frangi --core 384 --halo 48 --preview-limit 16 --full-mask-npy
```

Quick fast classical control:

```bash
python scripts/research/bubble_chamber/physicists_full_bmp_segmentation.py \
  --bmp /PRIVATE/H50Z/H50Z_030_01_005.bmp \
  --private-out-dir /PRIVATE/results/seg_005_contrast \
  --method contrast --core 384 --halo 48
```

To run **the actual original AZUNet architecture** (not random toy weights), a locally trusted, source-locked S14 checkpoint is required:

```bash
python scripts/research/bubble_chamber/physicists_full_bmp_segmentation.py \
  --bmp /PRIVATE/H50Z/H50Z_030_01_005.bmp \
  --private-out-dir /PRIVATE/results/seg_005_azunet \
  --method azunet --checkpoint /PRIVATE/checkpoints/S14_SOURCE_LOCKED_AZUNET.pt
```

The checkpoint path is an **example**; it must refer to a real locally verified file. The tool refuses AZUNet mode without one. A model trained only on synthetic masks or sparse pseudo-labels is **not** physically certified. Using the same BMP for model selection and reported TEST is prohibited. Classical methods are controls, not certified track detectors.

To resume after interruption with the **same** BMP, model weights and configuration, rerun the identical command plus `--resume`. Every cached mask shard is SHA-256 checked. Changing the original BMP or model/configuration makes resumption fail. No outputs go inside the Git repository.

## Output (all PRIVATE)

| Output | Meaning |
|---|---|
| `index.html` | Offline gallery of predetermined, spatially distributed previews: original / segmentation overlay / mask |
| `masks/y*_x*.png` | Non-overlapping uint8 candidate masks: `255` machine foreground, `0` no machine proposal **not certified background** |
| `full_candidate_mask.npy` | Optional original-size uint8 mask via disk-backed memmap, e.g. `np.load(..., mmap_mode='r')` |
| `records/y*_x*.json` | One auditable record per tile, original pixel offsets, full ordered geometric centerlines, 14 sample points, local quadratic fit, candidate status |
| `fragments.ndjson` | Streamable candidate fragments across the entire BMP; each has stable source-coordinate ID |
| `seam_advisories.ndjson` | Geometric **observed-pixel adjacency** between endpoints at different tile cores. Does not interpolate gaps or assert `same_particle`. |
| `tiles.ndjson`, `manifest.json`, `run_identity.json` | Completeness, exact source checksum, method/config, skip-budget counts and `physical_auto_joins_authorized=false` |

### Design boundaries

- `GrayBMP8` memory mapping is used so a 9000x26000 image is not materialized in RAM. The original is never edited. A halo contributes context to each core; only the non-overlapping **core** is exported, so all source pixels are covered exactly once.
- Each tile is scored independently. Frangi uses its existing **tile-relative** threshold and contrast uses the original fixed P0-style threshold. This is a known possible source of visible seams. The halo helps at tile boundaries; it does not certify seam consistency.
- The skeleton is traced only within each core. Segments crossing a core edge are deliberately split and reported with `touches_tile_boundary`, rather than being silently assigned the same physical identity. Direct 8-neighbor endpoint adjacency is exported **separately** as `UNKNOWN` advisory only. Graph-LIRA must not automatically accept it.
- Dense/noisy tiles may exceed `max_skeleton_pixels` or `max_fragments_per_tile`. Candidate **masks are still exported** and the skipped fragments are reported explicitly, not quietly omitted.
- `geometric_fit` is 2D PCA-local polynomial geometry, not magnetic curvature or particle momentum; 14 points are geometric samples, not visualMetro stereo observations.
- Only the already implemented classical Frangi/contrast and original AZUNet are supported. No new claim of improved H50Z Dice, recall or safe joins is made.
- Input supports uncompressed 8-bit BMPs with identity grayscale palette, like the audited H50Z source. Other formats must be converted **losslessly** without rescaling before inference.

## Verification and next scientific gate

Run `python -m pytest -q tests/test_full_bmp_segmentation.py`. These are **synthetic BMP software tests**, covering complete coverage (including ragged right/bottom edges), unchanged originals, coordinate shifts, checkpoint guard, seam UNKNOWN semantics, fragment budget, Frangi smoke and SHA-bound resumption. They do not measure H50Z accuracy.

Next: compare identical independent whole-film slices with specialist-confirmed line/artifact labels. Use expert review UI (`S19`), never automatically convert pseudo-positive pixels to truth. Report mask/centerline precision and recall, fragment continuity, seam errors, and false-bridge risk, grouped by physical film/run. Obtain separate real H50Z run(s) for TEST. Keep S72/S85 experiment records frozen.
