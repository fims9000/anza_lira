# S66 — supervised-by-CAL-only endpoint tracing on image-derived ridge fragments: failed promotion

**26 October 2026 · `research/bubble-chamber-graph-lira` · zero native particle GT.**

## Question

After S64 showed 79 synthetic-noncorrect suggestions in 96 full image-to-graph scenes and many incorrect selected endpoints, can **pixel-derived** centerline extension correct the endpoints? This is a new controlled synthetic-on-real-photo experiment, not a modification of the earlier held-out S64 or S65 TESTs.

## Frozen method

Preregistered `S66_FROZEN_IMAGE_ENDPOINT_TRACING_2026_10_26.md` **before** opening S66 CAL or TEST. All proposed polylines come exclusively from the original S39 pixel ridge detector (192x192 gray). Two experimental variants extend each end at most 8/10 px by following only strong multiscale ridge and local darkness pixels with +/-12-degree steering, without any simulator or particle labels. Both use identical original S56 TRAIN geometry scorer (64 TRAIN and 64 source-CAL scenes), negative-pair threshold and global abstention margin .04, plus S59 strict pair gate. No ANZA weights or native photo labels are altered.

The fixed image detector has both a no-extension baseline and strict and relaxed extension arms. New S66 calibration uses 32 artificial scenes per SOURCE H50Z photo on the 30-frame visualMetro CAL fold, seeds `20266601`/`20266611`. Prespecified selection demands at least +2 correct synthetic joins OR +5 one-to-one matched synthetic fragments, without additional noncorrect CAL joins or more matched decoys; otherwise the original pixel detector remains default.

### CAL verdict before opening fresh TEST

| CAL source photo | Original correct/noncorrect | Strict extension correct/noncorrect | Relaxed extension correct/noncorrect | Selected |
|---|---:|---:|---:|---|
| 005 | 8 / 26 | 7 / 30 | 8 / 32 | **Original S39** |
| 006 | 2 / 30 | 5 / 36 | 6 / 36 | **Original S39** |

Neither extension qualifies because it increases incorrect synthetic links; even a higher count of CAL positives is not enough to pass a safety criterion. The immutable saved CAL file has SHA-256 `4b34135bb9c1e2359c98d1af1f351f76fe09a86b338ed158ab0cc6385ea83902`. The TEST file records the same digest. The default algorithm was not modified.

### NEW 96 held-out synthetic TEST scenes

New frame-disjoint TEST seeds `20266603`/`20266604` for 005->006 and `20266613`/`20266614` for 006->005 (24 scenes per seed). These are new scene seeds independent of S64/S65, and source templates are drawn only from the unchanged S37 36-frame TEST split. Both H50Z photographic backgrounds still come from one physical run.

| Fixed arm | Artificial-correct joins | Noncorrect relative to synthetic GT | Complete artificial 3-piece paths | Scenes with noncorrect join |
|---|---:|---:|---:|---:|
| **Original S39 (CAL selected)** | **18** | **73** | **3** | **45** |
| Pixel endpoint extension, strict | 16 | 91 | 1 | 53 |
| Pixel endpoint extension, relaxed | 19 | 93 | 1 | 54 |

Despite the relaxed variant detecting one additional correct pair, it creates **20 additional noncorrect suggestions** and loses two of the original three complete synthetic paths. Strict extension performs worse on all practical metrics. The number of extracted polylines is unchanged by construction (460 across 96 scenes), but the endpoint features and candidate links change, showing why increased candidate length alone is not a meaningful success criterion.

## Verdict and scientific limits

**Negative, no promotion**: original S39 is still the default. Similar-looking photometric fragments and line junctions remain a major limit. No algorithm in S64–S66 has established automatic recognition of physical particles or a valid false-join rate on independently labeled H50Z; synthetic FPR/recall is not the same as physical accuracy. VisualMetro film 070 measured source geometry is not the H50Z photo ground truth. The underlying real two-photo bank is not independent camera-run validation.

The appropriate next research focus is stronger **instance-aware** line extraction and endpoint localization, with false-positive controls and an independently sourced image distribution, not adding more purely geometric global rules or accepting high AUROC on source-provided coordinates.

## Reproduction

```bash
python -m pytest -q tests
python scripts/research/bubble_chamber/s66_test_image_only_endpoint_extensions.py cal \
 --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/source_39_H50Z_crops \
 --out /PRIVATE/S66_CAL.json
python scripts/research/bubble_chamber/s66_test_image_only_endpoint_extensions.py test \
 --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/source_39_H50Z_crops \
 --calibration /PRIVATE/S66_CAL.json --out /PRIVATE/S66_FRESH_TEST.json
```

Full numerical per-seed conditional scene-bootstrap intervals appear in public aggregate JSON. No private photos, 2D pixel coordinates, model checkpoints, or per-scene source identities are committed. No user manual labels were requested or created.
