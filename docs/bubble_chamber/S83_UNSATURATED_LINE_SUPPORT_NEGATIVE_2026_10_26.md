# S83 — Non-saturating local line support: successful feature fix, no validated graph improvement

**Experiment `ANZA-LIRA-BC-S83-20261026`, preregistered 2026-10-26.** Public source-only result: negative CAL, no promotion of S83 or native automatic joins.

## Research hypothesis

S82's local contrast score saturated at a value of `1.0` for most detector-extracted multiscale extra lines and could not distinguish candidates. S83 removes the artificial contrast clipping and introduces spatial consistency along 31 equally arclength-spaced line samples. Each pixel-only candidate gets `mean(positive_sideband_center_contrast) * (0.35 + 0.65 * q25(signal)/(q75(signal)+0.005))`. The source-line arclength resampling was borrowed from frozen S82, but S82 TEST outcomes are **not** used for S83 selection. S83 has entirely new CAL and TEST seeds. Reversing the vertex order leaves the feature unchanged.

S72's source direction graph policy, S56 source-trained scorer and 5% negative-pair CAL threshold, S59 strict pair gate, S69 orientation-invariant correctness, S76 shape-vs-endpoint audit and S80 two-scale consensus extractor are all unchanged. Base polylines are retained as identical objects; only S80 extra lines can be selected. The new feature sees pixels only, no artificial or physical track identities.

## New SOURCE CAL

Source-CAL seeds: photo 005 `20268301`, photo 006 `20268311`, 20 scenes/source (40 scenes total), clean + blur sigma=1.1 paired. The local quality 40th and 70th percentiles are derived from 26 and 10 extra-line pixel scores respectively, without artificial ground truth.

| Source direction | Pixel-derived extra lines | q40 | q70 | CAL selection |
|---|---:|---:|---:|---|
| 005→006 | 26 | **0.172138** | **0.215101** | **S72 baseline** |
| 006→005 | 10 | **0.127184** | **0.218290** | **S72 baseline** |

The new feature **does not saturate**; the numerical thresholds are distinct and below 1.0. However on clean CAL in the 005 source direction S83 q40 and q70 each add one noncorrect synthetic join (7/9 original correct/noncorrect vs 7/10). On blurred CAL in the 005 source direction both yield 2 correct / 10 noncorrect vs baseline 2 / 9. Source 006 has no correct blurred CAL gain. **Neither meets the frozen CAL gate**. S72 baseline was frozen for both directions before TEST.

## NEW held-out synthetic TEST

TEST seeds source 005→photo 006 `20268303`,`20268304`; source 006→photo 005 `20268313`,`20268314`, 24 scenes each (96 per quality condition). Artificial film-070-derived tracks over original H50Z photo crops. These are not confirmed real particle lines; the two H50Z backgrounds belong to one physical run.

| Condition | Method | Artificial-correct graph joins | Noncorrect or unverifiable graph joins | Complete artificial 3-piece paths | Shape-matched artificial fragments |
|---|---|---:|---:|---:|---:|
| Clean | Original S72 | **46** | **36** | 14 | 241 |
| Clean | S83 local q40 | 50 | 41 | 14 | 248 |
| Clean | S83 local q70 | 48 | 39 | 14 | 243 |
| Blur σ=1.1 | Original S72 | **17** | **21** | 3 | 139 |
| Blur σ=1.1 | S83 local q40 | 17 | 21 | 3 | 140 |
| Blur σ=1.1 | S83 local q70 | 17 | 21 | 3 | 139 |
| Blur σ=1.6 | Original S72 | **11** | **14** | 2 | 114 |
| Blur σ=1.6 | S83 local q40 | 11 | 15 | 2 | 114 |
| Blur σ=1.6 | S83 local q70 | 11 | 15 | 2 | 114 |

S83 q40 accepts 31/40 extra candidates on clean TEST, 11/25 at blur1.1, 4/29 at blur1.6. The stronger q70 keeps 16/40, 3/25, 2/29. Local filtering is therefore actually selective, unlike S82, but **neither arm recovers any extra full chain or improves the number of correct joins on blurred TEST**. The clean gain adds noncorrect proposals; at strong blur the false-proposal count rises by one without additional true synthetic joins. Do not promote S83.

The 96-scene fresh TEST was replayed without modification, and the two aggregate JSON files were verified byte-identical (SHA-256 `ed272ea566513d20a4141900ecbd8b1564ebb14bb5fc387be652b25b4dc840e6`). TEST reports per-seed per-direction scene-paired 1200-bootstrap intervals. These are *conditional synthetic scene* intervals, not independent physical camera-run confidence intervals. The source 070 measured geometry is unrelated to the photographed H50Z events.

## Genuine H50Z native machine-only audit (no insertions)

| Machine-only arm | Candidate 2D lines | Proposed advisory UNKNOWN joins | Added S80 extra lines | Extra considered / rejected |
|---|---:|---:|---:|---:|
| Original S72 | 83 | 3 | 0 | 0 / 0 |
| S83 q40 | 90 | 4 | 7 | 12 / 5 |
| S83 q70 | 85 | 3 | 2 | 12 / 10 |

These native 2D lines/joins are unverified machine proposals, NOT physical track labels, never a 3D momentum or stopped particle identification. Authorizing native automatic joins remains **false**.

## What the result changes

S82 revealed a ceiling in a clipped local score. S83 fixes that specific measurement defect and establishes clean separation of candidate selection from scoring. However, it does not yield a gain in the synthetic end-to-end graph metric under blur. The limiting factor is likely the *identity and extent of additional fragments*, not only the relative photometric contrast along them. Future work needs better instance-aware extraction/coverage, likely additional independently sourced photographs for physical validation, rather than cycling dozens more thresholded variants on one run.

## Exact reproduction (private data required)

```bash
python -m pytest -q tests/test_s83_unsaturated_support.py tests/test_s83_public_receipts.py
python scripts/research/bubble_chamber/s83_calibrate_non_saturating.py cal \
 --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/source_39_H50Z_crops --out /PRIVATE/S83_FROZEN_CAL.json
python scripts/research/bubble_chamber/s83_calibrate_non_saturating.py test \
 --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/source_39_H50Z_crops \
 --calibration /PRIVATE/S83_FROZEN_CAL.json --out /PRIVATE/S83_FRESH_TEST.json
python scripts/research/bubble_chamber/s83_native_readonly_audit.py \
 --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/source_39_H50Z_crops \
 --calibration /PRIVATE/S83_FROZEN_CAL.json --out /PRIVATE/S83_NATIVE_UNKNOWN.json
```
