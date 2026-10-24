# P2: Multiscale ridge segmentation and transverse image photometry

Date: 2026-10-24. Branch: `research/bubble-chamber-graph-lira`.

## Purpose / boundary

Test whether a classical multiscale dark-ridge detector (Frangi) improves extraction of long, possibly broken, low-contrast 2D structures over the P0 local-background contrast baseline. This is a new **classical** comparator; neither ANZA nor Graph-LIRA is trained on bubble-chamber photographs yet. No new particle data are fabricated or labeled. Existing CT28 modules and `main` are unchanged.

## Implementation

- `bubble_chamber/ridges.py`: Frangi with fixed scales 1,2,3, dark ridges and 8-connected component filtering. Outputs the original binary mask; never merges tracks.
- `bubble_chamber/pilot.py`: opt-in `segmentation_method="frangi"` path. Default P0 `contrast` is preserved. Ordered fragments and 14 measurement points remain in original image pixel coordinates, with explicit geometric rather than physical endpoints.
- `bubble_chamber/photometry.py`: transverse, image-intensity-based FWHM and local darkness sampled at the 14 measurement locations, each with an observable or explicit abstention reason. This **cannot** be treated as ionization, `dE/dx`, particle mass or identity without empirical photometric and physics calibration. The profiles are provided in Frangi mode only; P0 is left unchanged.
- `bubble_chamber/segmentation_study.py`: synthetic scene families with ground-truth *visible* centerlines, random 2D geometry, varying background, gaps, crossing/parallel tracks and artifacts. Note: `clean_proxy` and `clutter_proxy` are **not** physically validated H2/propane simulators.
- `scripts/research/bubble_chamber/run_segmentation_study.py`: validation-only selection and fixed test evaluation; runs both methods on identical synthetic scenes.

## Dataset and protocol

- Deterministic seed 20261024; 384x384 synthetic images; 2 optical appearance families.
- **Validation:** 12 independent scene seeds per family; tune contrast threshold from `[12,18,24]` and Frangi response fraction from `[0.08,0.12,0.16]`. Primary selection is validation macro-F1 of *surviving long fragments*. Tie-break fewer spurious fragments and higher threshold.
- **Test:** 18 different scene seeds per family (36 test scenes total). No threshold changes or model selection on test data.
- Matched centerline tolerance = 3 pixels. Raw foreground pixels do **not** count unless part of a surviving fragment of at least 28 px. No 3D track identity truth is provided in this protocol.
- Paired-scene 5,000-repetition bootstrap with a fixed RNG seed describes uncertainty. The statistical unit is one synthetic scene, not pixels or crops.

## Test results (macro across 18 independent synthetic scenes per family)

| Synthetic family | Method | Precision | Recall | F1 | Surviving fragments / scene | Spurious fragments / scene |
|---|---|---:|---:|---:|---:|---:|
| clean_proxy | contrast, selected threshold 12 | 0.907920 | 0.991040 | **0.937061** | 2.611 | 0.389 |
| clean_proxy | Frangi, selected fraction 0.16 | 0.907511 | 0.975108 | 0.926042 | 2.500 | 0.389 |
| clutter_proxy | contrast, selected threshold 12 | 0.785462 | 0.795220 | 0.735100 | 3.889 | 1.333 |
| clutter_proxy | Frangi, selected fraction 0.16 | 0.844766 | 0.913403 | **0.835721** | 4.722 | 1.333 |

Paired mean per-scene F1 difference (Frangi minus contrast):
- clean_proxy: **-0.01102**, 95% bootstrap CI `[-0.037833, +0.004115]`;
- clutter_proxy: **+0.10062**, 95% bootstrap CI `[+0.006609, +0.220956]`.

Frangi costs notably more CPU: ~0.18 s/384px scene, versus ~0.01 s for contrast in the measured local runtime. Machine- and run-dependent runtimes are only indicative. Per-scene details and full conditions are preserved in `docs/bubble_chamber/p2_segmentation_metrics.json`.

## One real-photo, NO-GT qualitative check

Source: archive `пузыри полезное.zip`, member `старые автоматы/action.jpg` from user. 912x1744 pixel grayscale, one photo, no external licence or independent expert segmentation. ROI in original image coordinates `[75,65,850,1450]`.

- Previously P0 default threshold 18: 64 fragments, 17 geometry-only links for review. *This default is not the validation-selected contrast setting.*
- P2 contrast at validation-selected threshold 12: 95 fragments, 25 candidate links.
- P2 Frangi at validation-selected fraction 0.16: 37 fragments, 3 candidate links.
- Frangi profile measurements for 37 fragments: 518 sampled positions, 511 valid image-space width estimates, 2 low-local-contrast and 5 unbounded-profile abstentions. **No physical dE/dx** or correct/incorrect links are inferred.
- Both methods visibly miss some real faint tracks and sometimes extract photograph artifacts. More/fewer fragments is *not* accuracy; without expert labels there is no quantitative sensitivity/specificity for this real photograph.
- Human review HTML output and CSV are generated locally only. Do not commit real-photo pixels to public Git without permission.

## Exact commands

```bash
python -m pip install -r docs/bubble_chamber/requirements.txt
python -m pytest -q tests/test_bubble_chamber_pilot.py tests/test_bubble_chamber_review.py tests/test_bubble_chamber_public_samples.py tests/test_bubble_chamber_relation_study.py tests/test_bubble_chamber_segmentation_study.py tests/test_bubble_chamber_photometry.py
python scripts/research/bubble_chamber/run_segmentation_study.py --out results/bubble_chamber/p2_segmentation_study.json
python scripts/research/bubble_chamber/run_pilot.py --image '/local/action.jpg' --segmentation frangi --ridge-fraction 0.16 --roi 75 65 850 1450 --out-dir results/bubble_chamber/real_frangi_p2
python scripts/research/bubble_chamber/make_review.py --result results/bubble_chamber/real_frangi_p2/result.json --image '/local/action.jpg' --out results/bubble_chamber/real_frangi_p2/review.html
```

## Interpretation and next gate

Frangi improves this small, noisy **synthetic** test family, while the contrast method remains a strong fast comparator on clean scenes. It is not evidence of H2-to-propane transfer, particle identification, valid isochronous 3D reconstruction, or safe automatic joining of tracks. The next meaningful research gate is expert segmentation and same/different-fragment labels on independently sourced calibrated **3-view H2 and propane films**. Use entire runs/films as split groups. Report false-bridge event risk and review time, not only pixel F1. Do not import CT28 thresholds, labels or head weights; they were trained for unrelated coronary CT evidence.
