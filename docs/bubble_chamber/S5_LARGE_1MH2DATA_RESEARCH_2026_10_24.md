# S5: 1mH2Data — large real expert coordinate study (24 October 2026)

## Input and provenance

User archive `1mH2Data.7z`: SHA-256 `9d34dc71fd023de8cb7a21d2e6ef2b973fdc33a670a80bdc5f89ebfebfe65d1a`, 12 MB compressed. It contains 5,627 archive entries and 5,429 extracted nonempty files (~52.5 MB). It is an **export of measurements**, not a directory of photographs. Original `070_008.7z` overlaps substantially with it and is not appended as an independent dataset.

Original measured track files have one 3D point list and three 2D view lists per track. The exporter writes variable block order `1/3/2`, `3/1/2` and `1/2/3`. Read the view ID from each intact block; do not assume block position is view identity. The program retains original operator track identifiers verbatim; `e1to1`-style unusual IDs on frame 070_305 are intentionally not auto-relabelled.

| Provenance finding | Count |
|---|---:|
| Source frame directories | 152 |
| Complete, used frames | 145 |
| Complete, used tracks | 1,073 |
| Real 3D measured clicks | 17,321 |
| Corresponding 2D clicked positions | 51,963 |
| Frames quarantined / not used | 7 |
| Unlisted track files (not used) | 10 |
| Nonstandard but retained IDs | 3 |
| Matched film-070 BMPs in archive | **0** |

One frame (070_034) has a track file with interleaved/scrambled view IDs; the entire scene is quarantined. Six other folders have incomplete/no track listings. No editing of the source files and no manual reconstruction of questionable assignments was performed. See local `audit.json` for exact quarantined files; do not publish the raw source or the test-level reconstruction records without confirming rights.

## Core experiment: can geometry-based Graph-LIRA reconnect measured fragments?

Artificially hide 1, 3, or 5 **operator clicks** inside real manually measured tracks. Build all candidate continuations among the *named* tracks within the same frame, using only geometry of the visible left and right pieces. Do not add oracle candidates after ranking. Deterministic frame-level SHA split: 94 train / 26 validation / 25 test for the 1/3-click experiments (5-click test retains 24 frames after no-eligible filtering). All test thresholds selected on validation, candidate FPR bound 2%, mutual top-1 and 0.05 margin. Same candidate pools for every model. Model classes: fixed distance/direction, logistic ordinary geometry, logistic ordinary+axial, boosted trees ordinary geometry, boosted trees ordinary+axial.

**No original AZConv neural weights are trained in this study.** Axial geometric support is ANZA-inspired evidence **only**, not an ANZA segmentation claim.

| Artificial hidden clicks | Test positive candidates | Plain HGB TP/FP | Axial HGB TP/FP |
|---:|---:|---:|---:|
| 1 | 1,246 | 1,221 / 1 | 1,224 / 1 |
| 3 | 1,039 | 1,025 / 0 | 1,016 / 0 |
| 5 | 864 | 844 / 0 | 852 / 0 |

The paired frame-bootstrap 95% confidence intervals of axial-minus-plain **recall differences** are:
- 1 missing click: +0.24 percentage points; CI **−0.18 to +0.83 pp**;
- 3 missing clicks: −0.87 pp; CI **−1.95 to +0.19 pp**;
- 5 missing clicks: +0.93 pp; CI **−0.20 to +2.48 pp**.

**All include zero. No statistically convincing advantage of the axial variant is demonstrated.** A small number of false joins does not prove a safe operational error rate; these negatives are only alternative expert-measured tracks, not all possible film phenomena. Never auto-merge actual chamber traces from this threshold.

## Interpolation of withheld expert coordinates

Compare interpolation of measured pixel coordinates and 3D positions between visible fragments: linear (2 endpoints), quadratic (3 observed clicks either side), cubic Hermite (two-sided finite-difference tangents). Index parameterization is in measured-click order, not physical time or uniform arc-length.

| Hidden clicks | Median pixel error: linear | Quadratic | Hermite |
|---:|---:|---:|---:|
| 1 | 4.12 | 4.36 | 5.67 |
| 3 | 11.42 | 10.63 | 11.56 |
| 5 | 20.73 | 16.93 | 15.58 |

Long-gap Hermite has a substantially worse **p90 tail** than linear, so median error alone does not justify replacing the interpolation with a cubic. No ground truth about genuinely invisible intermediate bubbles was available.

## What this establishes, and what it does not

- Establishes ingestion and quality control on a **much larger expert-coordinate corpus** than the original 7-frame archive.
- Establishes strong baseline for artificial interruptions on real operator-measured geometries, with holdout of entire frames and per-frame uncertainty.
- Enables generating **synthetic images** based on real geometries (new `synthetic_from_clicks.py`) and a reproducible future comparison using the original AZUNet vs ordinary U-Net in the repository (`run_synthetic_anza_pretrain.py`). This runner is **provided but has not been executed as actual AZConv training** in the present environment; no synthetic neural scores are reported.
- Does **not** establish image segmentation quality on authentic paired bubble chamber BMP; no matching 070 photos were in the source. H50Z images are a different source and cannot be paired automatically.
- Does **not** demonstrate H2-to-propane transfer, neutral particle detection, real particle identification or momentum reconstruction.

## Reproduction

Requires Python, NumPy, SciPy, scikit-learn, OpenCV, PyTorch for the optional train script. Extract the **user-owned archive outside Git**; never run its bundled `deepseek*.py` source. The pipeline only parses measurement text.

```bash
python scripts/research/bubble_chamber/run_large_real_070.py --measurements /private/path/1mH2Data/S1 --out-dir results/bubble_chamber/s5_gap1 --missing-clicks 1
python scripts/research/bubble_chamber/run_large_real_070.py --measurements /private/path/1mH2Data/S1 --out-dir results/bubble_chamber/s5_gap3 --missing-clicks 3
python scripts/research/bubble_chamber/run_large_real_070.py --measurements /private/path/1mH2Data/S1 --out-dir results/bubble_chamber/s5_gap5 --missing-clicks 5
python -m pytest tests/test_large_visualmetro_audit.py tests/test_large_relation_study.py tests/test_large_interpolation.py tests/test_paired_frame_stats.py tests/test_synthetic_from_clicks.py -q
```

The current workspace reproduced the real-coordinate experiments and their metrics. Full AZUNet/CNN training requires the original model modules and suitable compute; compare both fairly on identical synthetic splits first, and **do not** promote any synthetic win to a real-photo claim.
