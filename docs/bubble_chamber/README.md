# Bubble chamber pilot P0 (new independent research line)

Based on the **2026-10-14 Varvara `research/coronary-connectivity-repair`** head (`80a7521c2000a9eced23b492cb7ecf082bf64b4a`). This **does not change** the coronary/CT28 study, its patients, splits, thresholds, or saved metrics.

## Scope and scientific claims

The first pilot is a **classical, deterministic 2D baseline** and data contract for later ANZA + Graph-LIRA experiments. It proposes (not confirms): dark track fragments, local PCA quadratic fits, geometric endpoints, 14 points per fragment, width/contrast proxies, and candidate joins. There is **no particle classification, physical direction, mass, momentum, full 3D reconstruction, or quantitative accuracy claim on real images**.

The CT28 Graph-LIRA lesson carried forward is to separate **candidate generation**, **relation identity** (`NONE/PAIR/JUNCTION/BOTH` later), and **selective repair/abstain**. A wrong merge across distinct particle tracks may destroy an event. In P0, all join proposals are `needs_review`; no paths are automatically filled. The work is not an implementation of the trained CT28 relation head. Later Graph-LIRA is a separate experiment with fitted thresholds and independent calibration.

## Real reference and data policy

One unannotated photograph named `action.jpg` is present in the user-provided research archive `пузыри полезное.zip` under `старые автоматы/`. It is **not committed**: source permission and scientific metadata are unverified. The following local pilot is reproducible if you supply that file separately:

```bash
python scripts/research/bubble_chamber/run_pilot.py \
  --image '/path/to/action.jpg' --roi 75 65 850 1450 \
  --out-dir results/bubble_chamber/real_action_p0
```

A public archival reference that can be downloaded in a normal networked environment is **W.pseudon, Recording bubble chamber (2015)**, Wikimedia Commons, CC BY-SA 4.0:
https://commons.wikimedia.org/wiki/File:Recording_bubble_chamber.jpg
https://creativecommons.org/licenses/by-sa/4.0/
This is a single example, **not** a labeled machine-learning dataset. Any distribution/derivative must retain license/attribution. The runtime used for this first pilot could not reach `upload.wikimedia.org`, so public download success is NOT claimed.

The archive consists principally of papers/manuals rather than a labeled 3-view collection. The needed actual dataset must include BMP triplets, film/run/camera IDs, fiducials, calibration constants, 3D field maps, original measured track-point files (`*_trkList.txt`, individual tracks), and expert identity/junction labels. Do not pretend individual crops from one frame are independent experiments.

## Execute controlled smoke tests

```bash
python -m pip install -r docs/bubble_chamber/requirements.txt
python -m pytest -q tests/test_bubble_chamber_pilot.py
python scripts/research/bubble_chamber/run_pilot.py --synthetic all \
  --out-dir results/bubble_chamber/synthetic_p0
```

Output: `fragment_overlay.png`, `foreground_mask.png`, `result.json`, `measurement_points.csv`; synthetic cases also include a ground-truth mask and tolerant centerline metrics. Metrics on synthetic images are **software smoke-test measurements only**, not estimates of scientific accuracy or domain transfer.

## Numerical meaning

The quadratic is fitted in a local PCA-aligned plane `(u, v)`: `v = a*u*u + b*u + c`. Coefficients must be interpreted with the returned `origin_xy_px`, `u_axis_xy`, `v_axis_xy`. This is **not** a global `y(x)` parabola; highly curled tracks require a different fitting family. A and B are the observed 2D fragment endpoints, not start/stop of the particle. Thickness is estimated from thresholded mask via a distance transform and must NOT be reported as calibrated `dE/dx` or ionization.

Pixel coordinates in the emitted CSV refer to the **original input image**, even when the pilot uses an ROI. Points are arc-length spaced, not stereo-corresponded; `physical_direction_known=false` is always emitted.

## Next falsifiable experiments (not part of P0)

1. Establish frame-based, run-disjoint annotated splits for H2 and C3H8; baseline classical ridge filtering, Steger, U-Net. Report endpoint localization and calibrated track centerline error.
2. Adapt ANZA local orientation-aware segmentation on the same splits; controls must include simple geometry baseline and generic anisotropic models, not just U-Net.
3. Train/compare relation heads on positives/negatives that include separate tracks looking aligned, gaps, crossings, star junctions, and `NONE`. Never infer relations from pointwise semantic masks alone.
4. Add 3-view epipolar/optical constraints, 3D matching, and eventually compatible export for visualMetro/VGFCalc. Validate the real file schema before claiming interoperability.
5. Only after reliable geometry: darkening/width calibration, secondary stars, rare-event triage, solver-assisted hypothesis testing.

## Source pointers

- Research base: `docs/varvara/CURRENT_TASK.md`, `docs/varvara/TASK_CT28_CT_CONDITIONED_GRAPH_LIRA_2026-10-06.md`, and `scripts/research/ccta_graph_lira_safe_repair/run_ct28_graph_lira.py`.
- JINR: A. V. Belyaev, E. A. Bushmina, E. A. Klevtsova, *visualMetro user guide* (2025), sections on stereopair points, fiducials and saved track files.
- Belyaev, JINR P1-2023-22, mass estimation by fitting a stopping track; physical mass inference requires calibrated 3D measurements.

## New real data, 2026-10-24

See [REAL_H50Z_AND_070_DATA_AUDIT_2026_10_24.md](REAL_H50Z_AND_070_DATA_AUDIT_2026_10_24.md). Two H50Z BMPs and point-only labels are available, but there is no dense line mask; film 070 has a different, sparse three-view measurement dataset without matching BMPs. Use `export_small_bmp_tiles.py` to send lossless point-registered 1024x1024 crops instead of multi-hundred-MB BMPs; do not pair H50Z and 070 as the same event.
