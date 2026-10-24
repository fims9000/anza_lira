# Bubble chamber: P0 pilot (2026-10-24)

Branch: `research/bubble-chamber-graph-lira`; base: Varvara's `research/coronary-connectivity-repair` at `80a7521c`.
This first milestone is an auditable **classical 2D baseline**, not a trained ANZA/LIRA model, and not a physics solver.

## Scientific target
Automate the laborious input stage of `visualMetro`: candidate track fragments, continuity proposals, 14 arc-length measurement points, 2D local quadratic parameters, image-intensity and thickness proxies. Later add stereo matching, fiducial calibration, vertex identification and physico-kinematical solvers. No single image is enough to infer 3D momentum or particle identity.

Varvara's CT28 Graph-LIRA was reviewed as design reference: explicit candidate generation; separate geometry and image evidence; relation decisions with NONE/abstain; conservative structural repair. Its CT28 dataset, relation models and thresholds are NOT transferred to photographic bubble-chamber data.

## Reproducible commands
```bash
python -m pip install -r docs/bubble_chamber/requirements.txt
python -m pytest tests/test_bubble_chamber_pilot.py -q
python scripts/research/bubble_chamber/run_pilot.py --synthetic all --out-dir results/bubble_chamber/synthetic_p0
python scripts/research/bubble_chamber/prepare_reference.py --archive "/path/to/пузыри полезное.zip" --out-dir data/bubble_chamber/action_subset
python scripts/research/bubble_chamber/run_pilot.py --image data/bubble_chamber/action_subset/action.jpg --out-dir results/bubble_chamber/action_p0
```
Do not commit the source archive, image, extracted crops, or generated artifacts without verifying redistribution permissions.

## Measurements and limitations
- `result.json`: 2D fragments, quadratic fit in a local PCA frame, pixel RMS, geometrical A/B ends, width and darkness proxies.
- `measurement_points.csv`: 14 *geometrically* arc-spaced points per surviving fragment, in **original image pixel coordinates**. The number is an initial example, not a justified scientific optimum.
- `join_proposals`: geometry-only pairs for **human review**, never auto-joined.
- `synthetic_centerline_metrics`: tolerance 3 px, synthetic only; not physical precision or stereo track identity.
- No mass, momentum, charge, ionization dE/dx, physical production/stopping direction or neutral-particle labels. No representative expert masks or correct cross-camera correspondences available in this milestone.
- A source reference `action.jpg` may be extracted from the user-provided archival ZIP; it is **one unannotated photograph**. The derived four crops share the same parent and are not separate statistical samples. Do not cross-split.
- Source media (hydrogen vs propane) and optics are not inferred. No claim about adaptation to another filling without separate calibrated experiments.

## Acceptance and next research gate
1. Obtain expert-reviewed 3-view BMP triplets and corresponding manual `visualMetro` files (`trkList.txt` and individual track measurements), camera/run optics and magnetic maps.
2. Validate a basic named-track identity protocol: disconnected same-track pieces, parallel independent tracks, 2D crossings vs actual stereo vertices, stopping/escaping tracks, weak/noisy images.
3. Report centerline position error in calibrated units, false bridges **per event**, review coverage, run-disjoint transfer and operator time; choose thresholds on train/validation only.
4. Compare classical ridges/Steger and U-Net against ANZA under identical splits; then test LIRA connection evidence and stereo/physics veto. Do not claim gain unless this paired comparison passes.
5. Implement and test explicit `visualMetro` export only after inspecting actual per-track file contents. Never guess the schema.

## Status
Code and tests committed as P0 bootstrap. **No held-out real bubble-chamber accuracy claim, no trained neural network, no final scientific experiment.** This README does not certify that tests were executed in the remote GitHub runtime.
