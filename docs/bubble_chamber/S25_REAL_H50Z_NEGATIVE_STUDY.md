# S25 — real H50Z image experiments (2026-10-25)

**Result: negative / exploratory, not real-track validation.** Original source-locked AZUNet and BaselineUNet were evaluated locally on actual BMP pixels from user-supplied H50Z_030_01_005 and H50Z_030_01_006. Neither `models/azconv.py` nor `models/segmentation.py` was modified.

## Data and correct label interpretation

Two authentic photographs from ONE run; the private `labels.json` describes six `start` and seven fiducial points on 005, three `start` and eight fiducials on 006. **These are not ground-truth track masks.** Original expert-clicked visualMetro shapes from film 070 were injected synthetically into authentic BMP background crops; only injected trace and synthetic scratch pixels were annotated. All native photographic content remained UNKNOWN.

## Reproducible exploratory results

| Artificial overlay budget | Model | injected positive detected | missed positive | synthetic scratch pixels falsely accepted |
|---|---|---:|---:|---:|
| 18/photo, 3 epochs | U-Net | 698 | 58 | 679 |
| 18/photo, 3 epochs | original AZUNet | 756 | 0 | 750 |
| 48/photo, 8 epochs | U-Net | 1631 | 385 | 1296 |
| 48/photo, 8 epochs | original AZUNet | 2007 | 9 | 1995 |

At 48/photo, AZUNet still produced an almost everywhere-positive prediction. The test metrics describe **only synthetic injected content**, not physically correct segmentation of native H50Z tracks. A longer 120/photo U-Net run finished but the matching AZUNet run did not, so no paired 120-photo result is claimed.

A second experiment classified **human labeled** local start-vs-fiducial patch types, trained on 13 points from photo 005 and tested on 11 from 006 (8 epochs). Held-out AUC: ordinary U-Net 0.4167; original AZUNet 0.0. This is not track segmentation, and the point sample is too small to establish generalization.

## Decision

S24's apparent reduction to 2 false geometry joins remains a different task: manually measured coordinates with synthetic gaps and HGB predictors, **not original AZConv image features**. Do not transfer that number to real BMPs. Missing real expert track foreground, certified distractors and same/different instance relation labels prevent a credible end-to-end accuracy estimate. Keep every proposed join **PROVISIONAL_REVIEW_REQUIRED**. External film/run testing and stereo/fiducial/B-field calibration are prerequisites for physics.

The reproducibility source, local checkpoints, image visualizations and full receipts are in the user-private S25 research artifact; real BMP and private labels are NOT committed to GitHub.
