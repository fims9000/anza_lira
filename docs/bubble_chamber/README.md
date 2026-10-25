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

## S2/R3 (24 October 2026): actual sparse measurements and partial mask bootstrapping

New code: `bubble_chamber/real_sparse_lira.py`, `h50z_seed_analysis.py`, `track_mask_review.py`, `anza_segmentation_adapter.py`. The accompanying S2 study shows that basic geometry **outperformed** ANZA-inspired features in the very limited same-film sparse coordinate task. H50Z keypoint proposals are **not** ground truth. See `docs/bubble_chamber/S2_REAL_SPARSE_LIRA_AND_H50Z_REVIEW.md` for exact commands, metrics and gates.

## S4 acquisition, crop and label-quality audit (2026-10-24)

See [S4_DATA_ACQUISITION_AND_FID_QC_2026_10_24.md](S4_DATA_ACQUISITION_AND_FID_QC_2026_10_24.md). An automated CSV request selects three-view 070 and H50Z vertex crops. Fiducial duplicates/outliers are flagged without relabeling. The local image-to-track segmenter still needs real reviewed masks.

## S5 real sparse expansion (24 October 2026)

A new user-provided archive `1mH2Data.7z` contains 152 frame folders of film 070, not BMP photographs. Audited ingestion yields **145 complete frames, 1,073 measured tracks, 17,321 3D clicks and 51,963 stereo click positions**. Different source view block orders are parsed using explicit IDs, and incomplete/ambiguous source files are quarantined. See `S5_LARGE_1MH2DATA_RESEARCH_2026_10_24.md` and `s5_metric_summary.json`.

The S5 coordinate-only relation study tests artificially withheld clicks (1/3/5) on frame-disjoint train/val/test. Ordinary geometry is very strong; adding ANZA-inspired axial features has **no established incremental benefit**. This study does NOT train AZConv and does NOT evaluate real-image pixel segmentation.

The new `synthetic_from_clicks.py` builds explicitly artificial film-like masks from real measured curve geometry. `scripts/research/bubble_chamber/run_synthetic_anza_pretrain.py` is a reproducible training entrypoint for the *authentic* repo `AZUNet` vs `BaselineUNet`, but no real BMP segmentation performance may be claimed from those synthetic examples.

## S6 preliminary neural segmentation

See [S6 synthetic model comparison](S6_SYNTHETIC_NEURAL_RESULT.md). A local port of AZ local_hyperbolic lost to U-Net on normal generated images but was more robust under one **artificial** low-contrast/noise stress. These results are not real-film superiority evidence and must be reproduced using untouched original model modules.

## S7 original-model reproducibility

[Source-locked original AZUNet versus U-Net](S7_SOURCE_LOCKED_REPRODUCTION.md): exact Git blob checks, validation-only thresholding and checkpoints; automated GitHub Actions CPU smoke for procedural images. No real-film superiority established.

## S8 resumed checkpoints and topology

[S8 reproducible training and topology](S8_RESUME_AND_TOPOLOGY.md): train/resume with frozen data fingerprints, validation-only threshold and track-continuity assessment; no real-film pixel performance claim.

## S9 original-model study on real-click-derived synthetic images

[Report](S9_ORIGINAL_MEASURED_GEOMETRY_REPORT.md) and [three-seed results](s9_multiseed_metrics.json). The original AZConv kernel remains unchanged; residual AZ and robust contrast/noise training show mixed outcomes. **No stable ANZA advantage over standard U-Net or reliable real-film pixel accuracy is established.**

## S9 single-command scientific reproduction

A verified result package containing original model code, six weights, qualitative H50Z examples and a full SHA256 inventory can be replayed against the user-provided measured-track archive without re-training:

```bash
python source/scripts/research/bubble_chamber/s9_replay_release.py \
  --package /path/to/unpacked/bubble_chamber_S9_verified_complete \
  --archive /path/to/1mH2Data.7z
```

See `docs/bubble_chamber/S9_ORIGINAL_MEASURED_GEOMETRY_REPORT.md`. This replay verifies **synthetic generated-pixel** metrics from expert-measured geometry, not segmentation accuracy on real bubble-chamber photographs.

## S10 fixed-data architecture comparison

[Fixed-frame S10 study](S10_FIXED_SPLIT_EVALUATION.md): original ANZA vs U-Net vs Frangi across 3 model initializations, synthetic images derived from expert clicks. All 6 checkpoints replayed; **no stable ANZA advantage** established and **no real film Dice** is claimed.

## S11–S12 topology and direct tangent supervision

[Original ANZA experiments and ablations](S11_S12_EVALUATION_2026_10_24.md): real visualMetro geometries with synthetic pixel renderings, 3 independent model initializations, unchanged original AZConv and U-Net source. Heavy clDice harms quality; direct tangent orientation yields a **small clean synthetic Dice gain**, not stable under stress. A safety-gated 2D polynomial fit reports image-pixel coefficients/endpoints and abstains on fragmented tracks; no physical 3D reconstruction or real-film accuracy established.

## S13 gated AZConv guidance (negative result)

[Controlled S13 report](S13_GUIDED_AZ_EVALUATION.md): three-seed original-AZConv-guided U-Net vs regular-conv-guided U-Net, plain U-Net and pure AZUNet on fixed expert-measured track geometries with generated image pixels. No statistically supported AZ-specific gain and no real-camera pixel masks. Do not promote this branch to automated physical reconstruction.

## S14: few expert labels -> automatic provisional track inference

[Small-label ANZA/Graph-LIRA research and workflow](S14_SMALL_LABEL_RESEARCH_2026_10_24.md): source-locked AZUnet trained on sparse 1/0/255 expert labels with controlled teacher consistency, active collection of same/different connection reviews, and graphical inference on new crops. Three split seeds use real 070 clicked geometry with artificial gaps, not real photographic labels; physical autonomous operation remains unvalidated.

## S17 book-grounded physics priors and synthetic augmentation

[Book-source limited physics and data generators](S17_BELYAEV_BOOK_PRIOR_2026_10_25.md): implements Belyaev's *near-stop-only* hadroida angular law and generates artificial curve/mask data for training original AZUNet, with optional grayscale appearance adaptation from **unlabeled** H50Z. Mixed dataset is 96 expert-click geometries + 24 book curves for training (all raster pixels simulated). Real-photo 3-view calibration and expert masks are absent; no physical PID/real-film accuracy proven.

## S18 book-prior ablation (30 trained models; synthetic-only)

[Controlled S18 report](S18_BELYAEV_ABLATION_2026_10_25.md) compares original AZUNet/U-Net across equal-size measured-shape training, 20% Belyaev stopping-track replacement, real-unlabeled H50Z photometric styles, synthetic histogram styles and generic robust noise (post-hoc exploratory). Three training seeds; one measured film; 13 held-out original frames; **no expert real-BMP pixel masks**. Belyaev geometry alone did not help. Strong photometric augmentation improved AZ average stress Dice but bootstrap intervals include zero and U-Net baselines remain competitive. Source and unit tests committed; private generated checkpoints distributed only as a user archive, not pushed.

## S19 real H50Z low-label expert review (human annotations required)

[Offline expert workbench and safe fine-tuning](S19_REAL_H50Z_EXPERT_ANNOTATION_2026_10_25.md): 7 real grayscale crops from 2 H50Z BMPs, 37 Frangi suggestions **not** ground truth, browser brush to add omitted lines/artifacts, same/different expert link review, source-locked 0/1/255 masks, review-centered patches, original AZUNet vs U-Net sparse training, provisional Graph-LIRA. No actual expert marks have been provided; automatic real-photo accuracy and physical link release are **not** established. Source crops and review answers remain outside Git.

## S20 real unlabeled H50Z pretext and prioritized low-budget expert review

[Source-locked S20 findings](S20_UNLABELED_AND_REVIEW_2026_10_25.md): original AZUNet and ordinary U-Net were trained on artificial missing blocks of **real unlabelled H50Z pixels**, splitting by original source photos. Both reconstruction models lost to classical Telea; **no track segmentation accuracy claim**. Optional encoder-only checkpoint transfer into S19 requires the same source hashes and still refuses to train without true 0/1 expert scribbles. 37 source-hashed Frangi candidate lines and 23 possible pairs are ranked to minimize review effort; scores are uncalibrated and no human decisions are invented. User-owned real images, private checkpoints and reviewer answers are not published in Git.

## S21–S23 few-expert-label Graph-LIRA and review queue

[Source-locked mixed-gap curriculum and graph risk report](S21_S23_FEWSHOT_TOPOLOGY_RESEARCH_2026_10_25.md). On 3 new same-film frame partitions, mixed 1/3/5 missing-click training increased 640-review long-gap Graph-LIRA correct joins from 369 to 554 of 876, with 4 vs 3 false joins. Graph context did **not** consistently beat geometry-only HGB. These are real operator **coordinates** with artificially hidden points, NOT real-photo segmentation. Optional S23 H50Z review priority keeps all 37 line/23 pair decisions unknown; its embedded real-photo HTML and private classifier remain user-local, not committed to GitHub. Physics auto-merging is prohibited.

## S24 — conservative graph consensus, independent audit, review recovery

[S24 controlled report](S24_CONSENSUS_AND_AUDIT_2026_10_25.md) evaluates two geometry/graph HGB models and their agreement-only matcher on new frame splits from one expert-measured film 070 with simulated missing clicks. Graph-only: 900 correct / 9 wrong; geometry-only: 882 / 5; agreement: 834 / 2 on 993 possible correct continuations. A counterfactual 80 manual review questions per split adds 114 verified positive links across three splits but does NOT eliminate two false matches. The audit contains too few accepted joins for a real physical error guarantee; automatic release remains prohibited. No raw BMPs or manual photographic labels committed.

## S26–S29: original ANZA + classical baselines on 2 genuine H50Z film photos

[Scientific S27–S29 report](S27_S29_ANZA_H50Z_ROBUST_TRACK_RESEARCH_2026_10_25.md) documents 39 authentic 512×512 photo crops, 9 real operator START points, original AZUNet/U-Net, seam-safe overlapping inference, and 72 candidate 2D parametric quadratic/brightness/width measurements. On software-injected trace controls, curvature-constrained classical tracking outperforms the ANZA-assisted mixture; all **native photographic track pixels remain UNLABELED**. Private embedded-photo S29 browser workbench and checkpoint weights are provided in the user research archive only, not in Git. Artificial gap accuracy and seam smoothness must **never** be reported as true H50Z physical reconstruction accuracy. 140 locally passing cross-stage tests; no unattended physical joins.

## S30 dataset integrity and private preservation

S30 includes a conservative path-identity gate that never makes automatic physics merges. Source images and operator annotations are privately archived (public repository). See [data recovery + integrity](S30_DATA_PRESERVATION.md) and [original source SHA-256](s30_dataset_integrity.json). The repository stores code, metadata and hashes only, **no original BMP, visualMetro coordinate dataset or reviewer HTML**. Verify raw datasets with `python scripts/research/bubble_chamber/s30_verify_dataset_archive.py --manifest docs/bubble_chamber/s30_dataset_integrity.json --source-dir /path/to/raw`.

## S31 machine-first draft annotations (later expert review)

[Full S31 protocol and negative controls](S31_MACHINE_FIRST_PSEUDOLABELS_2026_10_25.md): **133** source-locked 2D machine draft polylines across **39** H50Z real-photo crops, **28** provisional fragment relation questions, independent separate JSON of locally editable user corrections. The first weak training mask keeps **10,187,169** photo pixels UNKNOWN and flags 11,097 machine provisional positives + 25,350 assumed low-weight negatives. Source-locked original AZUNet and U-Net trained on the draft masks, with held-out artificial-injection controls at two seeds; results vary and neither real-track accuracy nor physical links are validated. Machine proposals are never promoted to expert ground truth. **Private HTML, real photo crops and experimental weights live only in the user's private Library archive, not GitHub.**

## S32: confidence-aware machine pseudo-labeling and priority correction workbench

[Results and protocol](S32_MACHINE_FIRST_CONFIDENCE_2026_10_25.md). Original AZUNet and U-Net, identical two-seed pseudo-label ablation on 39 real H50Z crop pixels and 96 synthetic trace/scratch overlays on a held-out original photo. S31 loss silently ignored mark confidence; S32 now uses it and balances pseudo-positive/negative windows, but **no reproducible external AUROC win was established**. 133 unverified machine line drafts and 28 provisional pair questions are ranked for local correction; no confirmed real track pixel/physics link is created. Private S32 offline HTML, real H50Z crops and checkpoints are in the user's private Library archive, not GitHub.
