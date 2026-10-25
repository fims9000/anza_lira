# S26 — ANZA–LIRA on TWO authentic H50Z photographic backgrounds

**Date: 2026-10-25 · Exploratory research, not a validated automated bubble-chamber reconstruction.**

## 1. Sources and careful definition of ground truth

Actual user-owned H50Z BMP images are `H50Z_030_01_005.bmp` (9020×26151 pixels) and `H50Z_030_01_006.bmp` (9268×25324 pixels), from the **same H50Z run** and the **same view `01`**. No cross-run generalization or three-view 3D reconstruction can be tested with them alone. User-supplied `labels.json` includes **6 start and 7 fiducial markers** on 005, and **3 start plus 8 fiducials** on 006. It does **not** provide true pixel masks for those physical particle tracks, individual bubble contours, true track-identity links, or particle labels.

An audited visualMetro dataset (film 070) supplies real, manually clicked track SHAPES only; those shapes are newly drawn in software over unrelated authentic H50Z backgrounds. Dark line strengths, artifact scratches and labels of the added marks are **synthetic**. Every original native pixel without an independently certified expert mark remains `UNKNOWN` (`255`), never a background class.

The physically relevant end goal is: detected dark track/individual bubbles → true line fragment instances → conservative continuation and star/crossing proposals → expert or physically validated confirmation → calibrated geometry. There are not yet sufficient verified real-photo line/pair examples for end-to-end accuracy claims.

## 2. Fixed source-photo split and evaluation

- **Training**: 144 random 64×64 patches from the upper/middle spatial region of photo 005, artificially overlaid with one visualMetro-derived trace and negative synthetic scratch marks.
- **Model selection and threshold calibration**: 48 new patches from a spatially disjoint lower portion of photo 005. No class label of 006 is used to select epochs or thresholds. Threshold chosen to maximize positive detection while keeping **observed val synthetic-scratch FPR ≤10%**.
- **Test**: 96 patches from photo 006, same real image acquisition but different frame; **the entire test target measures synthetic inserted traces vs injected scratches**, not native photo truth. Two injection/training seeds `20261027` and `20261028` used identically for compared methods.
- Same 144 train/48 val/96 test artificial mark examples per seed for all models. Original unchanged `AZUNet` and `BaselineUNet`, small `widths=(4,6,8,12)`, 2 original AZ rules, AdamW 6e-4, 10 epochs, batch 8. Use training augmentation of intensity and scale. Source models remain pinned to Git blob IDs `f4217ada00024927506fa8ebe520aa1f68ae5854` (`models/azconv.py`) and `0b971c117f3d56cc588ebade6991c65faceb8a2a` (`models/segmentation.py`).
- Input ablations: original inverted grayscale `raw`, `local` Gaussian-background subtraction / 45 clipped to 0–1, and **exploratory** two-channel `local + steerable_ridge` using the same source-unchanged architecture instantiated with two input channels.
- **Study-wide caveat:** although 006 was kept out of each checkpoint/threshold calibration, later variants were designed after seeing earlier test results. Thus 006 becomes an **exploratory development holdout**; these multiple comparisons cannot establish an unbiased final superiority claim. A fresh independent physical run with true pixel masks is needed for final confirmation.

## 3. Head-to-head baselines: ROC AUC on injected synthetic marks

| Method | Seed 20261027 | Seed 20261028 | Notes |
|---|---:|---:|---|
| Inverted grayscale darkness | 0.5944 | 0.5681 | 1 scalar pixel intensity |
| Local contrast alone | 0.5562 | — | No trained weights |
| Black hat morphology | 0.6061 | 0.5752 | Dark-structure filter |
| CLAHE + black hat | 0.5923 | — | Photometric baseline |
| Frangi multi-scale ridges | 0.5038 | — | Poor on this synthetic-scratch comparison |
| Sato | 0.6309 | 0.6027 | Stronger ridge filter here |
| Meijering | 0.5872 | — | Traditional neurite enhancer |
| Six-direction steerable ridge | 0.6119 | 0.5807 | Matched orientation filter |
| Pixel-wise handcrafted HGB (7 features) | 0.6406 | 0.6483 | Trained on SAME 005 injected marks |
| Pixel-wise handcrafted logistic (7 features) | 0.6378 | 0.6272 | Same supervision |
| **Original U-Net, raw** | 0.8315 | 0.8423 | 10-epoch cap |
| **Original U-Net, local** | **0.8778** | **0.8611** | Effective photometric normalization |
| **Original AZUNet, raw** | 0.7968 | 0.7252 | Seed 2 val threshold chose abstain-all |
| **Original AZUNet, local** | 0.8469 | 0.8426 | Stable in both seeds |
| **Original U-Net, local + ridge 2ch** | 0.8680 | 0.8484 | Exploratory 8-epoch arm |
| **Original AZUNet, local + ridge 2ch** | **0.8819** | 0.8438 | Exploratory 8-epoch arm; seed1 AZ checkpoint selected at epoch 8 before completion of planned 10-epoch run |

**Caveats to the table**: These are synthetic-injection pixel ranking AUCs, NOT actual H50Z track-Dice/recall. Only the primary 1-channel methods have 10 complete epochs; the two-channel arms used completed 8-epoch checkpoints and are therefore not strict equal-budget against the main 10-epoch methods. There are only two training seeds and only one heldout physical photo. No statistical superiority claim for AZ over U-Net.

The fixed photo005-calibrated binary threshold did not always hold at 10% FPR on 006: for seed1, AZUNet raw FPR=0.164; U-Net local FPR=0.115, while AZUNet local FPR=0.088. This domain shift matters for unattended operation.

## 4. Conditional paired patch-level uncertainty

Resample 96 test patches (3000 bootstrap repeats per seed, full patches as blocks). Because all patches come from one photo and all two seeds reuse the same physical film, this is **not** a confidence interval for independent chamber runs:

| Comparison (mean of per-patch AUC differences) | Seed 27 (95% interval) | Seed 28 (95% interval) |
|---|---|---|
| AZ local − AZ raw | +0.0328 [+0.0009,+0.0643] | +0.1124 [+0.0742,+0.1502] |
| U-Net local − U-Net raw | +0.0413 [+0.0117,+0.0710] | +0.0277 [−0.0013,+0.0605] |
| AZ local − U-Net local | −0.0311 [−0.0547,−0.0073] | −0.0203 [−0.0447,+0.0042] |

Local normalization helps the **original ANZA** consistently on this artificial task, but no compelling evidence of advantage over a well-tuned U-Net has yet emerged.

## 5. Additional REAL human markers: separate, different task

The `start`/fiducial categories in `labels.json` are real annotations. With only 13 annotated 64×64 crops from 005 for training (6 positive start, 7 fiducials) and 11 from 006 for testing (3 start, 8 fiducials), low-capacity handcrafted logistic/HOG/SVM classifiers achieved AUC=1.0 on these particular 11 held-out markers. This can reflect the distinctive fiducial-cross appearance; it **does not establish line-segmentation performance**, because a `start` marker is not a pixel-by-pixel track mask. The original S25 8-epoch U-Net/AZUNet anchor classifiers obtained 0.4167/0.0 on this separate task; those were specialized high-capacity models with very few examples.

## 6. On genuine H50Z 512×512 image crops: provisional downstream capability

Nine 512×512 patches centered around **genuine start markers** were processed at full original pixel resolution using 64×64 tiled neural inference (never shrinking the entire input to 64px). The developer-private offline HTML can toggle exact source PNG, Sato, U-Net, original AZUNet and two-channel AZUNet, strict consensus and model disagreements.

A conservative intersection of trained U-Net-local and AZUNet-local+ridge suggestions produced **98 ordered skeletal fragment candidates, 29 potential local endpoint continuation questions and 32 purely geometric crossing/junction questions** across 9 ROIs. These are not operator-certified line instances or secondary physical stars. Curves are fitted as `v = au² + bu + c` in a PCA-local 2D pixel frame (safe for vertical fragments); the record also stores observed image transverse profiles and widths **NOT calibrated as dE/dx**. Every link is `needs_review`, automatic physical merges=0.

The conservative consensus yields **zero sufficiently long fragments on `H50Z_030_01_006_start00`**. The fallback keeps one candidate from the AZ-only mask available for review; it does not claim recall or automatically infer a track. Strict intersection is useful for limiting conflicting proposals but can miss real structures.

The private offline browser review allows the expert to draw genuine sparse `track` / `artifact` strokes, export a source-SHA locked JSON, reload it with SHA checks and undo actions. The companion `s26_compile_human_review.py` compiles **only real manual strokes** to 0/1/255 PNG masks compatible with the original S14/S19 training schema; absent marks, conflicts and source tampering trigger fail-closed behavior. The delivered review file contains **0** certified strokes and **0** newly validated particle joins.

## 7. Relevant external methods (not all were implemented)

- C. Steger (1998), *An Unbiased Detector of Curvilinear Structures*, DOI `10.1109/34.659930`: subpixel line center and width extraction, a potential next classical benchmark.
- Y. Sato et al. (1998), *Three-dimensional multi-scale line filter for segmentation...*, DOI `10.1016/S1361-8415(98)80009-1`: Hessian-vesselness motivation, already approximated in the Sato 2D skimage baseline.
- Shit et al. (CVPR 2021), *clDice*, topology-aware loss; cannot fairly train or evaluate true topological metrics before native real-photo track masks are supplied.
- Liu et al. (AAAI 2024), *DeepBranchTracer*: joint image and geometric tracking, conceptually relevant to future AZ/LIRA linking; not an implemented or measured benchmark here.
- *Curvi-Tracker* (Pattern Recognition 2025): image-guided iterative refinement of curvilinear segmentation, a sensible future point-of-comparison after trusted native masks are available; not run here.

Do not compare against gas-dynamics "bubble instance segmentation" papers as if they solve historical bubble-chamber **charged-particle track identity**; their target object is different.

## 8. Required stop condition / what closes the physical task

1. A very small number of **genuine** manually confirmed tracks and distractors on both BMPs, with unknown remainder preserved. This is now executable using S26 HTML, not merely a proposal.
2. Train and select original AZUNet/ordinary U-Net only on the source photo005 reviewed marks; evaluate genuine native-line and fragmentation accuracy on human-sealed photo006 marks. Because photo006 has already been observed in S26 development, treat that as a pilot rather than final independent generalization.
3. Obtain another film or run (ideally different filling/optics), plus original stereo views and calibration/fiducial maps, and evaluate **false merges per real physical track** before any unattended use in mass/charge/ionization calculations.
4. Continue scientific work if no expert marks: generate better review priorities and low-cost candidate coverage diagnostics, but never call them validated segmentation.

### User-side quick reproduction

```bash
# The two private BMPs and visualMetro points are NOT committed to Git.
python scripts/research/bubble_chamber/s26_two_photo_baselines.py --bmps /PRIVATE/H50Z --measurements /PRIVATE/1mH2Data/S1 --tiles /PRIVATE/H50Z_CROPS --output /PRIVATE/S26/seed27 --seed 20261027 --train-n 144 --val-n 48 --test-n 96 --epochs 10
python scripts/research/bubble_chamber/s26_anchor_baselines.py --bmps /PRIVATE/H50Z --labels /PRIVATE/labels.json --out /PRIVATE/S26/real_anchor_baselines.json
python scripts/research/bubble_chamber/s26_compile_human_review.py --review-json /PRIVATE/S26_real_H50Z_expert_strokes.json --source-crops /PRIVATE/H50Z_CROPS --out /PRIVATE/S26/reviewed
```

This research milestone is **an operational candidate detector + reviewer, not the final autonomous physical analysis**.
