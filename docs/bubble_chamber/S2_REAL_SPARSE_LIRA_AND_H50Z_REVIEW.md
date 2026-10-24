# S2/R2/R3: Real sparse visualMetro and H50Z annotations — 24 Oct 2026

Branch: `research/bubble-chamber-graph-lira` (forked from Varvara's last CT28 Graph-LIRA branch). No changes to her branch or `main`.

## 1. Objective and available evidence

We aim to adapt original ANZA (`models.azconv`, `models.segmentation.AZUNet`) for photographic thin-track segmentation and LIRA/Graph-LIRA for conservative connection of true track fragments. The aim is to **test**, not assume, improvement over Frangi, classical background subtraction, and ordinary U-Net.

Two provided archive formats have different but complementary annotation semantics:

| Source | Actually available | What can be supervised now |
|---|---|---|
| visualMetro film `070`, text exports | 7 frames, 42 expert track identities, 620 measured xyz points, 1,860 aligned 2D clicks and 168 fiducial positions | Sparse graph/triplet geometry and supervised SAME / DIFFERENT fragment relations on *withheld measured points* |
| H50Z `labels.json` | 105 described frames with 570 `start` keypoints and 837 fiducial keypoints, of which **two matching BMPs** are available | 9 actual-image `start` centers, 15 actual-image fiducials; no dense mask or track identities |
| H50Z BMP 005 and 006 | Two 8-bit, 200+MB film scans | Unlabelled film-image appearance, human-auditable proposals for partial segmentation |

Film 070 has no matching image BMP in the provided text archive. The H50Z images do not correspond to film 070. **Never join those image pixels with the 070 measurements.** Other 103 entries in the H50Z annotation JSON cannot train an image detector until their corresponding images are available.

`.gra` files include physical-coordinate visualizations, parameter curves, axes and fitted results, **not** matching full-image pixel masks. Fitted spatial curves are not independent manual film contours.

## 2. S2: LIRA relation isolation using real expert coordinates

- Input: all 42 verified visualMetro tracks in 7 frames and 3 matched views.
- Fragment protocol: for each track/view, three deterministic interior cuts; hide at least one measured click; withhold some right-hand tracklets to represent NONE; retain only observed left/right endpoint geometry for inference.
- Candidate generation: rank right fragments using only observable normalized distance/tangent compatibility; K=5. All 311 physically available positive pairs were retrieved in this small **easy** candidate set; there are 1,315 negative cross-track candidates, 1,626 candidates total.
- Validation: leave one full frame out for test and another for threshold selection, train on the remaining 5. Apply the same splits for each model, no withheld point or track ID as a model input.
- Three models: (A) basic normalized gap/angle geometry; (B) capacity-larger generic polynomial geometry; (C) geometry + ANZA-inspired bilateral anisotropic fuzzy compatibility features (this is **NOT** a trained AZConv). All use matched logistic model family and same graph rule (mutual-best, competition margin 0.05, abstention) with validation-only FPR cap 2%.
- Metrics are candidate-conditional, with **oracle expert tracklet coordinates**. One film for all 7 frames: no independent-run generalization and no image segmentation evaluation.

| Model | Correct accepted | Incorrect accepted | Correct rejected | Candidate recall | FPR |
|---|---:|---:|---:|---:|---:|
| Basic geometry | 276 | 0 | 35 | 0.887460 | 0/1,315 |
| Generic polynomial geometry | 168 | 0 | 143 | 0.540193 | 0/1,315 |
| Axial fuzzy geometry prototype | 261 | 1 | 50 | 0.839228 | 1/1,315 |

**Decision: reject a claim of ANZA improvement for S2.** Basic geometry is stronger. An image-conditioned learned model would require matching original film 070 images (or new paired expert pixel contours), not fabricated image evidence. Do not transfer any threshold to real unmeasured data.

Reproduce:

```bash
python scripts/research/bubble_chamber/run_real_sparse_lira.py \
  --measurements /local/extracted_070_tree --out-dir results/bubble_chamber/s2_real_sparse
```

Exact per-frame validation selection, test results and failure cases are in `docs/bubble_chamber/s2_real_sparse_metrics.json`. Raw per-candidate predictions are local only (no need to publish user source coordinates).

## 3. R2: H50Z real film *keypoint* proximity, not segmentation

The previous P2 Frangi and contrast settings were held fixed (Frangi fraction 0.16; contrast threshold 12; minimum surviving fragment 28 pixels), without H50Z tuning. The keypoint extraction crops all 24 available source-image landmarks at 384x384 original resolution; no downsampling, no false background mask.

Among 9 actual `start` coordinates, a surviving extracted skeleton came within **12 pixels** of a landmark in **1/9 for contrast and 1/9 for Frangi**. These are only marker-nearby-line diagnostics, **not** sensitivity/specificity, not evidence of a correct track, and not a measure of the model's performance on independent film runs. The start centers are point annotations with approximately 47 px annotation width; very small radii are severe. There is no dense track mask.

Reproduce:

```bash
python scripts/research/bubble_chamber/analyze_real_landmarks.py \
 --labels /local/H50Z/labels.json --bmps /local/H50Z \
 --out results/bubble_chamber/h50z_landmark_r2
```

## 4. R3: Human-in-the-loop centerline labels — bridge to genuine ANZA

`bubble_chamber/track_mask_review.py` proposes Frangi skeleton fragments for each actual H50Z start tile and generates a single offline browser editor. The first run produced **41 segment candidates across 9 tiles**, **zero certified** ground-truth centerline pixels. The interface offers:

- **track_centerline**: expert-certifies this local line as a particle track; rasterized as label 1;
- **certified_artifact**: expert-certifies the line as a photographic artifact; rasterized as label 0;
- **unknown**: inconclusive / different track / still unreviewed. Label **255**, excluded from loss.

Only verified positive and verified negative centerline pixels enter the partial-BCE objective. Rasterization never marks unreviewed film background as negative. The exported JSON is bound to exact photo/candidate hashes, and conflicting positive/negative pixels are ignored. No labels are silently accepted. Use:

```bash
python scripts/research/bubble_chamber/make_track_review.py \
 --tile-manifest /local/h50z_landmark_r2/manifest.json \
 --out results/bubble_chamber/h50z_review_r3
# Open track_mask_review.html in browser, export decisions JSON.
python scripts/research/bubble_chamber/make_track_review.py \
 --tile-manifest /local/h50z_landmark_r2/manifest.json \
 --out results/bubble_chamber/h50z_review_r3 \
 --reviews /local/bubble_dense_centerline_reviews.json
```

`bubble_chamber/anza_segmentation_adapter.py` instantiates **original** repo `BaselineUNet` and `AZUNet` (actual `AZConvConfig`, same widths), and implements masked BCE for explicitly human-certified pixels. It rejects insufficient positives/negatives and requires at least **three independent film/run groups** before making an ordinary train/validation/test comparison. The available two H50Z frames are the *same run* and currently have no certified masks: **real supervised ANZA-vs-U-Net training has NOT been run**. This guard is intentional, not a software failure.

## 5. Validation and next evidence requirement

Local: **82 pytest tests passed** (P0–S2/R3), Python compileall clean; the offline editor JavaScript passes `node --check`. No training weights or physics reconstruction were fabricated. The original AZUNet forward was not exercised in the partial local source snapshot (its official code is present in the full GitHub repo).

To establish improvement, obtain paired film 070 BMP triplets or several independent runs with actual reviewed contour masks and both real foreground and artifact annotations. The experiment must hold out entire films/runs, compare Frangi, conventional U-Net, and original AZUNet under fixed data/threshold/seed/compute policy; report centerline precision, recall, clDice, gap/bridge errors, per-film uncertainty, and wall-clock time. Graph-LIRA decisions remain review-only until an independent false-bridge safety gate passes.
