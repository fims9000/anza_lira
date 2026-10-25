# S31 — machine-first preliminary annotations, corrective review and original ANZA bootstrap

**Status 2026-10-25:** implemented and evaluated as an *experimental annotation workflow*. The native-real physical track recognition problem remains unsolved; no native full-track gold masks or independently verified physical associations are available.

## Scope and sources

- Source photos: exactly two genuine H50Z BMPs from the **same** physical acquisition, 005 and 006, with 39 512×512 crops from original S29 review HTML.
- 9 genuine operator `start` points (6 on BMP 005, 3 on BMP 006). This landmark is not an image-line mask.
- 72 native 2D S28 line hypotheses (36 curvature-constrained classical, 36 using **original AZUNet** visual features), plus 61 straight image-segment LSD proposals for the 39 real crops.
- Across the 39 crops: **133 machine draft polylines**, 28 short-range candidate fragment-link questions. **0** human-verified native track masks, **0** physicist-confirmed segment associations.
- 30 of the 39 crop image *file* SHA-256 match the original S29 PNG bytes. For the nine operator-start crops, the original encoded PNG bytes have a different checksum because of PNG encoding, **but their decoded grayscale pixels and pixel coordinates match the original BMP exactly**. S31 records both the original S29 PNG SHA and new derived PNG SHA, and a SHA of raw 512×512 pixel values.

## Our initial annotation strategy

S31 *does not wait for expert truth* before generating annotations. The `s31_machine_first.py` generator first provides the existing S28 curvature-constrained classical/ANZA candidates from each real `start` point. It uses S30 to label duplicate/fork cases and reports photo-local geometric fit and contrast, retaining **all alternatives**. For the rest of the film, an image line-segment detector yields other potential track pieces. Close, geometrically compatible fragment endpoints become **28 pair questions**; no real particle identity is inferred from geometric adjacency.

All of these have `status=MACHINE_DRAFT_UNREVIEWED`, `label_source=MODEL_NOT_HUMAN` and `physics_permission=false`. Short dark lines resembling fiducial crosses appear frequently; **fiducial-control and quiet crops are excluded from automatic *positive training* labels**. Candidate previews on control images remain available for rejection or editing.

## Provisional weak mask construction

The first weak dataset contains 39 masks with `1=machine positive`, `0=machine-assumed likely background`, and `255=unknown real photo pixel`. Crucially the 0 and 1 are NOT confirmed foreground/background. Unknowns have zero training weight. Positives are taken conservatively from mutually compatible classical/ANZA start candidates (and selected strong isolated LSD segments). Machine-sampled negatives are selected only from locally low-contrast pixels with a 10–38-px avoidance band and near known pseudo-positives, at a **very low nominal trust**. These are expressly fallible assumptions, NOT certified real negatives.

| Weak mask category | Pixel count in S31 V2 |
|---|---:|
| Machine-provisional positive pixels | 11,097 |
| Machine-assumed weak negative pixels | 25,350 |
| UNKNOWN (left unmarked) | 10,187,169 |
| Crops with provisional positives | 13 / 39 |

No S31 weak mask is allowed to use the original S14 `s14_sparse_image_reviews_v1` human certification schema. Instead it has a separate schema `s31_weak_training_masks_source_locked_v1`; the trainer requires that schema and records that human-certified masks are absent. Thus subsequent human corrections do not contaminate or overwrite original photo bytes or prior labels.

## Interactive correction before later physicist verification

The private `S31_MACHINE_FIRST_REVIEWER.html` contains all 39 real images and their original S29 manual-review workflow, plus 133 editable machine draft centerlines and 28 fragment-link questions. A user can provisionally accept, reject, mark uncertain, or drag polyline nodes; local pair labels can be `LOCAL_SAME_PROVISIONAL`, `LOCAL_DIFFERENT_PROVISIONAL`, or `LOCAL_UNSURE`.

New corrections export **separately** under `s31_machine_corrections_v1`. They are not authenticated physicist decisions, and they cannot authorize a physical join. Full source SHA checks and strict candidate/relation ID validation are enforced when loading correction JSON. The historical S29 `s26_real_track_manual_review_v1` export is left unchanged. An empty correction template is supplied so team members can later exchange only diffs. Retraining with any correction on a crop disables fallback machine pseudo-labels on that crop; only explicitly accepted provisional local paths are used, to avoid reintroducing rejected alternatives.

## Actual original-architecture training

The original Git SHA-protected `AZUNet` and the baseline `BaselineUNet` (`widths=(4,6,8,12)`) trained under identical sparse pseudo-label and photo split, using the same 2 epochs and 4 patch draws/crop. Training uses image 005 and holds out 006 (same physical run). Neither model is trained on 006.

| Model | Heldout pseudo-mark BCE ↓ | Pseudo-positive detection at 0.5 | False accept on machine-assumed negatives at 0.5 |
|---|---:|---:|---:|
| U-Net | 0.6795 | 97.2% | 88.1% |
| Original AZUNet | **0.5784** | 86.4% | 22.9% |

These values measure **agreement with machine-chosen training labels only**. Negative pixels were *assumed*, not certified, so these are *not real track sensitivity or real specificity*.

To make the study less circular, both models were then evaluated on artificially superimposed visualMetro tracks versus short synthetic scratches on photo 006 backgrounds, **not used in creating S31 weak masks**. Native film pixels not altered by the injection remain excluded from the metric. 48 scenes per random seed:

| Independent injection seed | U-Net AUROC | Original AZUNet AUROC | Classical steerable ridge AUROC |
|---|---:|---:|---:|
| 20261029 | 0.6997 | 0.6989 | 0.5793 |
| 20261030 | 0.6071 | 0.5800 | 0.5834 |

The neural advantage is **not stable across injection seeds**, and the original AZUNet **does not consistently outperform U-Net**. These are new synthetic overlaid traces, not actual physical track error rates. No S31 model is permitted to merge tracks automatically, infer particle species/momentum, or claim superiority on unobserved camera runs.

## Reproducibility (private inputs remain local)

```bash
PYTHONPATH=. python scripts/research/bubble_chamber/s31_build_drafts.py \
  --source-html /private/S29_ANZA_2D_metrics_human_reviewer.html \
  --proposals-json /private/S28_real_operator_START_path_proposals.json \
  --tiles /private/h50z_crops_recreated_from_original_BMP \
  --out /private/S31_MACHINE_DRAFTS.json
PYTHONPATH=. python scripts/research/bubble_chamber/s31_build_review.py \
  --s29 /private/S29_ANZA_2D_metrics_human_reviewer.html \
  --drafts /private/S31_MACHINE_DRAFTS.json \
  --out /private/S31_MACHINE_FIRST_REVIEWER.html
PYTHONPATH=. python scripts/research/bubble_chamber/s31_build_weak_masks.py \
  --drafts /private/S31_MACHINE_DRAFTS.json \
  --tiles /private/h50z_crops_recreated_from_original_BMP \
  --out /private/S31_WEAK_MASKS_V2
PYTHONPATH=. python scripts/research/bubble_chamber/s31_train_provisional_anza.py \
  --manifest /private/S31_WEAK_MASKS_V2/weak_manifest.json \
  --tiles /private/h50z_crops_recreated_from_original_BMP \
  --out /private/S31_TRAIN_MATCHED_V2 \
  --epochs 2 --train-per-crop 4 --val-per-crop 3
```

The original full BMPs and `labels.json` are retained unchanged in the S30 private backup. The new self-contained S31 private archive has all 39 derived source BMP crops, draft data, masks, reviewer HTML, model checkpoints, measurements and receipts; **do not push this private archive into the public GitHub repo**.

## Precise next scientific action

1. Start locally correcting S31 draft paths and 28 pair questions; save `S31_machine_labels_local_corrections.json`.
2. Rebuild weak masks with `--corrections /path/to/S31_machine_labels_local_corrections.json`, inspect changes and retrain AZUNet/U-Net on an identical budget.
3. When physicists review the corrections, maintain a **separate expert-certified layer** and test track-instance continuity / wrong joins on the real photos, eventually another run and fill gas. Never label unmarked pixels as gold background.
