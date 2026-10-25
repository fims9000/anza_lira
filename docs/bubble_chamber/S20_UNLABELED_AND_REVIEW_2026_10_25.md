# S20 — Use real H50Z pixels without inventing expert labels; reduce annotation workload

**Research status: source-verified, exploratory, incomplete physical validation.** The objective remains autonomous segmentation and safe Graph-LIRA track reconstruction **after** few trusted expert scribbles and link decisions. All real track pixels and true same/different link labels are still unavailable.

## 1. What is real vs generated?

- **Real pixels:** 7 8-bit grayscale 512×512 PNG crops, 4 from original `H50Z_030_01_005.bmp`, 3 from `H50Z_030_01_006.bmp`. Both images are from the SAME H50Z_030_01 physical run; they are not two independent particle-camera environments.
- **Missing-pixel masks:** software-generated 22% 6×6 missing blocks. The original withheld image intensity is the target for reconstructing image appearance, **not** a physical track mask.
- **Human track labels:** zero. **Human approved links:** zero. Unknown pixels remain UNKNOWN; do not create certified backgrounds or tracks from Frangi, AZUNet, the books or an image inpainting model.
- **Original ANZA code:** `models/azconv.py` git blob `f4217ada00024927506fa8ebe520aa1f68ae5854`, and `models/segmentation.py` `0b971c117f3d56cc588ebade6991c65faceb8a2a`. The code verifies these before training. The original sources are untouched.

## 2. Negative control: masked image reconstruction

Compare source-locked AZUNet, ordinary U-Net and **classical Telea inpainting** on EXACTLY the same synthetic holes and photo-level validation. Source photo 005 is training; source photo 006 is validation only, not an independent real test.

Matched comparison: 2 initialization seeds (`20261025`, `20261026`), 6 epochs, 64×64 windows, 16 patches per source tile for training, 8 per held-out tile, batch 8, widths `(4,6,8,12)` and 2 AZ rules. All methods use the same masked validation pixels within a seed. The augmented training arm adds random brightness shifts (up to 50 intensity units), gain ±0.25 and 2× gradient weighting **of the synthetic reconstruction loss**, not real track labels.

| Seed | Method | MAE raw train | MAE augmented train | Classical Telea MAE |
|---|---|---:|---:|---:|
| 20261025 | U-Net | 0.35569 | 0.35789 | 0.02360 |
| 20261025 | Original AZUNet | 0.44846 | 0.39206 | 0.02360 |
| 20261026 | U-Net | 0.13827 | 0.13813 | 0.02421 |
| 20261026 | Original AZUNet | 0.46181 | 0.46211 | 0.02421 |

The earlier separate 8-epoch exploratory run (96 train/36 val patches, different sampling budget) produced U-Net MAE 0.13638, AZ MAE 0.25576, Telea 0.02354. **Do not combine these runs or claim training budget equivalence across the tables.** Both cohorts fail to beat classical inpainting on this proxy, and the photometric/edge adaptation is unstable across seeds.

**Scientific decision:** reconstruction pretraining is only an optional, source-locked encoder initializer. We do **NOT** promote it to the primary segmentation or link detector, and it provides NO validated improvement in detecting real bubble tracks. The encoder needs a genuinely labeled sparse real-photo comparison (scratch vs pretraining) before it becomes a recommended training method.

## 3. Safe encoder transfer into S19

`s20_transfer.py` accepts only S20 checkpoints whose weights, original model Git hashes, architecture/widths/rules and explicit no-human-label flags match. It loads **encoder layers + bottleneck only** (not pixel reconstruction head) into a fresh classifier and logs a SHA-256 of the pretraining file. Training on real masks still requires both independently confirmed track **and artifact** pixels and separate source-photo validation. A pretraining checkpoint cannot bypass these requirements.

```bash
# Controlled unlabeled photo experiment (source data is outside Git):
python scripts/research/bubble_chamber/s20_pretrain_unlabeled_real.py \
  --tiles-dir /private/S19_H50Z_Expert_Annotation/tiles \
  --out-dir results/bubble_chamber/s20_real_unlabeled \
  --epochs 6 --seed 20261025 --train-per-tile 16 --val-per-tile 8

# Only after a HUMAN expert has returned a compatible S19 review JSON:
python scripts/research/bubble_chamber/s19_run_small_label_pipeline.py \
  --proposals /private/S19/proposals.json \
  --images /private/S19/tiles \
  --expert-review-json /private/S19/S19_H50Z_expert_reviews.json \
  --out-dir results/bubble_chamber/s20_real_sparse_finetune \
  --init-azunet results/bubble_chamber/s20_real_unlabeled/azunet/best.pt \
  --init-unet results/bubble_chamber/s20_real_unlabeled/unet/best.pt
```

This pretraining switch is explicitly optional. Compare scratch controls under exactly the same expert reviewed data and budgets; it may **hurt** true track detection. Do not publish source crops, full checkpoints or actual reviewer identities to public Git.

## 4. Active annotation queue for few-label Graph-LIRA

New `s20_review_priority.py` ranks S19's existing, source-hashed Frangi proposals (37 real photo candidates) and nearby candidate pairs (23 real geometry-based suggestions). It considers length, proximity and endpoint tangent alignment and balances source tiles/photos. **Ranking scores are heuristic and uncalibrated**, NOT likelihoods of the same particle, and no line/pair is automatically certified.

The updated offline workbench adds **Next line / Next pair** buttons without touching the canonical S19 source image or expert JSON schema. It supports the same old import/export, manual missing-line strokes and explicit track/artifact/same/different choices. The test checked the actual seven photo tiles with a browser; mock test decisions were discarded.

```bash
python scripts/research/bubble_chamber/s20_make_priority_workbench.py \
  --proposals /private/S19/proposals.json \
  --tiles /private/S19/tiles \
  --out-dir /private/S20_priority_review
```

## 5. Acceptance evidence and next scientific dependency

- New S20+S19 focused tests: 22 passed locally, including original model 1-epoch technical smoke on **simulated** source photos, strict source identity, no-label abstention, true checkpoint transfer, review provenance and unchanged expert JSON format.
- Local Chromium test passed: 7 original image crops opened; prioritized line was selected and a fictional test label exported/imported; the test JSON was deleted, so no actual human annotations were created.
- Current real H50Z review status: **0 expert confirmed lines, 0 certified artifacts, 0 human pairs**. No full-image real segmentation Dice, Graph-LIRA instance linking error or particle momentum results exist.
- Next prerequisite: an S19-compatible expert JSON with marks on BOTH source photos and some confirmed same/different fragment relationships, ideally plus independent films/runs. Only then measure whether S20 encoder initialization improves the original AZUNet vs same-budget scratch and U-Net under a photo-held-out sparse review protocol. Three-view and B-field calibration are additionally mandatory for physical momenta/masses.
