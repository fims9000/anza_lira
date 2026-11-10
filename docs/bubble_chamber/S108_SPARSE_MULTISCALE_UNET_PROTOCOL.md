# S108: multi-scale sparse-label track candidate ranker (not calibrated)

S108 is a small multi-scale U-Net prototype introduced after S107 created
a source-locked human visual trajectory review workflow. The network has two
image-only input channels: native dark contrast and a background-normalized
local contrast response. It never consumes the pre-existing S105 color
preview as label truth or as a model input.

## Supervision is explicitly partial

Human annotation from S107 is rasterized as follows:

- 0: UNMARKED / UNKNOWN. Ignored by the loss.
- 1: reviewer-confirmed visible TRACK stroke. Positive.
- 2: reviewer-confirmed ARTIFACT stroke. Negative.
- 3: uncertain / conflicting stroke. Ignored by the loss.

Training with no confirmed positive and negative classes is rejected. The
verified original H50Z 005/006 BMP files belong to one physical film run,
so a holdout from them cannot demonstrate external generalization. S108
requires explicit acknowledgement of a same-run research PILOT.
No independent test metrics are reported from that pilot.

## Installation and guarded usage

Install PyTorch separately (CPU builds are sufficient for a small smoke).
Run S107 expert review first and save the JSON.

    python scripts/research/bubble_chamber/physicists_s108_sparse_train.py train \
      --pack-dir /PRIVATE/H50Z_S107_review \
      --review-json /PRIVATE/H50Z_S107_reviewed.json \
      --out-dir /PRIVATE/H50Z_S108_pilot \
      --one-run-pilot-only \
      --steps 50

Only after a real, manually confirmed review has BOTH track and artifact
strokes will this produce model weights. A new output directory is required
for every run to prevent accidental overwrites.

To inspect the SOURCE-LOCKED uncalibrated ranking on one original tile:

    python scripts/research/bubble_chamber/physicists_s108_sparse_train.py predict \
      --pack-dir /PRIVATE/H50Z_S107_review \
      --model-dir /PRIVATE/H50Z_S108_pilot \
      --tile-id 005_start3 \
      --out-dir /PRIVATE/H50Z_S108_preview

Predictions include full-resolution float score logits and a percentile-
normalized display PNG. These values are not probabilities that a physical
particle passed through an image pixel, and they are not track IDs.

## Acceptance gates before any scientific claim

1. Annotate actual source pixels by an independent expert before seeing
   S105 / S108 predictions. A repeated visual ID denotes a visually
   continuous line proposal, not a verified physical particle identity.
2. Acquire at least one new independent film/run and keep its labels
   completely unseen during training, threshold selection and code repair.
3. Compare S105 ridge, S108 sparse network and matched ordinary U-Net
   under identical acquisition and independent expert ground-truth.
4. Report positive centerline coverage AND artifact overlap, whole-trace
   fragmentation, cross-track false joins, weak spirals and per-frame
   uncertainty. Never divide one physical frame into multiple independent
   test samples.

Status 2026-11-10: S108 code and generated-only engineering tests are
available; no authentic expert-signed S107 track labels or independent
H50Z validation are available for a genuine model quality claim.
