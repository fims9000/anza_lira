---
name: cracks-experiment
description: Run CRACKS crowd training, expert evaluation, limited-expert fine-tuning, and image-disjoint robustness without mixing the settings.
---

# CRACKS experiment

Use the frozen protocol and crowd targets under `results/anza_v2_study` and
`data/cracks/crowd_targets`. Setting A trains only on novice/practitioner fused
targets; expert masks provide no gradient or model selection. Call it
crowd-to-expert reconstruction on the same seismic sections, never unseen-image
generalization.

Train on deterministic 256x256 crops with 70% foreground-aware sampling. Run
validation and expert evaluation on padded 256x704 sections with overlap-tiled
inference when full inference is unsafe, then unpad to 255x701. All models share
the real BCE+Dice+clDice objective, optimizer, crop schedule, threshold
procedure, and evaluation code.

Setting B is five frozen folds of 28 expert train, 4 validation, and 8 test
sections, initialized from crowd-only checkpoints. Setting C excludes each held
out image, all of its annotations, and the frozen neighbor guard. Never combine
metrics across A, B, and C. Statistics use seismic sections, not pixels.

Structural replay is a named ablation. Real batches never receive synthetic
branch IDs. Preserve resume/config hashes and do not unlock expert scores until
crowd-only checkpoints and thresholds are frozen.
