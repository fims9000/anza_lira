---
name: segmentation-experiment
description: Integrate GeoCrack training, fair architecture comparisons, smoke runs, checkpoints, and resumable run metadata.
---

# Segmentation Experiment

Use the current `train.py`, `utils.py`, and model factory. Keep split,
augmentation, optimizer, loss, epochs, batch size, checkpoint rule, threshold
grid, and evaluation identical across models. Run the 32/16/16 one-epoch
baseline/AZ vertical smoke before the nine-run study. Select thresholds on val.
Store config/commit/split hashes, checkpoints, metrics, status, and heartbeat.
