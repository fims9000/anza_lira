# S7 — Source-locked original AZUNet versus ordinary U-Net

The **only permitted code branch** is `research/bubble-chamber-graph-lira`.
S6 showed a preliminary synthetic stress gain for a *local reimplementation* of an AZConv2d forward path, while the plain U-Net won on ordinary synthetic test images. The S6 result is **not** a demonstrated real bubble-film advantage and no original model checkpoints were retained.

## Current S7 deliverable

`scripts/research/bubble_chamber/s7_original_retrain.py` imports the **original unmodified** `models/azconv.py` and `models/segmentation.py`, and checks the Git blob fingerprints **before** importing them. It refuses to train if the original files are absent or changed. Their expected Git blob SHA-1 values from branch `cca89a3` are:

- `models/azconv.py`: `f4217ada00024927506fa8ebe520aa1f68ae5854`
- `models/segmentation.py`: `0b971c117f3d56cc588ebade6991c65faceb8a2a`

It compares original `AZUNet` (local hyperbolic, fuzzy/anisotropic) and original `BaselineUNet` with the same widths, optimization steps, augmentations, and frame-disjoint generated samples. Full model outputs are preserved as CPU checkpoint weights after every epoch; the selected best checkpoint is updated based **only on validation loss**. Each model's decision threshold is also selected on validation data only, then frozen for clean and low-contrast/noisy test results. Delta Dice confidence intervals are obtained by **paired original-frame bootstrap**, not by treating thousands of pixels as independent observations.

The `procedural` source mode is a software smoke with fully generated curves. The `visualmetro` source mode uses genuine operator-measured 2-D track shapes, but its pixels and masks are **still synthetic**, and **all frames are from a single film (`070`)**. Neither qualifies as real photographed segmentation accuracy.

## Run

From the repo root:

```bash
python -m pip install torch numpy opencv-python-headless pytest
python -m pytest -q tests/test_s7_original_retrain.py
python scripts/research/bubble_chamber/s7_original_retrain.py \
  --source procedural --out-dir results/bubble_chamber/s7_smoke \
  --size 32 --epochs 1 --train-limit 8 --val-limit 4 --test-limit 6 \
  --widths 4 4 4 4 --rules 2 --batch-size 2 --steps-per-epoch 2
```

Using local user-provided `1mH2Data.7z` unpacked at `.../1mH2Data/S1`:

```bash
python scripts/research/bubble_chamber/s7_original_retrain.py \
  --source visualmetro --measurements /path/to/1mH2Data/S1 \
  --out-dir results/bubble_chamber/s7_real_geometry_synthetic_pixels \
  --epochs 6 --size 128 --batch-size 4 --widths 8 16 24 32 \
  --rules 4 --train-limit 0 --val-limit 0 --test-limit 0
```

`train-limit/val-limit/test-limit=0` means unlimited in `visualmetro` mode. Full original local-hyperbolic AZConvs may require considerable GPU memory/compute; adjust batch size before modifying the model. The CI workflow is a small **source-locked procedural smoke**, not a full experiment and not proof of a scientific claim.

## Recorded evidence and limitations

- Data integrity audit was repeated locally on user archive: 145 complete frames, 1,073 tracks, 17,321 spatial measurements, 94/26/25 frame-disjoint split, source manifest SHA-256 `bd8e28247c5d178ae4f2f5a86215cbf5626f3ad86560928f4521093ce9188dea`.
- Local S7 test contracts: source lock (including mutation detection), generator determinism, stress determinism, frame-level bootstrap, atomic checkpoint creation, source lock refusing missing files.
- The current local container has CPU-only Torch and **does not contain** the original model source checkout. Local S7 network training must **not** be claimed successful until the exact-source workflow or another verified complete checkout runs.
- **The previous S6 local port source/checkpoint was not retained.** The original-source run validates the original module, not point-by-point equivalence with an unavailable port. Source/forward equivalence with that port remains **unverifiable**.
- To prove the model is better on actual H2/propane film, obtain confirmed per-pixel contours on *independent real films* and perform genuine held-out run tests. Synthetic stress improvements alone cannot establish this.

## Acceptance gates

1. Confirm exact-source Git blob hashes before runtime.
2. Run unmodified original AZUNet/BaselineUNet, save weights and metrics.
3. Independently rerun with multiple initialization seeds and matched training budgets on real measured-shaped synthetic pixels.
4. Evaluate precision, recall, Dice, false connections, clDice and runtime with paired frame confidence intervals.
5. Hold out independently photographed and physically reviewed films for final scientific superiority claim.
