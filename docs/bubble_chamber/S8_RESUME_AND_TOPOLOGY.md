# S8 — Deterministic resume, dataset fingerprints, and topology readout

**Scope:** engineering reproducibility and evaluation instrumentation for the **exact original** `models/azconv.py` and `models/segmentation.py` already locked in S7. We did **not** alter either model module or produce new real-photograph accuracy claims.

## Rationale

S6 trained a locally ported AZ operator, lacked preserved trained weights, and measured only synthetic images generated from physically measured click trajectories. S7 added a source-locked training script. S8 closes the next technical gap: resuming an incomplete experiment without silently changing the training examples or optimizer state, and evaluating line connectivity rather than merely raw pixel Dice.

## Resume contract

- `last.pt` and `best.pt` are required for `--resume`. `epochs` denotes the **total desired number** of epochs, not additional epochs.
- Every checkpoint stores the original model identifiers, RNG state, optimizer state, model state, dataset SHA-256 fingerprints (train/val/test), and all training hyperparameters.
- Resuming permits only an increase in total epochs. All other configuration fields and all training/validation/**test** dataset fingerprints must match.
- The model's `best.pt` is chosen by **validation loss alone**; the decision threshold is chosen by **validation Dice alone**. Test labels are not used to select the model or threshold.
- The two architectures are independently seeded with the same seed before initialization and trained with matched optimizer settings and batch ordering. This does not equalize parameter count or wall time; both are reported.
- A small PyTorch `Conv2d` test demonstrated **bitwise identical** final weights and losses for two uninterrupted epochs versus 1+resume-to-2. This test does **not** demonstrate original AZConv GPU determinism.

## Topology readout

`bubble_chamber/neural_metrics.py` computes binarized clDice, 2-pixel-tolerant skeleton precision/recall/F1, and 8-connected component counts. These are **additional metrics**, not a replacement for Dice or expert physical track identity. In particular, **component counts are not false-bridge counts**: erroneously merging two distinct tracks can decrease component count. Actual false-join rate needs instance/track correspondence ground truth.

## Verified locally

```bash
python -m pytest -q tests/test_s7_original_retrain.py tests/test_bubble_neural_metrics.py
# 14 passed, in the local CPU container; toy run and data contracts only
```

The measured `1mH2Data.7z` click archive was successfully parsed in the local CPU environment into 94/26/25 frame-disjoint splits with its previous audit checksum. A small sample was generated and checked. **Full training of the untouched original AZUNet has not run in this container because the exact model source files were unavailable locally**; the remote GitHub branch retains those unmodified files, and S7/S8 CI scripts refer to them.

## Reproduction on a complete checkout

```bash
python scripts/research/bubble_chamber/s7_original_retrain.py \
  --source visualmetro --measurements /data/1mH2Data/S1 \
  --out-dir results/bubble_chamber/s8_original \
  --epochs 2 --size 64 --widths 8 16 24 32 --rules 2 \
  --train-limit 96 --val-limit 32 --test-limit 48

# Same data, same hyperparameters, extend to 6 total epochs:
python scripts/research/bubble_chamber/s7_original_retrain.py \
  --source visualmetro --measurements /data/1mH2Data/S1 \
  --out-dir results/bubble_chamber/s8_original \
  --epochs 6 --size 64 --widths 8 16 24 32 --rules 2 \
  --train-limit 96 --val-limit 32 --test-limit 48 --resume
```

CI workflow uses the tiny procedural smoke and then resumes to two total epochs. It must not be reported as a successful original-model run unless GitHub Actions status, logs, and checkpoint artifacts confirm it. Any real-film superiority claim requires real experts' instance masks and run/film-held-out evaluation; the current 145 frame dataset covers only film 070 and contains no matched photo pixels.
