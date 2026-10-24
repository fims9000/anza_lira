# S13: original AZConv as a bounded, learned guidance channel for U-Net

**24 October 2026 · exploratory result: no AZ-specific gain.** Branch `research/bubble-chamber-graph-lira`.

## Experiment

S10–S12 never established repeatable original AZConv superiority on bubble film; S11 topology loss weight 0.20 degraded performance, and S12 tangent direction supervision increased geometry gradient but improved only clean synthetic masks a little. S13 isolates whether a small **learned AZConv image prior** can help the strong convolutional U-Net rather than replacing its encoder.

Four arms are trained with unchanged original `models/azconv.py` and `models/segmentation.py` (Git blob SHA1 `f4217ada00024927506fa8ebe520aa1f68ae5854` and `0b971c117f3d56cc588ebade6991c65faceb8a2a`):

1. Ordinary U-Net without guidance.
2. U-Net with additive guidance from an ordinary 3×3 Conv2d, a model-capacity control.
3. U-Net with additive guidance from authentic original local-hyperbolic AZConv2d, bounded by a learned sigmoid gate.
4. Standard original AZUNet with input residual (from S10).

All U-Net backbones have precisely matched initialization weights for a given seed; the guidance branch starts near zero, and the gate starts at 0.20. Model initialization seeds 20261024, 20261025 and 20261026. Fixed data seed 20261031, 8 epochs, 120 train, 26 validation, 25 test source-frame-disjoint samples. Same real expert-measured film-070 **geometry**, but every grayscale pixel, track thickness, nuisance artifact and dense mask was synthetically rendered. The source archive and real photos were not committed. Confidence intervals below come from paired whole-test-frame resampling, with all 3 trained seeds kept per sampled frame; they are conditional on the single film and selected model seeds.

## Measured results (generated-pixel masks)

| Model | Mean clean Dice | Mean stress Dice | Mean stress clDice |
|---|---:|---:|---:|
| Ordinary U-Net | **0.7509** | **0.5352** | See machine report |
| Conv-guided U-Net | 0.7125 | 0.5269 | See machine report |
| AZConv-guided U-Net | 0.7083 | 0.5340 | See machine report |
| Original AZUNet | 0.7535 | 0.4992 | See machine report |

Pooled paired micro-Dice contrasts, AZConv-guided minus:

- Ordinary U-Net clean: **−0.0429**, 95% frame-boot CI [−0.0584, −0.0277]. Worse.
- Ordinary-conv-guided U-Net clean: **−0.0045**, CI [−0.0098, +0.0006]. No AZ-specific gain.
- Ordinary U-Net stress: **−0.0066**, CI [−0.0207, +0.0095]. Inconclusive.
- Ordinary-conv-guided U-Net stress: **−0.0007**, CI [−0.0062, +0.0050]. Inconclusive.

**Conclusion:** a shallow additive AZConv guidance path did not outperform a matched conventional guidance branch or a bare U-Net under this synthetic evaluation. A learned gate near 0.20 did not in itself provide useful incremental information. Do not deploy the branch or claim ANZA is more accurate on real photos from this study. The baseline comparison remains strong.

## Additional result relevant to physicists

The S12 safe pixel-space quadratic fitting study accepted only 55/75 ordinary U-Net and 47/75 original AZUNet clean synthetic masks. In severe synthetic noise only **5/75** U-Net masks and **4/75** AZUNet masks passed connected, nondegenerate, near-quadratic fit checks; direct orientation AZ accepted 2/75. Therefore image segmentation quality is not sufficient for the requested full-track reconstruction: uncertain links and false merges need explicit topology-safe LIRA relation assessment and human review. A fitted 2D parabola is not a 3D particle trajectory and cannot recover momentum, charge or mass without cross-view optical reconstruction, fiducial calibration and magnetic field.

## Replication

```bash
python scripts/research/bubble_chamber/s13_train_guided.py \
  --measurements /private/path/1mH2Data/S1 --out-dir results/s13_seed20261024 \
  --seed 20261024 --data-seed 20261031 --epochs 8 --size 64 \
  --batch-size 4 --train-limit 120 --val-limit 26 --test-limit 25
python scripts/research/bubble_chamber/s13_collect.py \
  --metrics results/s13_seed20261024/s13_metrics.json \
            results/s13_seed20261025/s13_metrics.json \
            results/s13_seed20261026/s13_metrics.json \
  --out results/s13_fixed_frame_multiseed.json
```

All original model weight files and source code are retained in the private research ZIP, and the aggregate `docs/bubble_chamber/s13_fixed_frame_multiseed.json` is safe to commit. This report does **not** contain real dense pixel segmentation labels or true connected-track instance labels. Further defensible progress requires testing LIRA against visually verified links and no unsafe automatic merges, and obtaining real expert-reviewed BMP masks from multiple independent film/runs.
