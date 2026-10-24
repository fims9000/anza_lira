# S6: AZ-style anisotropic neural segmentation vs standard U-Net and Frangi

**Status: exploratory synthetic-only; NOT a proof of superiority on real bubble-camera film.**

## Data, protocol and source identity

- Input: archived `1mH2Data` visualMetro expert coordinates for film 070, **not paired BMP photographs**. Synthetic film intensities and thickened line masks were generated from measured track clicks.
- 145 audited frames, one film. Split by original frame: 94 training, 26 validation, 25 test; 2,172 synthetic training images, 362 validation images, 336 held-out synthetic test images (168 source tracks x two views). Three-view secondary test has 504 images.
- Model definitions correspond to `BaselineUNet` and `AZUNet(local_hyperbolic)` in the ANZA-LIRA repo, both at widths `(8,16,24,32)` and four AZ rules; ~57,041 and ~55,270 trainable parameters, respectively. Same generated training examples and four training epochs.
- **Critical code boundary:** due to network isolation, the completed CPU run used an **offline reduced implementation of the upstream local_hyperbolic AZConv forward equations**, omitting optional diagnostics/unused geometry modes. The exact original GitHub class was inspected but could not be directly executed locally. Forward/checkpoint parity with full upstream code remains unverified. Do not present these as byte-for-byte official AZConv results.
- Neural model decision thresholds were chosen on validation. Frangi response fraction was also selected only on validation. The test was not used to choose these thresholds.

## Observed results from the completed S6 run

| Synthetic test metric | Ordinary U-Net | AZ local | Frangi |
|---|---:|---:|---:|
| Original generator Dice | **0.960817** | 0.866758 | 0.841306 |
| Original generator precision | 0.948985 | 0.869169 | 0.839612 |
| Original generator recall | 0.972947 | 0.864360 | 0.843007 |
| Original centerline F1, tolerance 2px | **0.985490** | 0.915651 | 0.933897 |
| Decreased-contrast + noise Dice | 0.598924 | **0.677208** | 0.246434 |
| Stress centerline F1, tolerance 2px | 0.717499 | **0.750970** | 0.271283 |

- Main synthetic-test AZ minus plain U-Net Dice: **-0.094059**. Paired original-frame bootstrap 95% interval **[-0.103578, -0.085971]** (25 frames, 2,500 draws).
- Stress perturbation: contrast scaled to 0.43 relative to median; additive Gaussian noise σ=12 grayscale levels; weights and thresholds unchanged. AZ minus U-Net Dice: **+0.078285**, paired frame-bootstrap 95% interval **[+0.055634, +0.105639]**.
- Secondary all-three-view test with frozen neural thresholds: original U-Net **0.959124**, AZ **0.865805**; stress U-Net **0.577762**, AZ **0.669130** (504 test images, 25 frames).
- CPU exploratory port runtime, four epochs: approximately 53s standard U-Net and 291s AZ local. These timings do not include interpretation-cache overhead in the full upstream implementation and must not be cited as official original source benchmarks.

### Post-hoc perturbation sweep (three noise realizations per condition)

Contrast factor | U-Net mean Dice | AZ mean Dice | AZ-U-Net
:--|--:|--:|--:
1.00 | 0.889398 | 0.840710 | -0.048688
0.75 | 0.842937 | 0.819114 | -0.023823
0.60 | 0.775809 | 0.779187 | +0.003378
0.43 | 0.600792 | 0.677732 | +0.076940
0.30 | 0.316483 | 0.502972 | +0.186489

The sweep was investigated **after** the first result and is **exploratory, not confirmatory**. Three nuisance seeds are not three independent model training seeds; 25 frames all belong to the same film.

## Real H50Z qualitative check

Predictions were plotted on four 128x128 center crops from the original H50Z sample images (displayed at 2x). Neither network reliably separates true tracks from scratches/noise there; apparent false positives are plentiful. As **there are no expert pixel masks for those photographs**, there is no defensible real-image Dice, F1 or particle-identity accuracy to report.

## Integrity / reproducibility boundaries

The completed exploratory run produced weights and logs, but the temporary isolated container reset before they could be exported; **the original checkpoint files are not preserved in this commit or this package**. The numbers above are transcribed from the actual finished tool-produced metrics observed earlier in this conversation, not from newly re-run training. Associated figures were exported and survived the session reset. Code in `evaluate_s6.py` is a reconstructed reproducible evaluator intended to be run with **new** checkpoints created from the authentic full-repository trainer (`run_synthetic_anza_pretrain.py`). No exact-byte parity is asserted for the reconstructed evaluator or missing checkpoints.

## Reproduction in full repository

```bash
python scripts/research/bubble_chamber/run_synthetic_anza_pretrain.py --measurements /private/path/1mH2Data/S1 --out-dir results/bubble_chamber/s6_train --epochs 4 --batch-size 16 --size 64 --seed 20261024 --device auto
python scripts/research/bubble_chamber/evaluate_s6.py --measurements /private/path/1mH2Data/S1 --training-dir results/bubble_chamber/s6_train --out-dir results/bubble_chamber/s6_eval
```

The second run should start from exact upstream `models/azconv.py`; its results may differ from the exploratory offline port. Keep film/frame split and model selection frozen. Next genuine scientific gate: expert-confirmed track masks on real BMPs, independent film/run test and replicated seeds, including explicit false-bridge and graph continuity metrics. Do **not** treat synthetic a Dice advantage under one stress transform as physical-particle detection proof.
