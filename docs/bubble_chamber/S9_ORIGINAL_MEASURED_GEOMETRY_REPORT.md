# S9: original AZConv2d vs conventional U-Net and Frangi

**2026-10-24 · branch `research/bubble-chamber-graph-lira`**

## Result

Original source integrity verified: `models/azconv.py` blob `f4217ada00024927506fa8ebe520aa1f68ae5854`, `models/segmentation.py` blob `0b971c117f3d56cc588ebade6991c65faceb8a2a`. Existing AZUNet and BaselineUNet were trained using unchanged source files. AZ uses the already-supported `use_input_residual=True, residual_init=0.5` setting.

**This is an exploratory synthetic-pixel study, not validated real-film segmentation.** Input track shapes were extracted from 1mH2Data/visualMetro: 145 complete frames of film 070, 1,073 tracks and 17,321 reconstructed 3D points. Actual pixel masks, widths, contrast and noise were *simulated*. The sample splits are by whole frame, with no independent film/run holdout.

Three seeds, eight epochs, same train-time degradation for both neural nets, widths (4,6,8,12), two AZ rules. Each run uses 120 rendered train examples (60 frames), 26 validation frames and 25 test frames; seed changes the split, augmentation and model initialization. The Frangi baseline uses scales 1/2/3 and a validation-only threshold optimized for clean and stressed Dice.

## Test Dice (generated image masks)

| Seed | U-Net clean | AZ residual clean | Frangi clean | U-Net stress | AZ residual stress | Frangi stress |
|---|---:|---:|---:|---:|---:|---:|
| 20261024 | 0.6645 | 0.6267 | 0.8267 | 0.4272 | 0.4218 | 0.1181 |
| 20261025 | 0.4047 | 0.6810 | 0.8839 | 0.3307 | 0.5290 | 0.1377 |
| 20261026 | 0.7953 | 0.5893 | 0.7876 | 0.5151 | 0.4346 | 0.1281 |

The clean arithmetic means are U-Net 0.6215, AZ 0.6324, Frangi 0.8327. Under *simulated* contrast reduction 0.25 plus Gaussian noise SD 0.08, means are U-Net 0.4244, AZ 0.4618 and Frangi 0.1280. **The means are descriptive and not inferential proof.** AZ wins stress Dice in only one of three runs. Same-film validation cannot demonstrate transfer to propane vs hydrogen or to other apparatus settings.

Original pixel coordinates and 3D track measurements are not pixel-wise segmentation truth. The H50Z photos have only sparse seed/fiducial labels and are used solely to inspect unlabeled qualitative outputs. Estimated mass, momentum and particle identity were not computed.

## Methodological controls

- Exact Git blob SHA gate before training; original `AZConv2d` source left unchanged.
- Frame grouping prevents train/test leakage between crops or views; at least 25 separate test frames.
- Model checkpoint persists weights, optimizer, dataset SHA256 and validation threshold; S8 resume tests passed.
- Probability thresholds tuned only on validation, including the degraded validation condition when robust mode is enabled.
- Three run seeds conflate initialisation and frame split variation. Do not call the seed-to-seed variation isolated model instability.
- LIRA linking is a distinct measured-geometry experiment (S5), not interchangeable with AZConv segmentation metrics.

## Reproduction

Run `s7_original_retrain.py` with `--source visualmetro --robust-train --az-residual --size 64 --epochs 8 --batch-size 4 --train-limit 120 --val-limit 26 --test-limit 25 --widths 4 6 8 12 --rules 2 --seed <seed> --measurements <data>/1mH2Data/S1`. Use seeds 20261024, 20261025, 20261026. The full archive must stay private unless redistribution permissions are established.

## Remaining external validation

Need expert-confirmed per-pixel track masks on true images, paired 070 triplet BMP photos, and a separate film and filling to test cross-domain transfer. These do not exist in the supplied materials. **Do not claim proven real-camera ANZA superiority.**
