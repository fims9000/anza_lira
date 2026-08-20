# ANZA-KS K2 Seed-41 Report

Status: `STOP_ANZA_KS_FEATURE_NOT_TRANSFERRED`

This is a frozen seed-41 synthetic segmentation-transfer result. K2 confirm, seeds 42/43, CRACKS, and expert data remained closed.

K1.5 attribution: `SYMBOLIC_INFORMATION_PASS_ANOSOV_NOT_SPECIFIC`. Therefore an Anosov-specific claim is restricted unless M4 causally beats M2.

| Variant | Params | Known GFLOPs | Peak MB | Dice | clDice | Fragmentation | Mechanism TPR | Mechanism FPR | False/Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0_backbone | 249384 | 0.627 | 195.8 | 0.8365 | 0.8626 | 0.4382 | 0.8906 | 0.0010 | 1/1024 |
| M1_static | 254531 | 0.790 | 402.0 | 0.8488 | 0.8739 | 0.5102 | 0.9121 | 0.0010 | 1/1024 |
| M2_shear_ks | 254531 | 0.787 | 1270.4 | 0.8492 | 0.8731 | 0.5182 | 0.9033 | 0.0010 | 1/1024 |
| M3_cat_raw | 254531 | 0.777 | 655.6 | 0.8469 | 0.8712 | 0.4135 | 0.9141 | 0.0020 | 2/1024 |
| M4_anza_ks | 254531 | 0.787 | 1270.4 | 0.8577 | 0.8840 | 0.4284 | 0.9141 | 0.0039 | 4/1024 |

Known GFLOPs are the operations recognized by `torch.profiler` for a single 96x96 forward pass. Custom symbolic scatter/entropy/permutation kernels are not fully counted; batch-8 peak memory is the more complete resource comparison for those fixed operators.

## Frozen gates

`{"anosov_M4_vs_M2": false, "anosov_positive_ci": false, "kolmogorov_M4_vs_M3": false, "mechanism_M1": false, "natural_topology_M1": true, "pixel_safety_M1": true, "practical_M4_vs_M1": false}`

## Claim boundary

Synthetic feature and segmentation results do not establish real seismic improvement. No threshold, architecture, split, or feature family was changed after development results.
