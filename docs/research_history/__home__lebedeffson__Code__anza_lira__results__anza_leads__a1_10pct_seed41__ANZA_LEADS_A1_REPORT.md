# ANZA-LIRA LEADS V1 — A1 report

## Status

`STOP_ANZA_LABEL_EFFICIENCY_NO_SIGNAL`

This is a seed-41, 10%-optimization-section CRACKS development result. Thresholds were frozen on a separate calibration block. Expert annotations were not accessed.

| Variant | Threshold | Dice | Precision | Recall | AUPRC | clDice | Skeleton F1 | Fragmentation | Unknown-white FG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0_backbone | 0.95 | 0.5518 | 0.8107 | 0.5376 | 0.7878 | 0.5710 | 0.5831 | 0.2420 | 0.1388 |
| L1_isotropic | 0.95 | 0.4256 | 0.8258 | 0.4033 | 0.7919 | 0.4394 | 0.4792 | 0.3215 | 0.0802 |
| L2_generic_aniso | 0.95 | 0.4377 | 0.8209 | 0.4142 | 0.7928 | 0.4546 | 0.4889 | 0.3138 | 0.0842 |
| L3_anza_hs | 0.95 | 0.4742 | 0.8383 | 0.4526 | 0.7908 | 0.4912 | 0.5247 | 0.3137 | 0.0977 |

## Frozen L3 versus L2 gate

- Dice delta: `+0.036431` (required >= -0.005).
- clDice delta at the frozen precision constraint: `+0.036621` (required >= +0.015).
- Fragmentation ratio: `0.999822` (required <= 0.80 as the alternative topology gate).
- Unknown-white foreground ratio: `1.159515` (required <= 1.10).

The L3-L2 Dice and clDice gains are large positive seed-41 development diagnostics and their paired section-bootstrap intervals are above zero. They do not pass the predeclared result gate because the unknown-white safety ratio failed.

Calibration precision >=0.90 was infeasible for: `L0_backbone, L1_isotropic, L2_generic_aniso, L3_anza_hs`; the frozen rule therefore selected the highest-precision grid point (0.95) without development feedback.

Against the plain L0 backbone, L3 Dice is `-0.077585` and clDice is `-0.079734`. This is not the primary causal comparison, but it prevents an over-broad architecture claim.

## Compute

| Variant | Parameters | Peak GPU MiB | Wall seconds |
|---|---:|---:|---:|
| L0_backbone | 255160 | 304.6 | 46.4 |
| L1_isotropic | 285114 | n/a | n/a |
| L2_generic_aniso | 285146 | 356.3 | 51.4 |
| L3_anza_hs | 285114 | 356.2 | 49.5 |

## Claim boundary

No seeds 42/43, ANZA-MS, SSL, domain shift, OOF, expert evaluation, or LIRA continuation were opened. The decision above is the terminal state of this bounded A0-A1 run.
