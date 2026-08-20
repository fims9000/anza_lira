# ANZA-LIRA LEADS RC1 — Risk-calibrated frontier

## Status

`STOP_ANZA_LOW_LABEL_GAIN_WAS_OPERATING_POINT_SPECIFIC`

RC1 changed only the cross-fit sections, score-complete calibration frontier, and unsupported-white safety metric. The parent A1 STOP remains immutable. Expert annotations were not accessed.

| Variant | Threshold | Precision | Recall | Dice | clDice | AUPRC | Unsupported white |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0_backbone | 0.988424 | 0.9045 | 0.0293 | 0.0551 | 0.0595 | 0.7605 | 0.0020 |
| L2_generic_aniso | 0.967480 | 0.9338 | 0.2786 | 0.3966 | 0.4198 | 0.7858 | 0.0091 |
| L3_anza_hs | 0.970000 | 0.9352 | 0.2668 | 0.3835 | 0.4063 | 0.7842 | 0.0089 |

## Primary result

- L3-L2 Dice: `-0.013049`; paired section 95% CI `[-0.014564, -0.011565]`.
- L3-L2 clDice: `-0.013471`; paired section 95% CI `[-0.015515, -0.011508]`.
- L3-L2 AUPRC: `-0.001677`.
- Unsupported-white ratio L3/L2: `0.975414`; L3/L0: `4.538626`.

## Frozen causal checks

- `development_precision_L2`: `True`
- `development_precision_L3`: `True`
- `cldice_gain`: `False`
- `cldice_ci_positive`: `False`
- `dice_noninferior_L2`: `False`
- `cldice_noninferior_backbone`: `True`
- `dice_noninferior_backbone`: `True`
- `auprc_noninferior`: `True`
- `unsupported_white_vs_L2`: `True`
- `unsupported_white_vs_L0`: `False`

The topology-precision frontier is diagnostic only and cannot rescue the primary frozen operating-point gate.

## Claim boundary

No seeds 42/43, ANZA-MS, SSL, domain shift, LIRA continuation, OOF, or expert evaluation were opened. RC1 does not alter the negative parent A1 decision.
