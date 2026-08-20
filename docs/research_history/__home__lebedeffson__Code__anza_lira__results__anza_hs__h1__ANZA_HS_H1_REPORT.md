# ANZA-HS H1 report

## Status

`HYPERBOLIC_CONSTRAINT_NOT_INCREMENTAL`

This is a seed-41 synthetic development result on frozen StressBench V5. It is not a CRACKS, confirm, multi-seed, or expert result.

| Variant | Threshold | Dice | Precision | Recall | clDice | Fragmentation | Branch preservation | Parallel false connection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0_backbone | 0.20 | 0.9317 | 0.9265 | 0.9375 | 0.9852 | 0.0386 | 0.9982 | 0.0500 |
| B1_isotropic | 0.20 | 0.9265 | 0.9190 | 0.9352 | 0.9806 | 0.0364 | 0.9948 | 0.0364 |
| B2_generic_aniso | 0.55 | 0.9234 | 0.9314 | 0.9166 | 0.9828 | 0.0432 | 0.9951 | 0.0364 |
| B3_anza_hyperbolic | 0.35 | 0.9292 | 0.9276 | 0.9315 | 0.9844 | 0.0409 | 0.9972 | 0.0273 |

## Frozen B3 versus B2 gate

`{'dice_delta_B3_minus_B2': 0.0057924449068417205, 'cldice_delta_B3_minus_B2': 0.0015722430425147982, 'fragmentation_ratio_B3_over_B2': 0.9473684210526315, 'gate_checks': {'dice_noninferiority': True, 'cldice_gain': False, 'fragmentation_reduction': False}, 'matched_precision_target_from_B2_calibration': 0.9333666788315186, 'gate_precision_difference_B3_minus_B2': -0.003831744272690063}`

No lambda/M/base-scale alternative was used. H2, confirm, CRACKS, continuation, and expert data remained closed.
