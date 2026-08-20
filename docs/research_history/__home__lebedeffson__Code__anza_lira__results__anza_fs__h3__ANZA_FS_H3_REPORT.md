# ANZA-FS H3 report

## Status

`STOP_ANZA_FS_NO_PRACTICAL_STRUCTURAL_GAIN`

This is a frozen seed-41 synthetic development result on StressBench V6-HARD. It is not a confirm, CRACKS, multi-seed, H4, continuation, or expert result.

| Variant | Threshold | Branch recall | False bridges | Negative events | FBR | Dice | Precision | Recall | clDice | Fragmentation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F0_backbone | 0.950 | 0.9621 | 23 | 512 | 0.0449 | 0.8632 | 0.9662 | 0.7859 | 0.9558 | 0.1939 |
| F1_old_generic | 0.950 | 0.9825 | 22 | 512 | 0.0430 | 0.8905 | 0.9668 | 0.8273 | 0.9721 | 0.1150 |
| F2_free_foliation | 0.950 | 0.9815 | 30 | 512 | 0.0586 | 0.9029 | 0.9597 | 0.8544 | 0.9741 | 0.0674 |
| F3_anza_fs | 0.950 | 0.9791 | 32 | 512 | 0.0625 | 0.8830 | 0.9606 | 0.8199 | 0.9666 | 0.1155 |

## Frozen gates

- F3 vs F1: `{"dice_delta": -0.007498894096431563, "dice_noninferiority": false, "fbr_gate": false, "fbr_ratio": 1.4545454545454546, "paired_fbr_delta_ci": {"ci95_high": 0.03125, "ci95_low": 0.0078125, "mean_delta": 0.01953125, "resamples": 10000, "unit": "independent_scene"}}`
- F3 vs F2: `{"dice_delta": -0.019818626493661284, "dice_noninferiority": false, "fbr_gate": false, "fbr_ratio": 1.0666666666666667, "fragmentation_gate": false, "fragmentation_ratio_at_matched_dice": 1.472972972972973, "paired_fbr_delta_ci": {"ci95_high": 0.009765625, "ci95_low": 0.0, "mean_delta": 0.00390625, "resamples": 10000, "unit": "independent_scene"}}`

Thresholds were selected only on calibration. Development was evaluated once after threshold freeze. Confirm, CRACKS, expert, H4, and parameter alternatives remained closed.
