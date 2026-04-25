# Re-threshold Summary (No Retraining)

Date: 2026-04-25  
Run pack: `results/quick_arch_fix_20260425/drive_implfix_best_ms_414243_e20/`

Protocol:
- Checkpoints are unchanged (`checkpoint_best.pt` from existing 20-epoch runs).
- Only test-time threshold is re-evaluated at `thr=0.80`.
- Source files: `az_thesis_seed*/metrics_test_thr080.json`.

## Per-seed AZ metrics (thr=0.80)

| Seed | Dice | IoU | Precision | Recall | Balanced Acc |
|---:|---:|---:|---:|---:|---:|
| 41 | 0.7502 | 0.6002 | 0.7843 | 0.7189 | 0.8498 |
| 42 | 0.7515 | 0.6020 | 0.7842 | 0.7215 | 0.8510 |
| 43 | 0.7478 | 0.5972 | 0.7713 | 0.7256 | 0.8523 |

## Mean comparison: original selected threshold vs fixed 0.80

| Metric | Original mean (auto-selected thr) | Re-threshold mean (thr=0.80) | Delta |
|---|---:|---:|---:|
| Dice | 0.7403 | 0.7498 | +0.0095 |
| IoU | 0.5878 | 0.5998 | +0.0120 |
| Precision | 0.8139 | 0.7799 | -0.0340 |
| Recall | 0.6816 | 0.7220 | +0.0404 |
| Balanced Acc | 0.8330 | 0.8510 | +0.0180 |

Baseline reference from same run pack:
- Baseline mean Dice: `0.7456`
- AZ (thr=0.80) mean Dice: `0.7498`
- Delta Dice vs baseline: `+0.0043`

Interpretation:
- The large degradation in the original impl-fix report was primarily due to threshold calibration drift (too high selected thresholds).
- With conservative thresholding, the impl-fix checkpoints recover recall and become competitive again without retraining.
