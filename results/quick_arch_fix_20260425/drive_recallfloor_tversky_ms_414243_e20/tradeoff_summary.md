# Recall-Floor + Tversky Trial (Negative Trade-off)

Date: 2026-04-25  
Run: `results/quick_arch_fix_20260425/drive_recallfloor_tversky_ms_414243_e20/`
Reference: `results/quick_arch_fix_20260425/drive_implfix_policyfix_ms_414243_e20/`

Configuration tested (AZ only):
- `overlap_mode: tversky`
- `tversky_alpha: 0.40`
- `tversky_beta: 0.60`
- `eval_threshold_min_recall: 0.76`
- existing threshold policy guards preserved (`reference=0.80`, tolerance `0.005`, max `0.85`).

## Mean effect vs previous policyfix

| Metric | Previous policyfix | Recall-floor + Tversky | Delta |
|---|---:|---:|---:|
| Dice | 0.7498 | 0.7377 | -0.0121 |
| Recall | 0.7220 | 0.7449 | +0.0229 |
| Balanced Acc | 0.8510 | 0.8589 | +0.0079 |
| Precision | 0.7799 | 0.7388 | -0.0411 |

## Interpretation

- The intervention improved recall and balanced accuracy.
- However, it produced a substantial Dice drop and reduced precision.
- Threshold stability worsened (`selected_threshold` no longer fixed at 0.80; one seed selected 0.60).

Decision:
- keep this trial as a negative/diagnostic result;
- do **not** replace the current best policyfix configuration with this setting.
