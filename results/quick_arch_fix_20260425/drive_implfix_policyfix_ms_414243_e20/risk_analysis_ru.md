# Risk-Focused Analysis (DRIVE, 20e, seeds 41/42/43)

Date: 2026-04-25  
Run: `results/quick_arch_fix_20260425/drive_implfix_policyfix_ms_414243_e20/`

## What improved

- AZ mean Dice: `0.7498` vs baseline `0.7456` (`+0.0043`).
- Threshold policy fix stabilized selected threshold at `0.80` (all 3 seeds).

## Main weaknesses (critical)

1. Recall drop relative to baseline remains systematic.
- Baseline recall mean: `0.7540`
- AZ recall mean: `0.7220`
- Delta recall: `-0.0320`

2. Balanced Accuracy is lower than baseline.
- Baseline balanced acc mean: `0.8638`
- AZ balanced acc mean: `0.8510`
- Delta: `-0.0128`

3. Latency/cost is much higher.
- Mean forward time: AZ `0.213s` vs baseline `0.068s`
- Runtime multiplier: about `3.14x`

4. Statistical confidence is still weak with only 3 seeds.
- Paired Dice delta mean: `+0.00427`
- Approx 95% CI (n=3): `[-0.01646, +0.02501]`
- Current gain can be real, but is not yet robustly significant.

5. Architecture control knobs are effectively static.
- `hybrid_mix` stays near init (`~0.10`) across seeds.
- AZ input residual alpha stays near init (`~0.15`) across seeds.
- This indicates low adaptive utilization of these intended mechanisms.

6. Threshold sensitivity still exists.
- Unconstrained best validation threshold is often `0.85` or `0.90`.
- Using `0.80` avoids recall collapse but slightly sacrifices `core_mean` on validation (`~0.0028..0.0060`).

## Manuscript-safe interpretation

- The implementation-corrected AZ variant is competitive on DRIVE under calibrated thresholding.
- The current tradeoff is precision/specificity up vs recall/balanced-accuracy down.
- Claims should not present universal superiority until cross-dataset and stronger multi-seed evidence are available.

## High-value next fixes (minimal branching)

1. Add recall floor into threshold policy (e.g., `eval_threshold_min_recall`) and keep max-threshold guard.
2. Improve recall at training objective level (small Tversky/BCE rebalance) while preserving current calibration policy.
3. Validate on at least one additional dataset (CHASE or FIVES) with the same protocol before stronger claims.

## Addendum (2026-04-25): recall-floor + Tversky frontier check

A targeted recall-oriented trial (`drive_recallfloor_tversky_ms_414243_e20`) confirms the trade-off:
- recall can be lifted,
- but Dice drops noticeably and threshold stability degrades.

This reinforces that the current best production candidate remains the pure policyfix setup (without recall-floor forcing and without recall-heavy Tversky shift).
