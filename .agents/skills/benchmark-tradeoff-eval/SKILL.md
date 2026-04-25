---
name: benchmark-tradeoff-eval
description: Use for segmentation benchmark changes, metric-table updates, AZ-vs-baseline comparisons, sweeps, and manuscript result claims in this repo.
---

# Benchmark And Trade-Off Evaluation For `anza_lira`

Use this skill when:
- editing training/evaluation logic or result summaries;
- updating paper tables from saved `metrics.json`;
- comparing `baseline`, `attention_unet`, `az_cat`, `az_thesis`, or other AZ variants;
- choosing tuning policy for quality, complexity, interpretability, and speed trade-offs.

## Source Of Truth

- Use `results/**/metrics.json` as the primary source for numeric claims.
- Keep markdown summaries derived from explicit run bundles.
- Do not mix numbers from unrelated bundles unless the table is clearly labeled as exploratory.
- If a dataset/run/seed is missing, mark the comparison as incomplete.

Important existing summaries:
- `results/architecture_baseline_analysis_ru.md`
- `results/latest_results_summary_ru.md`
- `results/drive_real_comparison.md`

## Primary Metrics

For binary segmentation:
- primary quality: `test_dice`, `test_iou`;
- operating behavior: `test_precision`, `test_recall`, `test_balanced_accuracy`;
- threshold policy: `selected_threshold`, `threshold_selection_metric`;
- cost: `num_parameters`, `approx_gmacs_per_forward`, `seconds_per_forward_batch`.

For thin-structure tasks, never report Dice alone. Always inspect recall and precision:
- vessels/roads can look good by Dice while dropping weak branches;
- high precision with low recall means missing thin structures;
- balanced accuracy helps expose all-background or over-pruned behavior.

## Comparison Policy

- Always include the relevant baseline for the same dataset/seed/epoch policy.
- Prefer `mean +- std` across seeds for paper-facing claims.
- Use signed delta vs baseline with a clear sign convention.
- Keep claim language conservative when AZ is not a clear leader.
- If AZ wins Dice but loses recall/speed, state the trade-off directly.

Current evidence to preserve:
- old final-pack `az_thesis` lost to baseline across medical/ARCADE datasets;
- newer DRIVE implementation/policy fix is close and can slightly exceed baseline Dice;
- the gain is small and comes with lower recall and slower forward;
- CHASE policyfix is near parity, not a clean win.

## Sweep Strategy

1. Start with a targeted 1-seed smoke/probe.
2. Promote only promising configs to 3 seeds.
3. Promote final candidates to unified runs with baseline included.
4. Keep architecture, loss, threshold policy, and data split fixed inside each comparison bundle.

For AZ tuning, track:
- `az_geometry_mode`
- `az_normalize_mode`
- `az_compatibility_floor`
- `az_use_input_residual`
- `az_residual_init`
- `hybrid_mix_init`
- `encoder_az_stages`
- `bce_pos_weight`
- `overlap_mode`
- threshold selection policy

## Interpretation Policy

Mathematical correctness of AZ geometry is not the same as benchmark superiority.
When writing claims, separate:
- theoretical mechanism: fuzzy compatibility + anisotropic geometry-aware local aggregation;
- implementation behavior: normalization, residuals, thresholding, precision/recall trade-off;
- measured outcome: dataset-level metrics and speed.

Strong claim requires:
- same dataset;
- same split policy;
- same seed set or enough repeated runs;
- baseline included;
- metrics and cost reported.

## Reproducibility Outputs

For important runs, save or preserve:
- `metrics.json`
- `history.json`
- `checkpoint_best.pt`
- config used for the run
- summary markdown for decisions/paper claims

For background or long runs:
- record PID and log path;
- do not claim completion until file counts or final logs confirm it.
