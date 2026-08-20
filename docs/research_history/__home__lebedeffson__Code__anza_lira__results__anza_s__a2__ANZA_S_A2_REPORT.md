# ANZA-S Phase A2 causal Anosov audit

## Research status

`ANOSOV_COCYCLE_REDUNDANT_AT_ORACLE`

This is a zero-training oracle audit. It does not report a learned model or a CRACKS result.

## Identifiable validation tasks

| Method | Macro TPR | Macro FPR | Macro ranking | Macro pAUC@0.05 |
|---|---:|---:|---:|---:|
| A0_tangent_terminal | 0.6667 | 0.0125 | 0.6667 | 0.6667 |
| A1_isotropic_shadowing | 0.9083 | 0.0125 | 0.9863 | 0.9083 |
| A2_local_anisotropic_reset | 1.0000 | 0.0125 | 1.0000 | 1.0000 |
| A3_cocycle_cg_lambda0 | 0.9083 | 0.0125 | 0.9863 | 0.9083 |
| A3_cocycle_cg_lambda035 | 1.0000 | 0.0167 | 1.0000 | 1.0000 |

## Causal decision

Frozen gates: `{'x_gain_or_ceiling': True, 'macro_tpr_gain_at_least_0_08': False, 'macro_ranking_improves': False, 'parallel_safety': True, 'paired_ci_lower_above_zero': False, 'lambda_0_35_non_inert': True}`.

Paired macro TPR delta A3-A2: `0.0000` (95% CI `0.0000` to `0.0000`).

Lambda intervention: `{'task_tpr_gains_lambda035_minus_lambda0': {'P1_x': 0.275, 'P2_parallel': 0.0, 'P3_curved': 0.0}, 'max_absolute_score_difference': 0.34925974194994064, 'hyperbolicity_inert': False}`.

## Controls

Curved-confuser comparability: `{'positive_count': 20, 'negative_count': 38, 'positive_median_endpoint_distance': 6.304850291940666, 'negative_median_endpoint_distance': 8.51680201771935, 'negative_to_positive_distance_ratio': 1.3508333462899453, 'positive_median_axial_agreement': 1.0, 'negative_median_axial_agreement': 0.9999999999992585, 'absolute_axial_agreement_difference': 7.415179581471421e-13, 'frozen_rule': 'distance ratio in [0.5,2.0] and axial-agreement difference <=0.25', 'primary_eligible': True}`.

Matched straight-gap leakage control: `{'A0_tangent_terminal': {'threshold': 0.36856660259088847, 'tpr_at_fpr_0_05': 0.0, 'fpr': 0.0, 'low_fpr_pauc_normalized': 0.025000000000000005, 'ranking_probability': 0.5, 'auroc': 0.5, 'role': 'leakage control only; excluded from gate'}, 'A1_isotropic_shadowing': {'threshold': 0.16272554166808287, 'tpr_at_fpr_0_05': 0.0, 'fpr': 0.0, 'low_fpr_pauc_normalized': 0.025000000000000005, 'ranking_probability': 0.5, 'auroc': 0.5, 'role': 'leakage control only; excluded from gate'}, 'A2_local_anisotropic_reset': {'threshold': 0.19454976262078308, 'tpr_at_fpr_0_05': 0.0, 'fpr': 0.0, 'low_fpr_pauc_normalized': 0.025000000000000005, 'ranking_probability': 0.5, 'auroc': 0.5, 'role': 'leakage control only; excluded from gate'}, 'A3_cocycle_cg_lambda0': {'threshold': 0.1627255416680828, 'tpr_at_fpr_0_05': 0.0, 'fpr': 0.0, 'low_fpr_pauc_normalized': 0.025000000000000005, 'ranking_probability': 0.5, 'auroc': 0.5, 'role': 'leakage control only; excluded from gate'}, 'A3_cocycle_cg_lambda035': {'threshold': 0.10976588942342891, 'tpr_at_fpr_0_05': 0.0, 'fpr': 0.0, 'low_fpr_pauc_normalized': 0.025000000000000005, 'ranking_probability': 0.5, 'auroc': 0.5, 'role': 'leakage control only; excluded from gate'}}`.

All A1/A2/A3 methods use the exact same tangent centerline. A2 resets the local ellipse; A3 alone composes covariance across steps. Therefore only A3-vs-A2 can support a causal cocycle claim.

No training, Phase B, confirm/test, CRACKS, or expert data were opened.
