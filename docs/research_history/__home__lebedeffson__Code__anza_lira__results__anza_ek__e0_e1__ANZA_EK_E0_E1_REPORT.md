# ANZA-EK E0/E1 report

## Status

`STOP_ERGODIC_ANOSOV_LOCAL_FEATURE_NO_MECHANISM`

This is a zero-training mathematical and causal feature audit. It is not a learned segmentation, CRACKS, confirm, or expert result.

## E0

- Mathematical status: `ANZA_EK_E0_PASS`
- Bilinear-grid L2 relative error: `4.3626191e-05`
- Bilinear-grid integral error: `1.6728398e-17`
- Exact finite permutation inverse error: `0`

## E1 task metrics

| Method | Task | Ranking | AUROC | TPR@FPR05 | Fisher | Perturbed ranking | Stability corr. |
|---|---|---:|---:|---:|---:|---:|---:|
| E1_0_isotropic | straight_ridge_vs_blob | 0.0000 | 0.0000 | 0.0000 | -16.1275 | 0.0000 | 0.9893 |
| E1_0_isotropic | faint_visible_continuation | 0.0234 | 0.0448 | 0.0000 | -2.4199 | 0.1719 | 0.6961 |
| E1_0_isotropic | crossing_correct_vs_wrong | 0.0000 | 0.0035 | 0.0000 | -3.0575 | 0.1602 | 0.5958 |
| E1_0_isotropic | close_parallel_separation | 0.0000 | 0.0000 | 0.0000 | -50.6602 | 0.0000 | 0.9991 |
| E1_0_isotropic | curved_local_ridge | 0.0000 | 0.0000 | 0.0000 | -13.1024 | 0.0000 | 0.9380 |
| E1_0_isotropic | oriented_clutter | 0.0586 | 0.2054 | 0.0000 | -1.1174 | 0.0742 | 0.9591 |
| E1_1_static_anisotropic | straight_ridge_vs_blob | 1.0000 | 1.0000 | 1.0000 | 65.4855 | 1.0000 | 0.9972 |
| E1_1_static_anisotropic | faint_visible_continuation | 1.0000 | 1.0000 | 1.0000 | 38.7123 | 1.0000 | 0.9931 |
| E1_1_static_anisotropic | crossing_correct_vs_wrong | 1.0000 | 1.0000 | 1.0000 | 8.5443 | 1.0000 | 0.9406 |
| E1_1_static_anisotropic | close_parallel_separation | 1.0000 | 1.0000 | 1.0000 | 66.4492 | 1.0000 | 0.9956 |
| E1_1_static_anisotropic | curved_local_ridge | 1.0000 | 1.0000 | 1.0000 | 10.7585 | 1.0000 | 0.9608 |
| E1_1_static_anisotropic | oriented_clutter | 0.9922 | 0.9957 | 0.9805 | 4.5384 | 0.9961 | 0.9860 |
| E1_2_shear_koopman | straight_ridge_vs_blob | 1.0000 | 1.0000 | 1.0000 | 6.9545 | 1.0000 | 0.9900 |
| E1_2_shear_koopman | faint_visible_continuation | 0.0000 | 0.0000 | 0.0000 | -21.3404 | 0.0000 | 0.9489 |
| E1_2_shear_koopman | crossing_correct_vs_wrong | 1.0000 | 0.9933 | 0.9766 | 4.2406 | 0.7109 | 0.3674 |
| E1_2_shear_koopman | close_parallel_separation | 1.0000 | 0.5862 | 0.0586 | 0.1468 | 0.6289 | 0.8840 |
| E1_2_shear_koopman | curved_local_ridge | 0.0000 | 0.0000 | 0.0000 | -28.3643 | 0.0000 | 0.9836 |
| E1_2_shear_koopman | oriented_clutter | 0.3633 | 0.4198 | 0.0039 | -0.3480 | 0.3906 | 0.9573 |
| E1_3_cat_koopman | straight_ridge_vs_blob | 1.0000 | 1.0000 | 1.0000 | 55.1841 | 1.0000 | 0.9983 |
| E1_3_cat_koopman | faint_visible_continuation | 1.0000 | 1.0000 | 1.0000 | 30.4563 | 1.0000 | 0.9853 |
| E1_3_cat_koopman | crossing_correct_vs_wrong | 1.0000 | 1.0000 | 1.0000 | 7.8913 | 1.0000 | 0.9379 |
| E1_3_cat_koopman | close_parallel_separation | 1.0000 | 1.0000 | 1.0000 | 60.9780 | 1.0000 | 0.9957 |
| E1_3_cat_koopman | curved_local_ridge | 1.0000 | 1.0000 | 1.0000 | 10.6918 | 1.0000 | 0.9674 |
| E1_3_cat_koopman | oriented_clutter | 0.9961 | 0.9980 | 0.9922 | 4.4859 | 0.9961 | 0.9868 |

## Frozen causal gate

Strongest control: `E1_1_static_anisotropic`.
Passing identifiable tasks: `0` / `6`; required >=2.
Safety checks: `{"macro_clean_ranking": true, "macro_perturbation_correlation": true, "macro_perturbed_ranking": true}`.

No classifier, network training, conjugacy, E2, confirm, CRACKS, or expert data were opened.
