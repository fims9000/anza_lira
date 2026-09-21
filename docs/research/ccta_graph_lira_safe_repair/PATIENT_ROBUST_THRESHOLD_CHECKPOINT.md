# Patient-robust threshold checkpoint before expanded CT

Date: 2026-09-21

The six-case CT experiment showed that pooled validation calibration can look safe while transferring unevenly between patients. Before seeing the 28-patient CT results, we therefore tested a secondary patient-robust threshold rule on the frozen geometry baselines.

## Policies

All thresholds are selected on the five validation patients only.

Primary existing rule:

`pooled FPR <= 5%`

Secondary robust rule:

`pooled FPR <= 5% AND max validation-patient FPR <= 5%`

Two diagnostic rules were also evaluated:

- pooled <=5% and max-patient <=10%;
- zero validation false positives.

The robust rule is not a replacement for the primary pooled operating point. It is a pre-registered secondary safety operating point for the upcoming CT evidence.

## Geometry-HGB result

Primary pooled rule:

- validation recall 32.09%, pooled FPR 4.48%;
- held-out test recall 37.72%, pooled FPR 1.20%;
- maximum held-out patient FPR 4.35%.

Patient-robust rule:

- validation recall 30.60%, pooled FPR 3.73%, maximum validation-patient FPR 5.00%;
- held-out test recall 32.93%;
- held-out pooled FPR 0.60%;
- maximum held-out patient FPR 2.94%;
- precision 98.21%.

Thus, on geometry-HGB, a modest recall reduction buys a lower and more even false-acceptance rate across held-out patients.

## OGMC-style result

Primary:

- test recall 39.52%, FPR 3.59%, max-patient FPR 8.70%.

Patient-robust:

- test recall 31.14%, FPR 2.99%, max-patient FPR 5.13%.

Again the tradeoff is visible.

## Consequence for CT

When the real 28-patient CCTA scores become available, report both:

1. the original pooled FPR<=5% operating point;
2. the secondary patient-robust pooled<=5% + max-val-patient<=5% operating point.

Both thresholds must be selected from validation only.

If radial CT looks excellent only under a pooled threshold but collapses under the patient-robust rule, that is evidence of patient-specific calibration instability.

If radial CT keeps a useful recall advantage under both rules, that is stronger evidence that the signal transfers across hearts.

## No test retuning

The held-out six test patients are evaluation only. The patient-robust rule is fully specified now, before real expanded CT features are available.

Exact script:
`scripts/research/ccta_graph_lira_safe_repair/evaluate_patient_robust_thresholds.py.gz.b64`

Source SHA256:
`95d25cddd4aed7fa4714a4cf834ff2d3d3f5156171489ece68d7cd9ac2ab7bdb`
