# Selective-repair risk uncertainty

Date: 2026-09-20.

Status: descriptive finite-sample analysis of the canonical V2 controlled cross-patient benchmark.

## Why zero observed false links is not zero risk

Perturbation consistency produces several operating points with zero observed false structural links among accepted scenes. With only tens or hundreds of accepted scenes, however, the underlying false-link probability can still be nonzero.

For each threshold we therefore compute an exact one-sided 95% binomial upper bound on the false-link probability among accepted scenes.

This is only a descriptive scene-level bound. The generated scenes are clustered within a small number of anatomical branch points and only two patients are available, so it must not be reported as a clinical population confidence interval.

## Primary 30 deg + 1 mm results

### Train 921 -> test 953

| consistency | accepted | observed false | exact among accepted | one-sided 95% upper bound for false-link rate |
|---:|---:|---:|---:|---:|
| 0.60 | 199 | 0 | 96.48% | 1.49% |
| 0.80 | 168 | 0 | 98.21% | 1.77% |
| 0.90 | 137 | 0 | 99.27% | 2.16% |
| 0.95 | 93 | 0 | 100% | 3.17% |

### Train 953 -> test 921

| consistency | accepted | observed false | exact among accepted | one-sided 95% upper bound for false-link rate |
|---:|---:|---:|---:|---:|
| 0.60 | 132 | 0 | 93.18% | 2.24% |
| 0.80 | 102 | 0 | 98.04% | 2.89% |
| 0.90 | 67 | 0 | 98.51% | 4.37% |
| 0.95 | 50 | 0 | 100% | 5.82% |

At consistency `>= 0.50` in the reverse direction, 1 false structural scene is observed among 136 accepted scenes (0.735%); the exact one-sided 95% upper bound is about 3.44%.

## Sample-size implication

If a future independent confirm cohort observes **zero** false links, the approximate minimum number of accepted independent cases required for a one-sided exact 95% upper bound below:

- 3% is 99;
- 2% is 149;
- 1% is 299;
- 0.5% is 598.

This is why a result such as `0 / 67` cannot substantiate a sub-1% risk claim.

For the planned paper, an operating point around 1% false-link risk therefore needs roughly 300 accepted negative/at-risk confirm cases with no failures, or more cases if failures occur.

## Threshold-selection warning

The current consistency thresholds have been inspected on scans 921 and 953. The observation that every false structural decision in this V2 run has consistency below 0.60 is scientifically interesting but **cannot be used to declare 0.60 the final threshold**.

Publication protocol:

1. choose/calibrate the consistency policy using calibration patients only;
2. freeze it;
3. open independent confirm patients once;
4. report both observed false-link rate and uncertainty interval;
5. keep REVIEW/abstention as a real outcome rather than retuning the threshold to make false links disappear.

## Machine artifact

`results/ccta_graph_lira_safe_repair/2026-09-20/selective_risk_exact_ci_v2.csv`
