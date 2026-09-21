# Expanded 28-case geometry baselines before CT extraction

Date: 2026-09-21

This result uses the already-frozen 1,360-row geometry-matched relation plan. It therefore does not require raw CCTA and can be frozen before the image experiment.

## Validation discipline

For every score, the decision threshold is selected on the five validation patients only, maximizing recall subject to FPR <= 5%.

The OGMC-style score uses:

`exp(-distance / lambda) * anchor_alignment * candidate_alignment * tangent_consistency`

with lambda selected from a predeclared validation grid `[2,3,4,5,6,8,10] mm`. Validation selected `lambda=5 mm`.

This is an **OGMC-style** baseline, not a reproduction of Zhu et al.

## Held-out six-patient test

| model | AUROC | recall | FPR | precision | pair top-1 |
|---|---:|---:|---:|---:|---:|
| nearest endpoint | 0.9189 | 16.17% | 4.19% | 79.41% | 95.21% |
| tangent mean | 0.6476 | 14.37% | 8.38% | 63.16% | 77.84% |
| OGMC-style touch | 0.8148 | 39.52% | 3.59% | 91.67% | 92.22% |
| geometry logistic regression | 0.9366 | 8.38% | 1.80% | 82.35% | 95.81% |
| geometry HGB | **0.9685** | 37.72% | **1.20%** | **96.92%** | 95.21% |

The result exposes the same important distinction seen earlier in Graph-LIRA:

- candidate identity ranking is relatively easy: pair top-1 is around 92–96% for several geometry models;
- deciding **when it is safe to accept a relation** is much harder;
- calibration / selective acceptance is therefore a separate scientific problem from candidate ranking.

The OGMC-style score has lower overall AUROC than nearest endpoint but much higher recall at the validation-constrained low-FPR operating point. This is another reason not to evaluate reconnection systems by one global ranking metric alone.

## Patient-cluster uncertainty

Bootstrap unit = patient, 2,000 resamples across the six held-out patients.

Geometry HGB:
- recall median 37.72%, 95% cluster-bootstrap interval 24.82–47.80%;
- FPR median 1.19%, interval 0–2.74%;
- AUROC median 0.9683, interval 0.9460–0.9868.

OGMC-style:
- recall median 39.22%, interval 30.77–48.69%;
- FPR median 3.55%, interval 1.38–5.58%;
- AUROC median 0.8155, interval 0.7727–0.8453.

Paired against nearest endpoint:
- geometry HGB recall delta median +21.56 percentage points, 95% interval +8.46 to +30.38;
- geometry HGB FPR delta median -3.00 points, interval -4.44 to -1.29;
- OGMC-style recall delta median +23.24 points, interval +13.42 to +31.74.

These intervals describe the six held-out patients in this controlled synthetic-gap relation task; they are not population-level clinical confidence intervals.

## Consequence for the CT experiment

The image-conditioned model must beat a stronger geometry operating-point baseline than simple distance.

Primary local comparison after raw CT is available should therefore include:

1. nearest endpoint;
2. OGMC-style touching score;
3. geometry HGB;
4. radial CT;
5. geometry + radial CT.

The target is not merely higher AUROC. The desired result is more true relation acceptance at equal or lower false relation risk under the same validation-only threshold rule.
