# ANZA-S Phase A2 causal stop

Phase A2 was a zero-training causal audit. A1, A2, and A3 used the exact same
tangent centerline. A2 reset its anisotropic ellipse at every step; A3 alone
recursively transported covariance as `Sigma[k+1] = J[k] Sigma[k] J[k]^T`.

Frozen result:

- A1 isotropic macro TPR: 0.908333;
- A2 reset-local anisotropy macro TPR/ranking: 1.0 / 1.0;
- A3 composed Cauchy--Green macro TPR/ranking: 1.0 / 1.0;
- A3 minus A2 macro TPR: 0.0, paired bootstrap 95% CI [0.0, 0.0];
- A2 and A3 parallel false-positive rate: 0.0;
- lambda 0.35 is not inert: it changes scores and improves X TPR by 0.275
  relative to the lambda-zero composed null, but not relative to A2;
- matched StraightGap/NegativeGap geometry-only AUROC: 0.5, as expected.

Formal status: `ANOSOV_COCYCLE_REDUNDANT_AT_ORACLE`. This is not an unsafe or
mathematically inert operator; it is a failure to establish incremental causal
value over the required local-anisotropy control. Do not open Phase B, synthetic
confirm/test, CRACKS, or expert evaluation for this construction.

Evidence is under `results/anza_s/a2/`; validate with
`python scripts/validate_anza_s_a2.py`.
