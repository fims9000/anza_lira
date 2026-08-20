# ANZA-KS K0/K1 frozen causal feature PASS

Date: 2026-08-19.

K0 validated the exact N=17 finite Cat permutation, fixed four-cell partition,
symbolic words through length four, normalized image density, finite block and
conditional entropy, and predictive information. These are finite-partition
features, not image KS entropy.

`DYNAMICS_MATCHED_V1` was frozen before symbolic scoring. Each of five tasks
used 2048 train, 1024 development, and 2048 hash-only confirm pairs. Static
development AUROC was 0.5 on every task; maximum positive/negative static
signature delta was 3.10e-14. Confirm hash is
`65d7f388350c5f47712db52149bfdddbaf3db13907adb0236b34ddbf0b29936e`
and confirm was not evaluated.

The first evaluator applied the mechanism gate to ranking only and produced
`STOP_KOLMOGOROV_FEATURES_REDUNDANT`. This was an implementation defect because
the frozen master protocol explicitly permits TPR@FPR0.05 OR matched ranking.
`ANZA_KS_K1_GATE_AUDIT_R1` used the immutable `per_pair.csv` scores, retrained
nothing, and applied 10,000 paired bootstraps to the missing TPR branch.

Corrected frozen evidence:

- full ANZA-KS vs static passed 5/5 task gates;
- ANZA-KS vs CatRaw macro TPR delta +0.108398, 95% CI
  [0.099805, 0.120703];
- CatRaw vs nonhyperbolic shear macro TPR delta +0.015820, 95% CI
  [0.001172, 0.026758], with positive task deltas on 2/5 tasks;
- corrected status `ANZA_KS_CAUSAL_FEATURE_PASS`;
- no feature/readout was recomputed, confirm remained closed, and K2 was not
  opened.

K2 is authorized only under a new frozen protocol. Preserve the K0/K1
partition, Cat/shear maps, K, matcher, readout policy, score rows, confirm hash,
and gate. The current result is a controlled feature-level causal PASS, not yet
a segmentation, practical architecture, CRACKS, or expert result.
