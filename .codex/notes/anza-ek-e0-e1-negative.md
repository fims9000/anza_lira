# ANZA-EK E0/E1 frozen negative result

Date: 2026-08-19.

The exact Cat-map / Koopman implementation passed E0. The mathematical toral
automorphism has determinant one, exact inverse, reciprocal hyperbolic
eigenvalues, correct stable/unstable finite-time growth, invariant constants,
and exact finite-grid permutation diagnostics. Bilinear readout is explicitly
treated as a numerical approximation, not as an exact finite ergodic map.

E1 was frozen before scores were computed: six identifiable tasks, 256 paired
examples per task, four fixed controls, one deterministic unlearned score, and
no classifier or training. The strongest control was static anisotropy. Cat
Koopman passed 0/6 task gates: five tasks tied the saturated control, while the
largest gain on oriented clutter was only +0.00390625 paired ranking and
+0.01171875 TPR@FPR0.05, below the frozen +0.08 requirement.

Research status:

`STOP_ERGODIC_ANOSOV_LOCAL_FEATURE_NO_MECHANISM`

Do not tune the score, replace tasks, open E2, add conjugacy, train a network,
or access confirm, CRACKS, or expert data for this branch. The valid conclusion
is narrow: the Cat-map implementation is correct, but this frozen finite-time
Koopman feature construction did not show incremental causal value beyond a
strong static anisotropic control.
