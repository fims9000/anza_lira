# ANZA-LIRA H1 final correctness hotfix and terminal STOP

H1 did not revise the immutable natural-gap, intervention V1, or Graph-Cut V2
STOPs. It replaced only the invalid round-cap capsule with an exact flat-cap
ribbon defined by nearest projection onto the complete ordered trace and a
frozen arclength interval. Targeted synthetic tests reproduced V2 anchor
destruction and verified ribbon disconnection, anchor preservation, no
longitudinal spillover, curved-trace behavior, reversal invariance, and
collateral rejection.

The old sections 263--344 were used only as a mechanical bug audit. Of 1,753
pre-treatment eligible traces, only 34 had a valid local separation using the
frozen radii 3/5/7/9/11/13/15. Retention was 0.0193953 versus the frozen 0.50
gate; 1,719 cases remained connected at every permitted radius. Accepted-case
TreatmentValidity was 1.0, but that does not override the retention failure.
Final status is `STOP_H1_RIBBON_BENCHMARK_FAIL`.

Sections 347--400 remained unopened by H1. Frozen SBPP, P0, path, expert, and
all new architectures remained locked. There is no H2/V3 or other rescue
branch. The terminal manuscripts preserve the controlled measured result
(pair AUROC 0.9923; learned recovery 0.6719; false bridge 0.0078; oracle path
1.0/0.0) together with its failed 0.70 learned-recovery gate and its synthetic
scope.

