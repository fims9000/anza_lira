# ANZA-2 Phase-2 controlled selectivity

Phase 2A is immutable negative evidence. Its predeclared path endpoint saturated
for both ANZA-2 and LegacyANZA, so it remains
`STOP_ANZA2_GEOMETRY_NOT_STRUCTURALLY_SELECTIVE`; do not rewrite or tune it.
The frozen Phase-2A metrics SHA256 is
`04b35a97c830b682f682084498673daf280e1c81dad407e850be199e8e15e383`.

Phase 2B used an independent replacement seed stream, the already frozen
thresholds, and a predeclared branch-recall endpoint. It passed with ANZA-2
branch recall `1.00` versus LegacyANZA `0.90` (paired delta `+0.10`, 95% CI
`[0.10, 0.10]`), including X-crossing recall `1.00` versus `0.75`. Path TPR was
`1.00` versus `0.9921875`; false-bridge FPR was `0` for both. Protocol SHA256:
`5b1789554722f91a28e32590f897d7c9f6c2642f5a83994c86ea43c352d4cd64`.

This is controlled oracle-field evidence only. It does not show that ANZA-2 can
learn its field from images and does not support a CRACKS or expert-data claim.
Phase 3 may run only as a bounded, pre-frozen learned-synthetic test. Keep CRACKS
and expert data locked until that gate passes.
