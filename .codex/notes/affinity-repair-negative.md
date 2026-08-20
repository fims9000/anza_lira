# Structural-affinity repair closeout (2026-08-18)

The frozen C0--C3 v4 development cycle ended as
`AFFINITY_REPAIR_NEGATIVE_WITH_ROOT_CAUSE`. Do not add C4, open v4 test, run
confirmation, CRACKS, or expert evaluation under this protocol.

Durable implementation lessons:

- affinity pair geometry must average doubled angles; ordinary vector
  averaging violates `theta == theta + pi`;
- beta must be nonnegative because the supervised score assigns higher values
  to true affinities; a signed beta can invert the learned relation;
- sparse radius-2 evidence must causally affect the local ANZA weight, not only
  receive an auxiliary loss;
- the frozen `near_parallel/close` stratum has no negative edges at local/radius
  2 support, so hard macro AP is `NA`, not an average over only covered strata.

The head learned general edge separation, but matched-negative AUROC remained
near chance, beta on/off topology CIs crossed zero, and gap gates failed. The
authoritative artifacts are under `results/affinity_repair/final/` and the ZIP
SHA256 is `9286bd9663dbd632f9ee6461f622b61fbdee33004996b1ad9bbae7d1c233d2ad`.
