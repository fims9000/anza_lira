# ANZA-S zero-training oracle Gate A

Date: 2026-08-18.

This is a new hypothesis and does not rewrite the frozen negative ANZA-2
history. Protocol `ANZA_S_ANOSOV_COCYCLE_SHADOWING_ORACLE_V1` compared O0
scalar ANZA, O1 mode-state, O2 tangent streamline, O3 hyperbolic-cocycle
rollout, and O4 cocycle plus two-sided shadowing. Train selected finite
operating points under FPR <= 0.05 for each negative stratum; validation was
used once for Gate A. Confirm, test, CRACKS, and expert remained closed.

Formal Gate A passed against both specified O0 and O2 baselines. O4 validation
macro positive recall was 0.50, X wrong-turn FPR 0.0, parallel false bridge
0.0, negative-gap false bridge 0.0, and derived curved-gap recall 1.0.

The claim boundary is essential:

- O2 and O3 scores are exactly identical (maximum absolute difference 0.0),
  so an incremental cocycle rollout effect is not established;
- O4 X-correct recall and StraightGap recall are both 0.0 at the frozen safe
  operating point;
- matched positive/negative straight-gap geometry is identical to
  trajectory-only O2/O3/O4, so both receive identical scores;
- no generic tangent-plus-shadowing causal control was in the frozen packet.

Therefore this is only a formal combined-shadowing oracle screen. It is not
evidence that the hyperbolic cocycle is useful, and it supports no learned,
segmentation, CRACKS, expert, or novelty claim. Phase B may occur only after a
separate protocol is frozen; it was not run here.

Canonical evidence: `results/anza_s/oracle/` and
`scripts/validate_anza_s_oracle.py`.
