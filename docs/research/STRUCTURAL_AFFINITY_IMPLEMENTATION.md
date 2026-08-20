# StructuralAffinityANZA implementation contract

This document records the frozen C0--C3 implementation. It does not claim a
positive result.

- `models/azconv.py` remains the unchanged current/published C0 baseline.
- C1 replaces the categorical rule softmax with independent sigmoid fuzzy
  degrees. The raw pair weight remains `mu_r(p) mu_r(q) G_r(p,q)` and global
  normalization remains over rules and valid neighbors.
- C2 predicts a symmetric, mode-conditioned local edge score and applies
  `w = w0 exp(beta s)` before the same global normalization. `beta` is bounded
  by a nonnegative centered-softplus parameter, initialized to exactly zero,
  and projected after each optimizer step. A negative beta is forbidden because
  it would invert the supervised meaning of a high affinity score.
- Pair geometry is axial. All angular pair features use doubled angles, so
  `theta` and `theta + pi` are equivalent.
- C3 additionally predicts eight sparse, collinear radius-2 edge scores. Each
  radius-2 score is averaged with its local directional score before direct
  modulation of the corresponding normalized local ANZA interaction. It is
  also supervised as an explicit edge. C3 does not introduce a dense 5x5
  convolution or a second segmentation path.
- At `beta=0`, both C2 and C3 are bit-exact clean C1 for the same state. The
  product form `w0 * exp(beta*s)` is algebraically the specified log-space
  update and preserves first-step gradient flow to `beta`.
- S1 freezes the entire segmentation path and trains only the context/affinity
  parameters on balanced lineage edges. S2 uses a ten-times smaller learning
  rate for the base than for the affinity head. No pixel gap BCE is used.

The v4 test split, CRACKS data, and expert masks remain inaccessible until the
frozen validation and independent confirmation gates authorize them.
