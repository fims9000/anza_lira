# ANZA-KS K1.5/K2 frozen transfer STOP

## Frozen results

- K1.5 added only the missing Shear+KS control. CatKS and ShearKS both reached
  macro TPR@FPR<=0.05 of 1.0; Cat-minus-shear delta was 0.0 with CI [0, 0].
  Symbolic information remains useful at patch level, but Anosov specificity is
  not established.
- Dense NumPy/PyTorch Static, CatRaw, CatKS, and ShearKS features matched within
  1e-6 before training. The 4096/1024/1024/2048 benchmark streams were hashed
  before model outputs; confirm stayed hash-only.
- Seed-41 M0--M4 trained for exactly 15 epochs. M4 improved natural synthetic
  Dice (0.85770 vs M1 0.84880) and clDice (0.88400 vs 0.87390), and passed the
  natural topology and pixel-safety gates.
- The primary mechanism gate failed: M4 accepted 4/1024 distractors versus
  1/1024 for Static and ShearKS and 2/1024 for CatRaw. Paired improvement CIs
  were not positive.

## Boundary

Research status is `STOP_ANZA_KS_FEATURE_NOT_TRANSFERRED`. Do not run seeds
42/43, K2 confirm, CRACKS, or expert, and do not retune the K2 threshold,
features, benchmark, or loss on development. The natural synthetic gains are
supportive diagnostics only and do not override the frozen mechanism failure.
