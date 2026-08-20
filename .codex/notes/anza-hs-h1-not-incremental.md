# ANZA-HS H0/H1 practical stop

The composed-cocycle A2 result remains frozen. ANZA-HS tested only two retained
mechanisms: a fixed M=8 axial evidence bank and reciprocal local scales
`sigma_u=ell exp(lambda)`, `sigma_s=ell exp(-lambda)` with frozen ell=1.5 and
lambda=0.35.

StressBench V5, train/dev/confirm streams, 20-epoch budget, seed 41, threshold
rules, and gates were frozen before training. Confirm remained locked. B0, B1,
B2, and B3 used the same residual-attention backbone, data order, optimizer,
segmentation loss, and epoch budget. B2 was initialized to exactly reproduce
the B3 kernel but could learn unconstrained scales.

Frozen dev-gate result, 220 independent samples:

- B2 Dice/clDice/fragmentation: 0.923381 / 0.982815 / 0.043182;
- B3 Dice/clDice/fragmentation: 0.929174 / 0.984387 / 0.040909;
- B3-B2 Dice: +0.005792;
- B3-B2 clDice: +0.001572;
- B3/B2 fragmentation ratio: 0.947368;
- B2/B3 parallel false connection: 0.036364 / 0.027273.

Dice non-inferiority passed, but neither predeclared practical structural gate
passed: clDice gain was below +0.015 and fragmentation reduction was below 10
percent. Formal status: `HYPERBOLIC_CONSTRAINT_NOT_INCREMENTAL`.

Do not run H2 shadowing stability, seeds 42/43, StressBench confirm, CRACKS,
continuation, or expert under this protocol. The favorable point estimates may
be reported only as seed-41 synthetic development diagnostics, not as a positive
architectural claim.
