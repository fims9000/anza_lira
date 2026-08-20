# ANZA-LIRA LEADS RC1 operating-point STOP

RC1 preserved the exact seed-41 L0/L2/L3 architectures, loss, 10% label
regime, and 20-epoch budget. It rotated optimization/calibration/development
within the old label-audited training pool and excluded all parent A1 active,
calibration, and development sections from RC1 evaluation. Expert data stayed
locked.

Unlike parent A1, calibration used score quantiles plus explicit thresholds
through 0.9999. Precision >=0.90 with nonzero recall was feasible, so thresholds
were frozen before development: L0=0.9884244204, L2=0.9674798250, L3=0.9700000000.

Fresh development results:

- L2: precision 0.933832, recall 0.278637, Dice 0.396562, clDice 0.419764,
  AUPRC 0.785844, unsupported-white foreground 0.009075.
- L3: precision 0.935199, recall 0.266801, Dice 0.383514, clDice 0.406293,
  AUPRC 0.784166, unsupported-white foreground 0.008851.
- L3-L2 Dice = -0.013049, 95% section-bootstrap CI
  [-0.014564, -0.011565].
- L3-L2 clDice = -0.013471, 95% section-bootstrap CI
  [-0.015515, -0.011508].
- L3-L2 AUPRC = -0.001677.
- Unsupported-white ratio L3/L2 = 0.975414 (safe relative to L2), while the
  ratio to the almost-empty high-threshold L0 prediction is 4.538626.

The frozen research status is
`STOP_ANZA_LOW_LABEL_GAIN_WAS_OPERATING_POINT_SPECIFIC`. The parent A1
`+0.0366` clDice diagnostic did not reproduce after correct calibration and a
fresh cross-fit evaluation. Do not open seeds 42/43, ANZA-MS, SSL, domain shift,
LIRA, OOF, or expert from this branch.

Artifacts are under `results/anza_leads/rc1/`; `validator.json` is PASS and
recomputes the decision from per-section rows.
