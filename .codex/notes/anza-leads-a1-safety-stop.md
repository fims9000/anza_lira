# ANZA-LIRA LEADS V1 A0/A1 safety stop

The new proof line used the exact frozen ANZA-HS operators, a contiguous
section-disjoint CRACKS split (274 train-pool, 32 calibration, 78 development,
and two four-section buffers), held-out nonexpert annotators for evaluation,
and no expert access. Seed 41 used 27 nested 10-percent optimization sections,
20 epochs, and the same partial-label plus orientation-auxiliary objective for
L0--L3.

L3 ANZA-HS produced a large favorable development point estimate relative to
initialized-equivalent L2 GenericAniso: Dice +0.036431 and clDice +0.036621;
paired section-bootstrap intervals were fully positive. Precision and recall
also improved. This is not a formal PASS because unknown-white foreground rose
from 0.084226 to 0.097661, a 1.159515 ratio versus the frozen maximum 1.10.
Fragmentation was essentially unchanged (ratio 0.999822). L3 also remained
below the plain L0 backbone by 0.077585 Dice and 0.079734 clDice.

All models failed the calibration precision >=0.90 constraint on the frozen
0.05--0.95 grid, so the predeclared infeasibility rule selected threshold 0.95.
Formal status is `STOP_ANZA_LABEL_EFFICIENCY_NO_SIGNAL`. Do not open seeds
42/43, ANZA-MS, SSL, domain shift, OOF, expert, or LIRA under this protocol.
The positive L3-vs-L2 deltas may be described only as seed-41 CRACKS development
diagnostics that failed the unknown-region safety gate.
