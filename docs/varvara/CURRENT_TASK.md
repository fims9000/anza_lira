# CURRENT TASK — CT-conditioned Graph-LIRA

Date: 2026-10-06

Canonical branch:

`research/coronary-connectivity-repair`

## One current milestone

The current milestone is:

```text
PAIR local geometry + CCTA evidence
+
JUNCTION local geometry + CCTA evidence
        ->
NONE / PAIR / JUNCTION / BOTH
        ->
frozen Graph-LIRA
        ->
tau = 0.85
consistency = 0.60
        ->
repair / abstain
```

The detailed execution task is:

`docs/varvara/TASK_CT28_CT_CONDITIONED_GRAPH_LIRA_2026-10-06.md`

Read that document before changing code.

## Current state

PAIR local CT evidence is frozen and reproducible.

JUNCTION train/validation local evidence is also ready enough to move upward:

- generator frozen for development;
- 143 degree-3 positives + 1,666 negatives;
- candidate recall 100%;
- 17 train / 5 validation patients;
- held-out JUNCTION test remains closed;
- CT representation `3 x 17 x 25`;
- 9 geometry + 152 CT features;
- validation geometry top-1 35/36;
- validation raw geometry+CT top-1 36/36;
- raw concat improves ranking but worsens the validation safe operating point.

Therefore generator tuning and another local concat model are not the main task.

## Important correction to the previous task note

Do **not** start CNN vs ANZA now.

The project roadmap requires:

1. CT-conditioned relation layer;
2. end-to-end Graph-LIRA comparison;
3. failure analysis;
4. only then radial vs compact CNN vs compact ANZA.

The existing `train_junction_lira_fusion.py` remains a development diagnostic/reference, not a separate research milestone.

## Test firewall

Do not open JUNCTION held-out patients until the scene/relation/graph protocol is frozen on train/validation.

Held-out patients:

`954, 958, 972, 973, 980, 984`

Do not retune `tau=0.85` or consistency `0.60` on test.
