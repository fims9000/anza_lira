# CT28 JUNCTION work branch

Branch:

`experiment/ct28-junction`

Canonical parent research branch:

`research/coronary-connectivity-repair`

## Status — 2026-10-06

The generator review from 2026-10-03 is historical development context. The current generator has since been simplified/frozen for the controlled benchmark:

- `segment_label` is audit-only;
- negative selection is geometry-distance based;
- `competitive / isolated` is explicit;
- degree-3 train/val candidate recall is 100%;
- held-out test remains closed.

Current code in this branch:

- `build_junction_plan.py` — frozen train/val controlled generator;
- `extract_junction_ct_features.py` — candidate-aligned CT extraction and alignment checks;
- `train_junction_baselines.py` — geometry / CT / raw geometry+CT baselines;
- `train_junction_lira_fusion.py` — patient-OOF score-level JUNCTION-LIRA v0.

Current baseline metrics are under:

`artifacts/varvara/ct28_junction/`

Read first:

`docs/varvara/CURRENT_TASK.md`

## Current scientific direction

Do not return to generator tuning unless a concrete candidate-recall failure appears.

The active line is now:

```text
JUNCTION geometry + CT evidence
        ->
JUNCTION-LIRA fusion
        ->
radial vs compact CNN vs compact ANZA
        ->
PAIR + JUNCTION relation type
        ->
Graph-LIRA
        ->
repair / abstain
```

Do not use held-out test to choose architecture or thresholds.
