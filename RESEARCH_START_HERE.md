# Active research branch — start here

Canonical active research branch:

`research/coronary-connectivity-repair`

This is the single current source of truth for the coronary
connectivity-repair / CCTA Graph-LIRA line.

## Current scientific milestone

The local evidence stage is now sufficiently complete for both relation
families:

- PAIR geometry + CCTA;
- JUNCTION geometry + CCTA on train/validation.

The next experiment is **CT-conditioned Graph-LIRA**, not another local
candidate model and not CNN/ANZA yet.

```text
PAIR local evidence
+
JUNCTION local evidence
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

Detailed current execution task:

`docs/varvara/TASK_CT28_CT_CONDITIONED_GRAPH_LIRA_2026-10-06.md`

## Branch rule

Use:

- `main` — stable/default repository state;
- `research/coronary-connectivity-repair` — active coronary/CCTA research.

Do not continue active work on the old checkpoint/experiment branches.

## Current evidence

### PAIR

Frozen CT28 held-out local result for geometry + radial CCTA:

- AUROC 0.9847;
- recall 82.63%;
- FPR 1.80%;
- precision 97.87%.

### JUNCTION train/validation

- 143 positives + 1,666 negatives;
- 17 train / 5 validation patients;
- 9 geometry + 152 CT features;
- geometry top-1 validation 35/36;
- raw geometry+CT top-1 validation 36/36;
- held-out JUNCTION test remains closed.

Current JUNCTION artifacts:

`artifacts/varvara/ct28_junction/`

## Research order

1. freeze/reproduce current local artifacts;
2. build explicit CT28 scene/relation layer;
3. compare geometry-only vs CT-conditioned Graph-LIRA;
4. run failure analysis;
5. only then radial vs compact CNN vs compact ANZA;
6. later expand matched CCTA cohort / natural segmentation failures.

Raw medical CT is intentionally not stored in Git.
