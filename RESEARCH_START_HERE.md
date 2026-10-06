# Active research branch — start here

Canonical active research branch:

`research/coronary-connectivity-repair`

This is the **single current source of truth** for the coronary connectivity-repair / CCTA Graph-LIRA line.

## Branch rule

Use:

- `main` — stable/default ANZA-LIRA repository state;
- `research/coronary-connectivity-repair` — active coronary/CCTA research.

Do **not** continue active work on:

- `research/ccta-graph-lira-safe-repair`;
- `research/varvara-ccta-graph-lira`;
- `research/anza-lira-q1-journal`;
- `experiment/junction-ct28`;
- `experiment/ct28-junction`.

Those names are retained only as historical/checkpoint pointers for now.

The CT28 JUNCTION experiment formerly developed on
`experiment/ct28-junction` was promoted back into this canonical branch on
2026-10-06. New work should continue here unless a deliberately scoped,
short-lived experiment branch is created.

Do not merge the active coronary research line into `main` while the
end-to-end protocol is still under development.

## Current verified research state

### CT28 PAIR

Matched ImageCAS / ImageCAS-X experiment:

- 28 patients;
- 17 train / 5 validation / 6 held-out test;
- 1,360 controlled pair relations.

Held-out PAIR baseline:

| method | Recall | FPR | Precision | AUROC |
|---|---:|---:|---:|---:|
| geometry HGB | 37.72% | 1.20% | 96.92% | 0.9685 |
| radial 2.5-D CCTA | 68.86% | 4.19% | 94.26% | 0.9440 |
| **geometry + radial CCTA** | **82.63%** | **1.80%** | **97.87%** | **0.9847** |

This remains a local binary PAIR result, not an end-to-end Graph-LIRA result.

### CT28 JUNCTION train/validation

The repository now also contains the controlled degree-3 JUNCTION train/val
pipeline:

- frozen controlled generator;
- candidate-aligned CT extraction;
- geometry / CT / raw geometry+CT baselines;
- current baseline metrics;
- development-only patient-OOF score-fusion prototype.

Held-out JUNCTION test remains closed while model/protocol choices are still
being selected.

## Authoritative current documents

Read in this order:

1. `docs/varvara/CURRENT_TASK.md`
2. `RESEARCH_START_HERE.md`
3. `docs/research/BRANCH_STRATEGY.md`

Supporting technical references:

- `docs/varvara/REPRODUCE_CT28_PAIR_BASELINE.md`
- `docs/varvara/CT28_DATA_ACCESS.md`
- `docs/varvara/NEGATIVE_RESULTS_THAT_MATTER.md`
- `docs/varvara/ANSWER_TO_QUESTIONS_2026-09-26.md`

Older handoff/review documents are preserved for provenance but are not
current instructions.

Raw medical CT is intentionally not stored in Git.
