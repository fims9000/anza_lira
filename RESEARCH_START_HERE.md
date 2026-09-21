# Active research branch — start here

Canonical active research branch:

`research/coronary-connectivity-repair`

This branch is the continuation point for the coronary CCTA connectivity-repair work. It contains the full technical history inherited from the previous research branches, the 28-patient matched-CCTA experiment, all compact reproducible results, the current article direction, and the handoff/roadmap material prepared for collaborators.

## Branch policy

Use:

- `main` — stable/default repository branch;
- `research/coronary-connectivity-repair` — **all active and future research work for this line**.

Do not continue new research work on:

- `research/ccta-graph-lira-safe-repair` — historical predecessor/checkpoint;
- `research/varvara-ccta-graph-lira` — temporary handoff snapshot created before the canonical branch was renamed neutrally.

Those branches are intentionally kept for safety/history so nothing is lost.

## Current research state

The strongest verified matched-CCTA result is the 28-patient relation experiment:

- 17 train patients;
- 5 validation patients;
- 6 held-out test patients;
- 1,360 controlled relation examples;
- 28/28 CT-to-ImageCAS-X geometry checks passed.

At the validation-selected FPR <= 5% operating point on held-out test:

| method | Recall | FPR | Precision | AUROC |
|---|---:|---:|---:|---:|
| geometry HGB | 37.72% | 1.20% | 96.92% | 0.9685 |
| radial 2.5-D CCTA | 68.86% | 4.19% | 94.26% | 0.9440 |
| **geometry + radial CCTA** | **82.63%** | **1.80%** | **97.87%** | **0.9847** |

The next core experiment is CT-conditioned Graph-LIRA under the already frozen selective policy:

- relation confidence `tau=0.85`;
- perturbation consistency `0.60`;
- no test retuning.

## What to read

For the concise scientific direction:

1. `docs/varvara/ARTICLE_DIRECTION.md`
2. `docs/varvara/RESULTS_TO_USE.md`
3. `docs/varvara/ROADMAP.md`

Those files were written as a collaborator-friendly handoff, but they are also the cleanest compact description of the current direction.

For technical provenance and reproducibility:

- `docs/research/ccta_graph_lira_safe_repair/REAL_CT28_RESULTS_AND_PROMOTION.md`
- `docs/research/ccta_graph_lira_safe_repair/CHECKPOINT.md`
- `results/ccta_graph_lira_safe_repair/2026-09-21/`
- `scripts/research/ccta_graph_lira_safe_repair/`

## What not to do

Do not:

- retune thresholds on held-out test patients;
- replace the official patient split;
- use anatomical labels as model inference features;
- claim end-to-end CT-conditioned Graph-LIRA improvement before that experiment is run;
- claim ANZA is superior before a clean architecture ablation.

Raw medical CT is intentionally not stored in Git.
