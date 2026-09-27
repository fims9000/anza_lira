# Active research branch — start here

Canonical active research branch:

`research/coronary-connectivity-repair`

Use:
- `main` — stable/default repository branch;
- `research/coronary-connectivity-repair` — all active and future research work for this line.

Historical checkpoints:
- `research/ccta-graph-lira-safe-repair`;
- `research/varvara-ccta-graph-lira`.

Do not continue new work on those historical branches.

## Current verified result

Matched ImageCAS / ImageCAS-X CT28 PAIR experiment:

- 28 patients;
- 17 train / 5 validation / 6 held-out test;
- 1,360 controlled pair relations;
- 28/28 CT geometry checks passed.

Held-out operating point selected from validation:

| method | Recall | FPR | Precision | AUROC |
|---|---:|---:|---:|---:|
| geometry HGB | 37.72% | 1.20% | 96.92% | 0.9685 |
| radial 2.5-D CCTA | 68.86% | 4.19% | 94.26% | 0.9440 |
| **geometry + radial CCTA** | **82.63%** | **1.80%** | **97.87%** | **0.9847** |

This is a local binary PAIR-relation result, not an end-to-end Graph-LIRA result.

## CT28 PAIR reproducibility status

The previous collaborator gap is closed.

Committed under `artifacts/varvara/ct28_pair/`:

- exact pair plan;
- full row-level predictions;
- summary/protocol;
- alignment/provenance;
- lossless compressed full feature table;
- optional regenerated model snapshots.

Restore/retrain instructions:
`docs/varvara/REPRODUCE_CT28_PAIR_BASELINE.md`.

## Exact next scientific task

The next missing block is **28-patient JUNCTION+CT evidence** on the same frozen cohort/split.

Required comparison:

1. JUNCTION geometry;
2. JUNCTION CT;
3. JUNCTION geometry + CT.

Only after patient-general JUNCTION_CT exists:

```text
PAIR geometry + PAIR CT
+
JUNCTION geometry + JUNCTION CT
        ↓
CT-conditioned scene-level relation head
        ↓
NONE / PAIR / JUNCTION / BOTH
        ↓
frozen Graph-LIRA
        ↓
tau = 0.85
consistency = 0.60
        ↓
repair / abstain
```

No held-out test retuning.

## Collaborator entry point

For the current execution state read:

`docs/varvara/FINAL_HANDOFF_2026-09-27.md`

Then:
- `docs/varvara/CURRENT_TASK.md`;
- `docs/varvara/NEGATIVE_RESULTS_THAT_MATTER.md`;
- `docs/varvara/ANSWER_TO_QUESTIONS_2026-09-26.md`.

## Scientific boundaries

Do not:
- retune on held-out test patients;
- replace the official patient split;
- use anatomical branch labels as inference features;
- copy the old six-patient CT add-only/hysteresis integration as the final method;
- claim full CT-conditioned Graph-LIRA improvement before that experiment exists;
- claim ANZA superiority before a clean encoder ablation.

Raw medical CT is intentionally not stored in Git.
