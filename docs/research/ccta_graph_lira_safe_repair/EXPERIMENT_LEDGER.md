# Experiment ledger — CCTA Graph-LIRA safe repair

## A. Historical context

The older ANZA-LIRA controlled continuation line is background evidence only. It separated candidate generation, contextual pair identity, and max-min path construction. It is not external natural-gap validation.

## B. Preserved exploratory artifacts from the previous session

### B1. Perturbation-consistency uncertainty

Artifact:
`results/ccta_graph_lira_safe_repair/2026-09-20/perturbation_risk_coverage_30deg_strong.csv`

Scene count in the per-scene file: 446.

Baseline setting represented in the file:
- base exact scene rate about 0.9036;
- base false-scene rate about 0.0426.

Important operating points:

| gate | threshold | accepted | coverage | false among accepted | exact among accepted |
|---|---:|---:|---:|---:|---:|
| stability | 0.90 | 250 | 0.5605 | 0.0040 | 0.9840 |
| baseline agreement | 0.90 | 249 | 0.5583 | 0 | 0.9880 |
| combined consistency | 0.90 | 249 | 0.5583 | 0 | 0.9880 |
| combined consistency | 0.80 | 325 | 0.7287 | 0.00308 | 0.9723 |

Interpretation: perturbation agreement is promising for selective repair, but all operating points remain exploratory until the generator script is reconstructed and re-run.

### B2. Compact sequence-context pilot

Artifact:
`results/ccta_graph_lira_safe_repair/2026-09-20/sequence_summary.csv`

| model | mean AUROC | median AUROC | mean FPR |
|---|---:|---:|---:|
| sequence CNN tokens | 0.9274 | 0.9377 | 0.2446 |
| sequence Transformer tokens | 0.9294 | 0.9472 | 0.2434 |

Interpretation: simply replacing the CNN by a Transformer over heavily compressed cross-section statistics did not solve the task. The next sequence experiment must preserve local image structure before sequence aggregation.

## C. Exploratory conclusions to re-run canonically

The previous session additionally indicated:

- geometry remains a strong baseline;
- radial 2.5-D / candidate-aligned 3-D tube representations were stronger than the compressed token sequence pilot;
- independent pair decisions can create incompatible repairs;
- joint graph consistency reduces those incompatible false links;
- fixed degree-3 junction assumptions fail on some real branching configurations;
- variable-degree junction reasoning is preferable;
- max-min path construction should remain downstream of structural acceptance.

These are direction-setting observations, not yet final paper numbers on this branch.

## D. New canonical experiments on this branch

### D1. Scan-953 alignment audit

Status: running / to be frozen.

Purpose: establish whether the uploaded ImageCAS-X anatomical labels for scan 953 can be safely mapped to the candidate original ImageCAS CT / binary mask before any CT+branch experiment is trained.

Pass criterion: a frozen coordinate transform with strong mask overlap and centerline coverage, or a documented blocker proving that the candidate volume is not the matched source representation.
