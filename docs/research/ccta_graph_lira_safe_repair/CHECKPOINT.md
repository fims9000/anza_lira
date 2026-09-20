# Research checkpoint — 2026-09-20

Branch: `research/ccta-graph-lira-safe-repair`

## Frozen direction

We are no longer searching architectures blindly. The active hypothesis is:

> false-link risk can be reduced by combining local pair/junction evidence with global graph consistency and selective abstention; image context should only be added where it reduces confident wrong-branch decisions beyond geometry alone.

## What is already learned

1. Geometry is a strong baseline and must stay in every comparison.
2. Independent local repair decisions create mutually inconsistent false bridges.
3. Joint Graph-LIRA / structural optimization is a meaningful component, not just post-processing.
4. Lost bifurcations require variable-degree junction modeling; fixed pair matching is structurally insufficient.
5. Simple top-1/top-2 margin is a weak abstention score.
6. Perturbation consistency is a useful uncertainty signal.
7. A Transformer over heavily compressed cross-section statistics does not outperform strong radial 2.5-D / 3-D representations; preserving local spatial image content is the next sequence-model requirement.
8. Max-min remains downstream: it constructs a path only after structural identity is accepted.

## Exact artifact-backed numbers currently preserved

See:

- `results/ccta_graph_lira_safe_repair/2026-09-20/perturbation_risk_coverage_30deg_strong.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/perturbation_scene_uncertainty_30deg_strong.csv.gz.b64`
- `results/ccta_graph_lira_safe_repair/2026-09-20/sequence_summary.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/sequence_cross_patient.csv`

Do not replace their numbers with recollection from chat.

## Important correction frozen in this checkpoint

At threshold `0.90`:

- raw `stability`: 250 accepted, 1 false (`0.4%`);
- `baseline_agreement` / `combined_consistency`: 249 accepted, 0 false.

The `0 / 249` statement therefore belongs to agreement/combined consistency, not raw stability.

## Immediate next executable action

1. Audit registration between scan-953 ImageCAS-X anatomical annotations and the candidate original ImageCAS volume/mask.
2. Find and freeze the transform using masks / centerline geometry.
3. Verify centerline points fall inside the transformed coronary mask and report per-segment coverage.
4. Only then build CT+branch-aware pair/junction examples.
5. Re-run the same graph/selective protocol with image evidence added; no threshold retuning on the held-out patient.

## Deferred until matched CT+labels are valid

- CNN/ANZA cross-section encoder -> sequence model;
- Transformer vs Mamba comparison;
- ANZA incremental-value ablation;
- publication-level comparison with external repair baselines.
