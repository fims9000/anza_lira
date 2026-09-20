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
9. The wrong-branch ambiguity is present on more than one labelled heart: controlled centerline gaps with a geometrically plausible candidate from another anatomical segment occurred in about 13.4% of scan 921 gaps and 12.6% of scan 953 gaps under the frozen audit rule.
10. The scan-953 ImageCAS-X labels are internally coherent with their VTK centerlines, but the currently available original ImageCAS BDMAP_00015590 CT/mask is **not geometrically aligned strongly enough** to be used as matched CT evidence.

## Exact artifact-backed numbers currently preserved

Previous-session artifacts:

- `results/ccta_graph_lira_safe_repair/2026-09-20/perturbation_risk_coverage_30deg_strong.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/perturbation_scene_uncertainty_30deg_strong.csv.gz.b64`
- `results/ccta_graph_lira_safe_repair/2026-09-20/sequence_summary.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/sequence_cross_patient.csv`

New canonical audits:

- `results/ccta_graph_lira_safe_repair/2026-09-20/alignment_953_report.json`
- `results/ccta_graph_lira_safe_repair/2026-09-20/alignment_953_centerline_segment_coverage.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/branch_ambiguity_cross_patient.json`
- `results/ccta_graph_lira_safe_repair/2026-09-20/branch_ambiguity_921_by_segment.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/branch_ambiguity_953_by_segment.csv`

Do not replace their numbers with recollection from chat.

## Important correction frozen in this checkpoint

At perturbation threshold `0.90`:

- raw `stability`: 250 accepted, 1 false (`0.4%`);
- `baseline_agreement` / `combined_consistency`: 249 accepted, 0 false.

The `0 / 249` statement therefore belongs to agreement/combined consistency, not raw stability.

## Scan-953 alignment result

The coordinate audit is complete enough to reject the unsafe shortcut.

The ImageCAS-X centerlines align to `953.coronary.nii.gz` with median distance about 0.12 mm and 100% of sampled centerline points within 1 mm after LPS -> RAS conversion.

In contrast, mapping the ImageCAS-X mask to the candidate original ImageCAS BDMAP_00015590 mask gives:

- exact Dice only `0.0176`;
- only `3.65%` of transformed ImageCAS-X vessel voxels within 1 mm of the candidate original mask;
- left / right centerline coverage within 1 mm of the candidate mask about `2.26%` / `0%`;
- an additional free rigid ICP surface fit still leaves median distance about `6.05 mm`.

Therefore raw indexing or simple rigid alignment is not defensible. Do not train CT+branch models from this pair.

Full details: `docs/research/ccta_graph_lira_safe_repair/ALIGNMENT_953.md`.

## Current next executable action

### Can continue without new data

Use the now-committed branch-aware ambiguity harness to expand structural tests on scans 921 and 953 and prepare the common candidate / graph-evaluation protocol.

### Hard blocker for the image-context question

To test **real CT + anatomical branch identity** we need one of:

1. the CT volume actually distributed with ImageCAS-X for scan 953; or
2. the official ImageCAS-X transform/source mapping that maps its labelled coronary geometry to the original ImageCAS image volume.

Until that is available, do not pretend the original BDMAP CT is matched.

## After the matched CT is available

1. freeze the transform and per-segment centerline coverage;
2. construct patient-level pair/junction examples;
3. compare geometry-only vs radial 2.5-D vs candidate-aligned 3-D tube vs full cross-section encoder -> sequence context;
4. feed all local scores into the same joint Graph-LIRA optimizer;
5. evaluate perturbation-consistency risk/coverage without threshold retuning on held-out patients;
6. only then test ANZA as an incremental local encoder and, if justified, Transformer/Mamba sequence aggregation.
