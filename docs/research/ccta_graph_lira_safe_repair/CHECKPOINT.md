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
9. Wrong-branch ambiguity is present on more than one labelled heart: controlled centerline gaps with a geometrically plausible candidate from another anatomical segment occurred in about 13.4% of scan 921 gaps and 12.6% of scan 953 gaps under the frozen audit rule.
10. The ImageCAS-X scan-953 labels are internally coherent with their VTK centerlines.
11. The previously tested BDMAP_00015590 CT/mask was a **provisional third-party row-index mapping and is rejected** as the source image for ImageCAS-X scan 953. The official ImageCAS-X description states that patient IDs are identical to original ImageCAS IDs, so the required source is the original ImageCAS volume belonging to scan ID 953.

## Strongest new structural evidence: cross-patient transfer

Controlled `30 deg + 1 mm` benchmark:

### Train 921 -> test 953

- pair AUROC: `0.98269`;
- local independent false-scene rate: `53.70%`;
- sequential junction -> pair: exact `87.04%`, false `8.80%`;
- joint Graph-LIRA: exact `94.44%`, false `1.85%`.

Perturbation consistency `>= 0.90`:

- coverage `60.19%`;
- `130` accepted scenes;
- `0 / 130` false among accepted;
- `99.23%` exact among accepted.

### Train 953 -> test 921

- pair AUROC: `0.98136`;
- local independent false-scene rate: `54.36%`;
- sequential junction -> pair: exact `80.54%`, false `9.40%`;
- joint Graph-LIRA: exact `88.59%`, false `2.68%`.

Perturbation consistency `>= 0.90`:

- coverage `53.69%`;
- `80` accepted scenes;
- `0 / 80` false among accepted;
- `100%` exact among accepted.

These are finite-sample controlled centerline stress results, not clinical natural-gap validation.

At `45 deg + 1 mm`, joint Graph-LIRA still degrades:

- 921 -> 953: exact `87.96%`, false `6.94%`;
- 953 -> 921: exact `85.91%`, false `12.08%`.

This defines the residual regime where image evidence should be tested.

## Branch-level failure localization

At `30 deg + 1 mm` the residual errors are not uniform.

- held-out 953 D1: `60/60` exact;
- held-out 953 RCA: `30/30` exact;
- held-out 953 LAD degree-3: exact `86.67%`, false `6.67%`;
- held-out 953 LCX degree-3: exact `89.39%`, false `3.03%`;
- held-out 921 LAD degree-3: exact `96.61%`, false `0%`;
- held-out 921 LCX degree-3: exact `93.33%`, false `3.33%`;
- held-out 921 LCX degree-4: exact only `63.33%`, false `6.67%`, incomplete-but-nonfalse `30%`.

For the degree-4 LCX stratum, only about `36.7%` of scenes pass consistency `>= 0.80`, although accepted scenes have no false structural links in this run. This is the clearest target for CT-conditioned evidence.

## Exact artifact-backed numbers currently preserved

Previous-session artifacts:

- `results/ccta_graph_lira_safe_repair/2026-09-20/perturbation_risk_coverage_30deg_strong.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/perturbation_scene_uncertainty_30deg_strong.csv.gz.b64`
- `results/ccta_graph_lira_safe_repair/2026-09-20/sequence_summary.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/sequence_cross_patient.csv`

Geometry / data audits:

- `results/ccta_graph_lira_safe_repair/2026-09-20/alignment_953_report.json`
- `results/ccta_graph_lira_safe_repair/2026-09-20/alignment_953_centerline_segment_coverage.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/branch_ambiguity_cross_patient.json`
- `results/ccta_graph_lira_safe_repair/2026-09-20/branch_ambiguity_921_by_segment.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/branch_ambiguity_953_by_segment.csv`

Cross-patient Graph-LIRA:

- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_summary.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_risk_coverage.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_settings.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_protocol.json`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_branch_strata_30deg.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_failure_cases_30deg.csv`

Exact exploratory sources are archived under:

- `scripts/research/ccta_graph_lira_safe_repair/*.py.gz.b64`

Do not replace machine-artifact numbers with recollection from chat.

## Important perturbation correction frozen in this checkpoint

In the earlier within-case artifact at threshold `0.90`:

- raw `stability`: 250 accepted, 1 false (`0.4%`);
- `baseline_agreement` / `combined_consistency`: 249 accepted, 0 false.

The `0 / 249` statement therefore belongs to agreement/combined consistency, not raw stability.

## Scan-953 source / alignment result

The failed BDMAP alignment is preserved because it prevents us from accidentally reusing the wrong CT.

The ImageCAS-X centerlines align to `953.coronary.nii.gz` with median distance about `0.12 mm` and 100% of sampled centerline points within 1 mm after LPS -> RAS conversion.

The rejected BDMAP candidate gave:

- exact Dice only `0.0176`;
- only `3.65%` of transformed ImageCAS-X vessel voxels within 1 mm of the candidate mask;
- left / right centerline coverage within 1 mm about `2.26%` / `0%`;
- free rigid ICP median surface distance about `6.05 mm`.

The official ImageCAS-X source description says patient IDs are the original ImageCAS IDs and the benchmark layout uses `volumes/<scan_id>.img.nii.gz`. Therefore the next image target is the original ImageCAS scan ID `953`.

Full details: `docs/research/ccta_graph_lira_safe_repair/ALIGNMENT_953.md`.

## Current next executable action

### Can continue without new image data

- keep the cross-patient structural benchmark frozen;
- analyse failure strata and candidate ambiguity around LAD/LCX/high-degree junctions;
- use those strata as the fixed target set for the future image-context experiment.

### Hard blocker for the image-context question

To test **real CT + anatomical branch identity**, obtain the original ImageCAS CCTA volume corresponding to scan ID `953` (or any other ImageCAS-X-labelled scan for which the same original ImageCAS CT is available).

Do not substitute a row-indexed BDMAP mirror.

## After the matched CT is available

1. verify CT and ImageCAS-X mask geometry;
2. freeze patient-level pair/junction examples and hard strata;
3. compare geometry-only vs radial 2.5-D vs candidate-aligned 3-D tube vs full cross-section encoder -> sequence context;
4. feed all local scores into the same joint Graph-LIRA optimizer;
5. evaluate perturbation-consistency risk/coverage without threshold retuning on held-out patients;
6. only then test ANZA as an incremental local encoder and, if justified, Transformer/Mamba sequence aggregation.
