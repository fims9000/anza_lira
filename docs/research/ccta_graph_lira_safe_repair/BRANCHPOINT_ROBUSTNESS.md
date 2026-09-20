# Branch-point robustness of joint Graph-LIRA

Date: 2026-09-20.

Status: descriptive robustness analysis of the canonical V2 30 deg + 1 mm cross-patient stress benchmark.

## Question

Is the aggregate improvement of joint Graph-LIRA driven by one unusually easy branch point, or is the same direction visible across the individual held-out branch points?

## Result

There are 13 usable branch-point groups in the primary transfer benchmark:

- 8 groups when training on 921 and testing 953;
- 5 groups when training on 953 and testing 921.

Compared with the sequential junction-then-pair strategy:

- joint Graph-LIRA has a **non-negative reduction in false-scene rate on all 13 branch-point groups**;
- it strictly reduces false-scene rate on 12 / 13 groups;
- the only tie is the difficult degree-4 LCX group, where both have 6.67% false scenes;
- joint Graph-LIRA has a **positive exact-scene gain on all 13 groups**.

Compared with independent local pair decisions, joint Graph-LIRA reduces false-scene rate on every group, usually by tens of percentage points.

This matters because the main aggregate result is not produced by one branch. The direction of improvement is consistent across the available branch-point groups.

## Paired scene discordances: joint vs sequential

These counts are descriptive because controlled scenes within a branch point are not independent clinical observations.

### Train 921 -> test 953

False structural decisions:

- sequential false / joint not false: 16 scenes;
- joint false / sequential not false: 1 scene;
- both false: 3 scenes.

Exact structural decisions:

- joint exact / sequential not exact: 16 scenes;
- sequential exact / joint not exact: 0 scenes;
- both exact: 188 scenes.

### Train 953 -> test 921

False structural decisions:

- sequential false / joint not false: 13 scenes;
- joint false / sequential not false: 3 scenes;
- both false: 1 scene.

Exact structural decisions:

- joint exact / sequential not exact: 12 scenes;
- sequential exact / joint not exact: 0 scenes;
- both exact: 120 scenes.

## Limitation

There are only two patients and a small number of branch-point groups. This is a robustness check inside the controlled benchmark, not an independent population-level statistical test.

The future CT-conditioned study must preserve patient-level splits and add new patients rather than generating more perturbations from the same two trees and treating them as independent evidence.

## Machine artifact

`results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_branchpoint_effects_30deg.csv`
