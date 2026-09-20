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

Canonical V2 perturbation consistency `>= 0.90`:

- coverage `63.43%`;
- `137` accepted scenes;
- `0 / 137` false among accepted;
- `99.27%` exact among accepted.

### Train 953 -> test 921

- pair AUROC: `0.98136`;
- local independent false-scene rate: `54.36%`;
- sequential junction -> pair: exact `80.54%`, false `9.40%`;
- joint Graph-LIRA: exact `88.59%`, false `2.68%`.

Canonical V2 perturbation consistency `>= 0.90`:

- coverage `44.97%`;
- `67` accepted scenes;
- `0 / 67` false among accepted;
- `98.51%` exact among accepted.

These are finite-sample controlled centerline stress results, not clinical natural-gap validation.

At `45 deg + 1 mm`, joint Graph-LIRA still degrades:

- 921 -> 953: exact `87.96%`, false `6.94%`;
- 953 -> 921: exact `85.91%`, false `12.08%`.

At the deliberately severe `60 deg + 1.5 mm` stress:

- 921 -> 953: exact `75.00%`, false `18.98%`;
- 953 -> 921: exact `69.80%`, false `28.19%`.

This exposes the geometry ceiling and defines the residual regime where image evidence should be tested.

## Branch-level failure localization

At `30 deg + 1 mm` the residual errors are not uniform.

- held-out 953 D1: `60/60` exact;
- held-out 953 RCA: `30/30` exact;
- held-out 953 LAD degree-3: exact `86.67%`, false `6.67%`;
- held-out 953 LCX degree-3: exact `89.39%`, false `3.03%`;
- held-out 921 LAD degree-3: exact `96.61%`, false `0%`;
- held-out 921 LCX degree-3: exact `93.33%`, false `3.33%`;
- held-out 921 LCX degree-4: exact only `63.33%`, false `6.67%`, incomplete-but-nonfalse `30%`.

For the degree-4 LCX stratum, canonical V2 gives consistency mean `0.647`, median `0.60`, and only `10 / 30 = 33.33%` of scenes pass consistency `>= 0.80`. This is the clearest target for CT-conditioned evidence.

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

Canonical cross-patient Graph-LIRA V2:

- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_summary.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_risk_coverage.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_settings.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_protocol.json`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_branch_strata_30deg.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_failure_cases_30deg.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/selective_risk_exact_ci_v2.csv`

The older cross-patient files without `v2` are retained only as provenance. V1 perturbation consistency depended on a global scene index; V2 uses stable scene IDs and is invariant to adding/removing other stress configurations.

Exact exploratory sources are archived under:

- `scripts/research/ccta_graph_lira_safe_repair/*.py.gz.b64`

Do not replace machine-artifact numbers with recollection from chat.

Reproducibility correction: `docs/research/ccta_graph_lira_safe_repair/REPRODUCIBILITY_FIX_V2.md`.

Finite-sample risk interpretation: `docs/research/ccta_graph_lira_safe_repair/SELECTIVE_RISK_UNCERTAINTY.md`.

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

- keep canonical V2 frozen;
- keep the anatomy-based hard strata frozen before CT-model development;
- use the existing failure table to define diagnostics, not to retune the held-out patient;
- preserve the distinction between observed zero failures and the much wider finite-sample risk bound.

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

## Dataset provenance checkpoint

The four CT + binary-mask image-context cases have now been re-verified by SHA256 against the exact Hugging Face mirror objects:

- `BDMAP_00015590`
- `BDMAP_00015593`
- `BDMAP_00015594`
- `BDMAP_00015597`

The earlier script labels `953/956/957/960` were row-derived convenience labels from the mirror's `ImageCAS_ID.txt`, not ImageCAS-X patient identities. A corrected result copy is stored as:

- `results/ccta_graph_lira_safe_repair/2026-09-20/sequence_cross_patient_bdmap_ids.csv`

Resume documentation:

- `docs/research/ccta_graph_lira_safe_repair/DATA_SOURCES.md`
- `docs/research/ccta_graph_lira_safe_repair/FOUR_CASE_IMAGECAS_PILOT.md`

This correction does not invalidate the image-only representation experiments; it corrects patient/source naming and prevents an invalid join to ImageCAS-X branch annotations.

## 800-case repair-aware correction

The full 800-case ImageCAS-X benchmark changes the interpretation of selective Graph-LIRA.

At strict relation-type consistency gate `0.70`, test30 accepts many scenes safely, but the accepted repair-needed strata remain mostly incomplete:

- pair-only: `66.25%` coverage, `0%` exact among accepted;
- junction-only: `43.58%` coverage, `2.45%` exact among accepted;
- mixed: `45.27%` coverage, `29.85%` exact among accepted.

By contrast, accepted NO-REPAIR scenes are almost entirely correct. Therefore selective consistency currently proves **risk control / stable abstention**, not high automatic repair recall.

A relation-presence oracle shows where the opportunity lies. On test30 repair-needed scenes, the current relation-type system is exact on `28.45%`, while oracle relation presence plus the existing geometry ranker reaches `70.81%` at top-1 and `94.57%` at top-3. At test45 the corresponding values are `18.50%`, `62.75%`, and `89.97%`.

This freezes the geometry-only conclusion: candidate generation/ranking is not the dominant bottleneck. **PAIR / JUNCTION / NO-REPAIR existence and branch identity are.**

Artifacts:

- `results/ccta_graph_lira_safe_repair/2026-09-20/large_scale_repair_aware_diagnostics.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/large_scale_selective_paired_bootstrap.csv`

## Current executable direction

1. Do not tune geometry thresholds further on the inspected test set.
2. Keep official patient splits, candidate generator, graph optimizer and uncertainty protocol frozen.
3. Run an intermediate binary-lumen **shape-context** experiment using synthetically broken masks with the hidden relation removed before feature extraction. The model may use only the observed binary vessel geometry, never anatomical segment labels.
4. Ask whether that spatial context improves relation existence / repair-needed exactness at the same false-repair budget.
5. Treat this only as a proxy for image evidence.
6. Final CT-conditioned work still requires true original ImageCAS `<scan_id>.img.nii.gz` volumes matched to ImageCAS-X IDs.

## Broken-mask context and robust selective-policy checkpoint

The full 800-case benchmark was extended with a controlled broken-binary-mask context experiment.

The mask is created by removing the hidden synthetic gap / junction before feature extraction. It is therefore a valid controlled occupancy-context test, but it is **not** CCTA intensity evidence and not an independent predicted segmentation.

Main findings:

1. geometry + broken-mask context improves PAIR / JUNCTION presence ranking substantially;
2. at a validation-selected 5% structural false budget, presence heads reduce false structural decisions relative to the canonical relation head but do not show a reliable exact-rate gain;
3. using the mask presence head strictly as a veto gives a strong safety / abstention tradeoff:
   - test30 false `0.10923 -> 0.05932`, exact `0.55461 -> 0.53861`;
   - test45 false `0.08051 -> 0.04991`, exact `0.51365 -> 0.49011`;
4. paired patient-cluster bootstrap confirms the false reduction and the accompanying increase in incompleteness;
5. the veto mostly improves NO-REPAIR scenes and heavily under-repairs pair-only scenes;
6. degree-4 junctions remain a hard unresolved stratum.

A further joint safe-policy search allowed pair-mask threshold, junction-mask threshold and perturbation consistency to vary while requiring <=1% false among accepted on both validation stress levels. It selected `pair=0.97`, `junction=0.96`, `consistency=0.0`. On test, this keeps false decision rate near 1% but does not yield a statistically reliable repair-exact improvement versus the canonical selective policy.

Robust multi-tau validation independently re-selected the original canonical policy:

- relation confidence `tau=0.85`;
- perturbation consistency `0.60`.

Validation false among accepted:

- val30 `0.00892`;
- val45 `0.00821`.

Thus the 0.85 + 0.60 operating point is now frozen by a two-stress validation criterion, not by test inspection.

Detailed interpretation and artifact map:
`docs/research/ccta_graph_lira_safe_repair/MASK_CONTEXT_CHECKPOINT.md`.

New artifacts:

- `results/ccta_graph_lira_safe_repair/2026-09-20/mask_veto_summary.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/mask_veto_validation_selected.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/mask_veto_bootstrap.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/mask_veto_key_subgroups.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/relation_multitau_robust_selected.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/selective_mask_veto_summary.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/selective_mask_veto_selected.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/selective_mask_veto_bootstrap.csv`

The central image-conditioned research question remains externally blocked until a true original ImageCAS CT is matched to an ImageCAS-X anatomical ID. Further geometry / binary-mask threshold tuning should not replace that experiment.
