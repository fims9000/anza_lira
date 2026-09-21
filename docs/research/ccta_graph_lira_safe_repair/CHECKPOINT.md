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

## Matched-CT blocker resolved — 2026-09-21

Six exact original ImageCAS CCTA volumes are now matched to ImageCAS-X anatomical labels under the official patient split:

- train: `953, 964`;
- validation: `957, 966`;
- test: `980, 984`.

All six pass exact CT/mask shape, spacing and affine checks. With the expected VTK LPS -> RAS conversion, centerline in-bounds, vessel occupancy and anatomical label agreement are all `1.0` for every case. Raw medical data remains outside Git; CT SHA256 hashes are committed.

Two real-CT controlled pilots are complete.

### Easy candidate-pair ranking

This setting is geometry-saturated on held-out test:

- geometry: AUROC `1.0000`, true-pair top-1 `1.0000`;
- CT only: AUROC `0.8892`, top-1 `0.8750`;
- geometry + CT: AUROC `0.9858`, top-1 `1.0000`.

This is a useful negative control: image context is not needed to solve this easy ranking problem.

### Geometry-matched wrong-branch relation-presence stress

Hard negatives are nearby pairs from different polylines and different anatomical segments, matched one-to-one to positives using geometry only before fitting any CT model. The model input never contains the hidden segment label or mask.

At the validation-selected operating point constrained to FPR <= 5%:

- validation geometry: recall `11.86%`, FPR `3.39%`;
- validation CT only: recall `50.85%`, FPR `1.69%`;
- validation geometry + CT: recall `74.58%`, FPR `3.39%`.

Held-out test:

- geometry: recall `1/47 = 2.13%`, false `0/47`;
- CT only: recall `22/47 = 46.81%`, false `0/47`;
- geometry + CT: recall `29/47 = 61.70%`, false `0/47`.

The test AUROC ordering does not favor CT (`geometry 0.9873` vs `geometry+CT 0.9746`), so the supported observation is specifically the conservative operating-point recall under a validation-frozen low-false policy, not a universal ranking improvement.

Patient heterogeneity is substantial: geometry+CT recall is `3/16` for scan 980 and `26/31` for scan 984. The observed `0/47` false count must not be interpreted as zero population risk or sub-1% risk.

Artifacts:

- `docs/research/ccta_graph_lira_safe_repair/MATCHED_CT_6CASE_CHECKPOINT.md`;
- `results/ccta_graph_lira_safe_repair/2026-09-21/matched_ct_alignment_6case.csv`;
- `results/ccta_graph_lira_safe_repair/2026-09-21/ct_candidate_pair_pilot_summary.csv`;
- `results/ccta_graph_lira_safe_repair/2026-09-21/ct_relation_geometry_matched_summary.csv`;
- `results/ccta_graph_lira_safe_repair/2026-09-21/ct_relation_geometry_matched_protocol.json`;
- archived exact scripts under `scripts/research/ccta_graph_lira_safe_repair/`.

## Current next executable action

Keep all test-set thresholds frozen. Add the matched-CT relation evidence to the same PAIR / JUNCTION / NO-REPAIR layer used by the 800-case Graph-LIRA benchmark, then evaluate it inside the frozen joint graph optimizer and the already validation-selected selective policy (`tau=0.85`, consistency `0.60`). Compare radial 2.5-D, candidate-aligned 3-D tube and a full cross-section encoder. ANZA remains a later explicit ablation, not an assumed improvement.

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

1. The matched-CCTA blocker is resolved for the six-case patient-level pilot.
2. Do not retune geometry, relation-confidence or consistency thresholds on scans 980/984.
3. Preserve the official train/validation/test patient separation.
4. Use real CCTA intensity to target relation existence / branch identity, not the already saturated easy candidate-ranking stage.
5. Integrate image-conditioned relation evidence into the frozen joint Graph-LIRA decision layer.
6. Quantify patient heterogeneity and uncertainty explicitly; six cases are pilot evidence only.
7. Expand to more original ImageCAS CTs before making publication-level population-risk or natural-gap claims.

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


## Representation and frozen-graph integration update — 2026-09-21

The matched six-case CCTA pilot was extended to compare image representations and to test direct insertion into the canonical four-class Graph-LIRA relation layer.

### Representation result

At the validation-defined FPR <=5% operating point on the geometry-matched wrong-branch stress:

- geometry: held-out recall `1/47 = 2.13%`, false `0/47`;
- radial 2.5-D: `29/47 = 61.70%`, false `0/47`;
- geometry + radial 2.5-D: `30/47 = 63.83%`, false `0/47`;
- coarse 3-D tube + PCA: `10/47 = 21.28%`, false `1/47`;
- geometry + coarse 3-D tube: `13/47 = 27.66%`, false `1/47`;
- small cross-section CNN: `9/47 = 19.15%`, false `0/47`.

Thus radial 2.5-D is the strongest current local CCTA representation. Larger learned representations are not justified with only two training patients.

### Canonical 800-case baseline reproduction

The complete geometry-only 800-case pipeline was independently regenerated from the archived source and full ImageCAS-X centerline package. The four-class relation head again selected `HGB, tau=0.85` on validation and reproduced:

- val30 exact `54.96%`, false `8.95%`;
- test30 exact `55.46%`, false `10.92%`;
- test45 exact `51.37%`, false `8.05%`.

This validates that the CT hybrid experiments use the same canonical geometry baseline.

### CT insertion result

A conservative CT auxiliary presence layer was trained only on scans 953/964 and calibrated only on 957/966. Geometry remained responsible for candidate identity / compatibility.

Direct CT add-only behavior on the matched subset:

- canonical test: exact `53.33%`, false `10.00%`;
- CT add-only test: exact `56.67%`, false `23.33%`.

A validation-only hysteresis search reduced validation false rate, but did not transfer:

- validation <=10% false setting: exact `57.14%`, false `3.57%`;
- held-out test: exact `50.00%`, false `20.00%`.

A setting matched to canonical validation risk similarly produced test false `20.00%`.

This is a negative integration result. CT is locally informative, but the relation-presence calibration is not patient-general enough with six matched cases to preserve the project safety objective. Do not retune on test scans 980/984.

Artifacts:

- `MATCHED_CT_REPRESENTATION_AND_GRAPH_CHECKPOINT.md`;
- `ct_representation_compare_test.csv`;
- `ct_graph_hybrid_summary.csv`;
- `ct_graph_hysteresis_selected.csv`;
- archived exact scripts `ct_representation_compare.py.gz.b64`, `ct_scene_graph_lira_hybrid.py.gz.b64`, and `ct_hybrid_hysteresis_search.py.gz.b64`.

### Exact resume action

Increase the number of exact original ImageCAS CTs matched to existing ImageCAS-X annotations while preserving the official split. Prefer cases already physically present in the downloaded Kaggle `801-1000.z04` multipart volume to avoid another large download. Then refit only the image-conditioned presence/calibration layer; keep the canonical geometry candidate models, four-class relation head, `tau=0.85`, and perturbation consistency `0.60` frozen.


## Expanded matched-CCTA execution checkpoint — 28 patients

The six-case calibration blocker has now been converted into a frozen 28-patient expansion protocol.

Official split preserved:

- train: 17 patients;
- validation: 5 patients;
- test: 6 patients.

Frozen relation set:

- 758 train rows;
- 268 validation rows;
- 334 test rows;
- 1,360 total balanced geometry-matched examples.

The exact radial 2.5-D extractor and frozen geometry metadata are committed on this branch. The runner has been compiled and launched successfully through initialization. The only current hard blocker is external raw ImageCAS data: `801-1000.z04` and the final split member `801-1000.change2zip` / `801-1000.zip` are not mounted in the current execution environment.

Do not change the experiment to work around this blocker. In particular:

- do not replace the exact original ImageCAS CT with BDMAP image-only cases;
- do not retune on scans 980 / 984;
- do not substitute the broken binary mask for CCTA intensity;
- do not escalate to a larger CNN / Transformer / ANZA before the 28-patient calibration check.

Canonical continuation after the raw archive becomes available:

1. run the frozen 28-patient radial extractor;
2. verify CT shape / spacing / affine / SHA256 for every patient;
3. freeze geometry vs radial CT vs geometry+radial CT;
4. compute per-patient and patient-cluster uncertainty;
5. refit only the image-conditioned relation-presence/calibration component;
6. insert it into frozen Graph-LIRA;
7. apply `tau=0.85` and consistency `0.60`;
8. audit repair-needed exact, false repair, incomplete decisions, risk-coverage, LAD/LCX and high-degree junctions;
9. only if calibration is stable, run the compact ANZA encoder ablation against radial CT and a matched conventional CNN.

Detailed executable roadmap:
`docs/research/ccta_graph_lira_safe_repair/EXPANDED_CT_28CASE_EXECUTION_AND_ROADMAP.md`.

Execution receipt:
`docs/research/ccta_graph_lira_safe_repair/EXPANDED_CT_28CASE_EXECUTION_STATUS.md`.


## Pre-CT 28-case geometry and promotion checkpoint

All analysis that can be completed without the original Kaggle z04 CCTA bytes has now been advanced.

Frozen 28-case relation baseline:

- 28 patients;
- 1,360 balanced relation examples;
- official 17 train / 5 validation / 6 test patient split;
- no patient leakage.

Strong held-out geometry references:

- nearest endpoint: recall 16.17%, FPR 4.19%;
- OGMC-style: recall 39.52%, FPR 3.59%;
- geometry HGB: recall 37.72%, FPR 1.20%, precision 96.92%.

For geometry HGB, patient-cluster bootstrap gives recall interval ~24.82–47.80% and FPR interval 0–2.74%.

The residual geometry problem is pre-registered anatomically. Hard positive groups are `OM1, IM, D2, LAD, OM2, R-PLA`; observed false-link sentinels are `LCX|OM2` and `R-PDA|RCA`.

A second safety operating point is frozen: pooled validation FPR <=5% plus maximum validation-patient FPR <=5%. It reduces geometry-HGB held-out FPR to 0.60% at recall 32.93%.

The expanded CT promotion gate is implemented and contract-tested. It:

1. rejects missing/constant/fake CT evidence;
2. verifies the exact 28-patient/1,360-row contract and all CT alignment gates;
3. selects the image representation from validation only;
4. compares against frozen geometry HGB, not the weak distance baseline;
5. runs paired patient-cluster bootstrap;
6. checks the pre-registered hard anatomy;
7. reports both pooled and patient-robust operating points;
8. authorizes Graph-LIRA integration only after the validation gate passes.

No real expanded-CCTA performance has been inferred. The only remaining external blocker for the real 28-case image stage is access to original `801-1000.z04` plus final `801-1000.zip`.


## Real 28-patient matched-CCTA result — promotion PASS

The raw `801-1000.z04` archive was reconstructed from ten Google Drive parts and the 28-patient matched-CCTA run is complete.

All 28 patients passed CT/ImageCAS-X shape, spacing and affine gates.

At the validation-selected FPR <= 5% operating point on six held-out test patients:

- geometry HGB: recall 37.72%, FPR 1.20%, precision 96.92%, AUROC 0.9685;
- radial 2.5-D CT: recall 68.86%, FPR 4.19%, precision 94.26%, AUROC 0.9440;
- geometry + radial CT: recall 82.63%, FPR 1.80%, precision 97.87%, AUROC 0.9847.

Paired patient-cluster bootstrap for geometry+CT minus geometry HGB:

- recall delta median +44.71 pp, 95% interval [+32.62,+60.00] pp;
- FPR delta median +0.61 pp, 95% interval [-1.29,+2.56] pp;
- AUROC delta median +0.0162, 95% interval [-0.0058,+0.0400].

The pre-registered promotion gate therefore PASSES. The CT relation signal may now be evaluated inside the frozen Graph-LIRA layer.

Do not retune test thresholds. Keep relation confidence `tau=0.85` and perturbation consistency `0.60` frozen.

Detailed result:
`REAL_CT28_RESULTS_AND_PROMOTION.md`

Immediate next executable action:
insert the calibrated CT relation evidence into the existing PAIR/JUNCTION/BOTH/NONE Graph-LIRA pipeline and evaluate repair-needed exact, false structural repair, incomplete/abstain, coverage and hard-anatomy strata under the frozen selective policy.
