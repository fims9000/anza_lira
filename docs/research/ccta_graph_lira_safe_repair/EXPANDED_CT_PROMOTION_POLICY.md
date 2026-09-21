# Frozen promotion policy for expanded CCTA evidence

Date: 2026-09-21
Branch: `research/ccta-graph-lira-safe-repair`

This policy is frozen before the real 28-patient CCTA feature extraction is available.

Its purpose is to prevent a promising test number from silently changing the research question or the acceptance criteria.

## Reference geometry baseline

The strongest pre-CT local safety baseline is `geometry_hgb` on the same frozen 1,360-row relation plan.

Validation operating point selected under FPR <= 5%:

- recall 32.09%
- FPR 4.48%

Held-out six-patient test at that frozen threshold:

- recall 37.72%
- FPR 1.20%
- precision 96.92%
- AUROC 0.9685

The CT model must therefore be compared primarily with geometry-HGB, not with nearest endpoint.

## Stage 0 — evidence integrity

No CT result can be promoted if any of the following fails:

- 28 unique patients;
- 17 train / 5 validation / 6 test;
- 1,360 frozen relation rows;
- 379/379 train, 134/134 validation, 167/167 test class counts;
- no patient split leakage;
- all 28 CT shape / spacing / affine gates pass;
- radial CT features are present and non-constant.

The non-constant feature check is deliberate: the synthetic contract fixture contains zero CT features and must never be accepted as scientific evidence even if its mock prediction columns look excellent.

## Stage 1 — validation-only representation gate

Candidates:

- radial 2.5-D CT;
- geometry + radial 2.5-D CT.

A CT representation becomes eligible for the **next Graph-LIRA integration experiment** only when, on validation patients:

1. its frozen validation threshold obeys FPR <= 5%;
2. its recall is strictly greater than geometry-HGB recall at the frozen geometry operating point.

If both candidates pass, choose by:

1. higher validation recall;
2. lower validation FPR;
3. higher validation precision.

No test quantity is used for this choice.

Passing Stage 1 is permission to run the next experiment. It is **not** an improvement claim.

## Stage 2 — held-out local confirmation

After the representation is selected from validation, report unchanged on the six held-out test patients:

- AUROC / AUPRC;
- recall;
- FPR;
- precision;
- patient-level metrics;
- paired patient-cluster bootstrap versus geometry-HGB;
- the pre-registered hard-anatomy strata.

Useful held-out support means:

- recall above geometry-HGB;
- FPR remains <= 5%.

A stronger descriptive support flag additionally asks for patient-cluster bootstrap evidence that recall delta is positive in at least 95% of resamples and FPR-worsening probability is no more than 20%.

This stronger flag is descriptive; it is not used to retune the model.

## Stage 3 — hard-anatomy confirmation

The subgroup definitions are frozen in `PREREGISTERED_HARD_ANATOMY_TARGETS.md`.

Hard positive set:

`OM1, IM, D2, LAD, OM2, R-PLA`

Observed geometry-HGB false-link sentinels:

- `LCX|OM2`;
- `R-PDA|RCA`.

Additional adjacent-branch monitoring groups:

- `D1|LAD`;
- `D2|LAD`;
- `LCX|OM1`;
- `LCX|LM`;
- `R-PLA|RCA`.

These are reported regardless of whether CT improves them.

## Stage 4 — frozen Graph-LIRA insertion

Only a Stage-1 eligible representation is inserted into the relation layer.

Remain frozen:

- geometry candidate generation;
- geometry candidate identity / compatibility evidence;
- global Graph-LIRA structural optimization;
- relation confidence `tau=0.85`;
- perturbation consistency gate `0.60`;
- test thresholds.

The CT component may recalibrate relation-presence evidence on train/validation. It must not alter the held-out test policy.

## Stage 5 — Graph-LIRA improvement claim

Do not call the CT-conditioned system an improved Graph-LIRA model merely because local relation recall improves.

A graph-level improvement claim requires the held-out graph evaluation to show a Pareto improvement over frozen canonical behavior:

- structural exact improves;
- false structural-repair rate does not increase.

If CT improves coverage but also increases false structural repair, report it as a calibration failure, as happened in the six-case pilot.

If CT reduces false repair but lowers repair recall, report it as a selective safety / veto result rather than a general repair improvement.

## ANZA gate

A learned ANZA encoder is justified after the 28-patient radial experiment only if the result clarifies a representation bottleneck.

- If radial CT transfers well: ANZA must be compared directly with radial CT and a compact conventional CNN under the exact same patients, pairs, Graph-LIRA and risk policy.
- If radial CT is locally strong but graph calibration fails: fix calibration / patient normalization / negative diversity before increasing encoder capacity.
- If radial CT itself does not transfer: candidate-aligned 3-D or full cross-section encoders become justified targets; ANZA can be tested as one compact local encoder.

## Executable gate

Exact source:

`scripts/research/ccta_graph_lira_safe_repair/evaluate_expanded_ct_promotion_gate.py.gz.b64`

Source SHA256:

`631591ebb7f152c3fdcf56d8b03ca22ef2561ca9821b3e137bebe9003b8f6c00`

The script compares a real `GraphLIRA_CT_expanded_radial_results.zip` against the frozen geometry-HGB predictions and generates:

- `promotion_gate.json`;
- `PROMOTION_GATE.md`;
- `ct_vs_geometry_metrics.csv`;
- `ct_vs_geometry_anatomy_subgroups.csv`;
- paired patient-cluster bootstrap results.

A synthetic contract test was run in both failure and pass modes. The zero-CT synthetic fixture is correctly rejected as `BLOCKED_OR_INVALID`; a deliberately non-constant synthetic fixture exercises the positive control. Neither fixture is scientific evidence.
