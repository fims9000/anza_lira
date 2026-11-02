# CT28 Graph-LIRA: validation plan before held-out test

Historical work plan. Its early status paragraphs predate K10/15 and the CT28 path stage. Current results are in `artifacts/varvara/ct28_graph_lira/README.md`. Diagnostic scripts and outputs that were not selected are archived; restoration instructions are in `artifacts/varvara/ct28_graph_lira/ablations/rejected/README.md`.

Date: 2026-10-19
Scope: train/validation only. Held-out test remains unopened.
Current safety policy remains fixed: `tau=0.85`, consistency gate `0.60`.

## Goal

Select and freeze a validation configuration for the complete Graph-LIRA
pipeline before implementing the path stage and opening held-out test. The first
priority is to let CT participate in candidate identity selection, because the
current CT-conditioned baseline uses CT for relation type but still selects the
specific PAIR/JUNCTION candidates by `geometry_score`.

## 1. CT-conditioned candidate identity

Status: completed on validation on 2026-10-19 for Clean/Stress30 ×
Logistic/HGB local models. Artifacts and the full table are in
`artifacts/varvara/ct28_graph_lira/ablations/`. The separated CT-conditioned
relation probabilities remained unchanged. Fusion identity improved structural
exact before consistency in all four regimes. Clean Logistic accepted 14 exact
`pair_only` scenes; Clean HGB still accepted none. Stress30 increased coverage
for both local-model families but retained nonzero false repair. No
`junction_only` scene passed consistency in any regime, so JUNCTION consistency
remains the next diagnostic target.

Keep the scene pack, relation heads, compatibility rules, thresholds and
perturbation protocol unchanged. Compare:

1. `geometry_score` for identity selection — current baseline;
2. `geometry_plus_ct_score` for identity selection — CT identity ablation.

For `PAIR` and `JUNCTION`, select the candidate with the largest configured
identity score. For `BOTH`, select the endpoint-disjoint PAIR/JUNCTION
combination with the largest sum of log identity scores.

Evaluate on validation:

- candidate top-1 for PAIR and JUNCTION;
- structural exact before consistency;
- false structural repair before consistency;
- type, PAIR, JUNCTION and exact-set consistency;
- coverage and structural metrics after the frozen gates;
- results by scene type, patient, `competitive/isolated` and
  geometry-ambiguous subgroup.

This is an identity-only ablation: it must not retrain or otherwise change the
relation head.

## 2. Decompose perturbation consistency

Status: completed on validation on 2026-10-19 for Clean HGB primary and Clean
Logistic control, both with the separated CT-conditioned relation head and
`geometry_plus_ct_score` identity. Report:
`artifacts/varvara/ct28_graph_lira/consistency_diagnostics/README.md`.
The primary HGB bottleneck is relation-type stability: proposed scenes retain
their post-`tau` type in 29.33% of repeats, while exact candidate identity among
repeats that preserve the type is 61.63%. Confidence and endpoint compatibility
do not explain the zero coverage.

For every validation scene and perturbation repeat, save:

- predicted relation type and confidence;
- selected PAIR candidate ID;
- selected JUNCTION candidate ID;
- final selected candidate-ID set;
- whether the selected `BOTH` combination is compatible.

Calculate separately:

1. type consistency;
2. PAIR identity consistency;
3. JUNCTION identity consistency;
4. exact candidate-set consistency;
5. compatibility consistency;
6. confidence and consistency distributions.

Run the same diagnostics for the main Logistic local-score baseline and the
saved local-HGB ablation, for geometry-only and CT-conditioned relation heads.
Do not alter `tau` or the consistency threshold.

## 3. Record explicit abstention reasons

Status: completed on validation on 2026-10-19 for the same clean HGB primary
and clean Logistic control used in the consistency diagnostic. Report:
`artifacts/varvara/ct28_graph_lira/consistency_diagnostics/README.md`.
Every model-scene row has one deterministic primary reason. Among 206
repair-needed scenes, relation-type instability explains 162 HGB rejections
(78.64%); Logistic is split mainly between relation-type instability (76) and
PAIR-identity instability (69). The classification changes no predictions or
gates.

Assign each validation scene one primary outcome/rejection reason:

- `accepted`;
- `correct_none`;
- `confidence_below_tau`;
- `no_repair_proposed`;
- `relation_type_unstable`;
- `pair_identity_unstable`;
- `junction_identity_unstable`;
- `both_combination_unstable`;
- `compatibility_failure`.

Report structural metrics both before and after consistency. Break the results
down by model, scene type, patient and JUNCTION group. The purpose is to explain
why local HGB has high structural exact before consistency but zero final
coverage, and whether CT changes type stability or candidate identity stability.

## 4. Check scene-count shortcuts

Use the same scenes, local OOF scores, relation-model family and validation
split. Compare:

1. full historical scene representation:
   - 42 features for a one-stream head;
   - 84 features for the separated geometry/CT head;
2. representation without explicit cardinality columns:
   - 38 features for a one-stream head;
   - 76 features for the separated geometry/CT head.

Remove only explicit `PAIR/JUNCTION candidate_count` and their repeated count
columns. Do not redesign the scene generator or remove score-distribution
features in this diagnostic.

Compare relation accuracy, balanced accuracy, macro-F1, log loss, confusion
matrix and downstream structural metrics. Keep the count-only train-to-validation
lookup as a diagnostic reference. Also inspect results within scene families and
candidate-count strata to determine whether the relation head is learning scene
construction shortcuts.

## 5. Failure analysis

For every incorrect or rejected validation scene record:

- patient and scene family;
- true and predicted relation type;
- confidence;
- best true and false candidate scores;
- selected and ground-truth candidate IDs;
- all consistency components;
- compatibility result;
- abstention reason;
- candidate counts;
- geometry-ambiguous and `competitive/isolated` labels.

Review especially:

- correct relation type with wrong candidate identity;
- CT correcting type but not identity;
- CT changing a correct identity into an incorrect one;
- `BOTH` failures caused by endpoint compatibility;
- correct base decisions rejected only by exact-identity consistency.

## 6. Select and freeze the validation configuration

Choose, using train/validation only:

- local-score family: Logistic baseline or the documented HGB ablation;
- full historical or no-cardinality relation representation;
- geometry-only or separated CT-conditioned relation evidence;
- geometry or geometry+CT identity score;
- perturbation-training regime;
- exact consistency definition used for acceptance.

Selection priority:

```text
low false structural repair
-> structural exact
-> coverage
-> stability across patients
```

Keep `tau=0.85` and consistency gate `0.60` unchanged. Save the configuration,
feature lists, patient folds, models, compatibility rule, perturbation settings,
identity rule, SHA256 values, validation metrics and reasons for rejecting other
configurations.

## 7. Implement and validate the path stage

Implement the path stage before opening held-out test so test is evaluated by a
complete frozen pipeline:

```text
local evidence
-> relation head
-> candidate identity
-> global compatibility
-> confidence and consistency
-> repair or abstain
-> path construction
```

On train/validation verify that paths connect the intended endpoints, preserve
the selected candidate identity, introduce no extra relation, handle `NONE`
without repair, and support `PAIR`, `JUNCTION` and `BOTH`.

## 8. Open held-out test once

After the validation protocol and path stage are frozen:

- generate test scenes with the same protocol;
- extract CT evidence with the same representation;
- apply frozen local and relation models;
- apply unchanged identity, compatibility, `tau=0.85`, consistency `0.60` and
  path rules;
- do not tune anything from test results.

Save overall structural and path metrics, per-patient results, patient-cluster
bootstrap intervals, scene-type results, `competitive/isolated`,
geometry-ambiguous and risk/coverage results.

## 9. Compare CT encoders after the full Graph-LIRA result

On the same frozen candidates, graph architecture, split and selective policy,
compare:

1. handcrafted radial CT representation;
2. compact CNN encoder;
3. ANZA encoder.

Only the CT evidence encoder may change in this comparison. Scene generation,
relation semantics, identity/compatibility rules and evaluation protocol remain
fixed.
