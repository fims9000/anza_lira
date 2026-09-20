# Broken-mask context checkpoint — 2026-09-20

## Scope and caveat

This checkpoint tests whether spatial occupancy from the **synthetically broken binary ImageCAS-X coronary mask** adds useful information beyond endpoint geometry.

This is deliberately an intermediate proxy experiment.

It is **not**:

- matched CCTA intensity evidence;
- a predicted segmentation from an independent network;
- natural-gap clinical validation.

For each controlled scene, the hidden true gap / junction is removed before mask-context features are extracted. The model only sees the remaining binary lumen occupancy around the candidate region.

The official patient split remains frozen: 560 train / 80 validation / 160 test. Validation stress levels are 30 deg + 1 mm and 45 deg + 1 mm. Test thresholds are never selected on test patients.

## 1. Relation-existence signal is real

Geometry + broken-mask context improves PAIR/JUNCTION presence ranking.

Test AUROC:

| relation | geometry | geometry + broken mask |
|---|---:|---:|
| pair, 30 deg + 1 mm | 0.85434 | 0.92877 |
| pair, 45 deg + 1 mm | 0.81664 | 0.90014 |
| junction, 30 deg + 1 mm | 0.91004 | 0.93144 |
| junction, 45 deg + 1 mm | 0.86615 | 0.89455 |

Therefore occupancy context contains useful relation-existence information beyond tangent / distance geometry.

## 2. Presence heads at a validation structural-risk budget

At the validation-selected <=5% structural false budget, geometry + mask presence gives:

- test30 exact 0.56497, false 0.06497;
- test45 exact 0.51601, false 0.05979.

Against the canonical four-class relation head, the patient-cluster bootstrap shows:

- test30 exact delta +1.04 pp, 95% CI [-0.52, +2.58];
- test30 false delta -4.43 pp, 95% CI [-5.68, -3.20];
- test45 exact delta +0.24 pp, 95% CI [-1.03, +1.47];
- test45 false delta -2.07 pp, 95% CI [-3.12, -1.03].

Interpretation: broken-mask context is a useful **safety / existence** signal, but this experiment does not show a reliable gain in exact repair completion.

## 3. Canonical relation-head + mask-presence veto

A stricter experiment keeps the frozen canonical relation-type prediction and allows the mask presence head only to **remove** predicted relations. It can never add a PAIR or JUNCTION.

At the jointly validation-selected 5% structural false budget:

- pair veto threshold: 0.98;
- junction veto threshold: 0.85.

### Test30

Canonical relation head:

- exact 0.55461;
- false 0.10923;
- incomplete 0.33616.

Mask veto:

- exact 0.53861;
- false 0.05932;
- incomplete 0.40207.

Patient-cluster paired bootstrap, veto minus canonical:

- exact -1.60 pp, 95% CI [-2.66, -0.56];
- false -4.99 pp, 95% CI [-5.96, -4.07];
- incomplete +6.59 pp, 95% CI [+5.55, +7.66].

### Test45

Canonical:

- exact 0.51365;
- false 0.08051;
- incomplete 0.40584.

Mask veto:

- exact 0.49011;
- false 0.04991;
- incomplete 0.45998.

Patient-cluster paired bootstrap:

- exact -2.35 pp, 95% CI [-3.20, -1.53];
- false -3.06 pp, 95% CI [-3.84, -2.31];
- incomplete +5.41 pp, 95% CI [+4.50, +6.43].

This is a clear safety / abstention tradeoff rather than a new repair engine.

## 4. Subgroup diagnosis

The veto mainly succeeds by suppressing false positive repairs.

Test30 examples:

- none_orphan_pair: exact 0.95625 -> 1.00000, false 0.04375 -> 0;
- none_incomplete_junction: exact 0.90775 -> 0.95722, false 0.09225 -> 0.04278;
- pair_only: exact 0.14688 -> 0.00313, false 0.10625 -> 0.01250, incomplete 0.74688 -> 0.98438;
- degree-4 junctions: exact remains 0; false 0.36667 -> 0.30000.

Test45 shows the same pattern: NO-REPAIR scenes improve, while true pair scenes are strongly under-repaired.

Conclusion: mask occupancy is best interpreted as a conservative relation-presence veto / uncertainty cue. It does not solve branch identity in the difficult repair-needed strata.

## 5. Robust two-stress validation of the canonical selective policy

The relation-type confidence threshold and perturbation-consistency threshold were re-selected jointly on **both** validation stress levels, requiring false among accepted <=1% on val30 and val45.

Among the tested grid, the selected operating point is again:

- relation confidence tau = 0.85;
- perturbation consistency = 0.60.

Validation:

- val30 coverage 0.76362, false among accepted 0.00892;
- val45 coverage 0.82977, false among accepted 0.00821.

This matters because the canonical 0.85 + 0.60 rule is not merely a lucky val30 threshold; it survives the predefined two-stress robustness criterion.

## 6. Can mask veto replace perturbation consistency?

A joint validation search allowed:

- pair mask-veto threshold;
- junction mask-veto threshold;
- consistency threshold;

while keeping the canonical relation head fixed at tau 0.85 and requiring <=1% false among accepted on both validation stresses.

The selected solution was:

- pair veto = 0.97;
- junction veto = 0.96;
- consistency threshold = 0.0.

It accepts all scenes as decisions and uses the mask veto to suppress most questionable relations.

### Test30

Canonical 0.85 + 0.60:

- coverage 0.74341;
- false among accepted 0.00823;
- repair exact yield 0.07237.

Mask-veto hybrid:

- coverage 1.00000;
- false among accepted 0.00847;
- repair exact yield 0.06990.

### Test45

Canonical:

- coverage 0.81874;
- false among accepted 0.00863;
- repair exact yield 0.03783.

Mask-veto hybrid:

- coverage 1.00000;
- false among accepted 0.00847;
- repair exact yield 0.04359.

Patient-cluster bootstrap for repair exact yield:

- test30 delta -0.25 pp, 95% CI [-1.72, +1.16];
- test45 delta +0.58 pp, 95% CI [-0.57, +1.69].

Therefore mask veto can replace some of the consistency abstention while keeping overall false decision rate near 1%, but **there is no reliable improvement in actual repair exact yield**. Its apparent overall exact-yield gain is driven mainly by correctly deciding NO-REPAIR scenes.

## Frozen interpretation

The 800-case geometry/topology work now supports a fairly sharp conclusion:

1. candidate generation / ranking is already strong;
2. relation existence and branch identity are the dominant bottlenecks;
3. perturbation consistency is a strong risk-control layer;
4. broken binary-mask context improves relation-existence evidence;
5. broken-mask context is mainly useful as a veto / safety cue and does not reliably improve repair completion;
6. higher-degree junctions remain unresolved;
7. further tuning of geometry or binary-mask thresholds is now a diminishing-return direction.

The next high-value experiment is still matched **CCTA intensity + anatomical branch labels** under the frozen candidate generator, Graph-LIRA optimizer and selective protocol.

## Reproducibility artifacts

Results:

- `mask_veto_summary.csv`
- `mask_veto_validation_selected.csv`
- `mask_veto_bootstrap.csv`
- `mask_veto_key_subgroups.csv`
- `relation_multitau_robust_selected.csv`
- `selective_mask_veto_summary.csv`
- `selective_mask_veto_selected.csv`
- `selective_mask_veto_bootstrap.csv`

Scripts:

- `canonical_mask_veto.py`
- `mask_veto_subgroups.py`
- `select_multitau_robust.py`
- `selective_mask_veto_joint.py`
