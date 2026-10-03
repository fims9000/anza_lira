# CT28 JUNCTION generator review and frozen-next decisions — 2026-10-03

This note reviews Varvara's current `build_junction_plan.py` using only train/validation data. Held-out test junction candidates have not been generated or inspected.

## Reproduced current draft

With:

- target cut = 2.0 mm
- max arm fraction = 0.9
- max span = 16 mm
- decoy step = 1.5 mm
- decoy endpoint margin = 2.0 mm
- max one-arm negatives = 12 / positive
- max two-arm negatives = 6 / positive

the draft reproduces:

- 144 train/val true junctions;
- 143 degree-3 + 1 degree-4;
- 143 degree-3 positives in the selected plan;
- 1,666 selected negatives;
- 1,094 one-arm negatives;
- 572 two-arm negatives;
- 7.9% positives;
- 20 selected negatives with same segment label fallback;
- raw generated negative pool = 143,116 candidates.

The code correctly keeps held-out test out of the development generator.

## Important diagnostic: current negatives are too easy for geometry

A train-only geometry sanity check was run on the nine existing `jg_*` features and evaluated on validation.

This is only a generator diagnostic, not a paper result.

- LogisticRegression: validation AUROC ~0.9951; top-1 positive rank 96.4% on non-trivial validation junctions.
- HistGradientBoosting: validation AUROC ~0.9993; top-1 positive rank 100% on non-trivial validation junctions.

Therefore the current negative selection should NOT be frozen yet for JUNCTION+CT. If geometry alone almost perfectly separates the selected negatives, the later CT comparison has little room to test the intended hypothesis.

The main reason is visible in the current selection key:

```python
(-all_decoy_labels_different, geometry_match_distance, candidate_id)
```

Different-label candidates are always preferred before the closest geometry-matched candidates. This makes the set anatomically false but often geometrically easier.

### Decision

For the first clean JUNCTION+CT baseline:

1. do not use `segment_label` to rank/select hard negatives;
2. select hard negatives primarily by `geometry_match_distance`;
3. keep decoy/replaced labels only as audit metadata;
4. after selection, report same-label vs different-label composition.

Recommended first selection key:

```python
(geometry_match_distance, candidate_id)
```

If later a stratified same-label/different-label study is needed, make it a separate ablation rather than silently changing the canonical generator.

## Adaptive cut

Current rule:

```text
cut = min(2 mm, 0.9 * shortest arm)
```

Twelve train/val junctions get cut <2 mm. Inspection shows these are six close junction pairs connected by very short inter-junction arms. With factor 0.9, the exposed endpoint can be placed very close to the neighboring junction.

This is undesirable for the controlled hidden-junction scene because the local target repair can become contaminated by the next branch point.

A train/val-only audit of max-arm fractions 0.4, 0.45, 0.5, 0.6, 0.75 and 0.9 kept all 144 positives span-valid under the historical 16 mm limit.

### Decision

Do not freeze 0.9.

Use:

```text
cut = min(2.0 mm, 0.45 * shortest incident arm length)
```

for the first protocol, and add an explicit `close_junction` / `adaptive_cut` audit flag whenever cut < 2 mm.

Reason for 0.45: two neighboring junction neighborhoods on the same short arm cannot overlap/cross the midpoint. This is a safety convention for the new protocol, not a recovered historical rule.

Before held-out test generation, inspect the 24 train/val adaptive-cut cases visually/numerically.

## Decoy step and endpoint margin

Current:

- decoy step = 1.5 mm
- endpoint margin = 2.0 mm

These are acceptable as the first frozen generator discretization.

Do not tune them further against model performance. Their role is candidate-pool sampling, not a learned hyperparameter.

Keep them fixed for the first baseline unless a structural audit shows they systematically miss plausible competitors.

## Negative types and caps

For the first local JUNCTION candidate model keep:

- one-arm replacement;
- two-arm replacement.

Keep the current caps provisionally:

- <=12 one-arm negatives / positive;
- <=6 two-arm negatives / positive.

The resulting 7.9% positive rate is reasonably close to the historical naturally imbalanced pool (~5.3%) and is preferable to forcing 1:1.

However, candidate availability is uneven:

- 47/143 degree-3 junctions currently have zero selected negatives at all;
- 90/143 have the full 18 negatives;
- the remainder have partial pools.

Therefore every ranking/evaluation report must separate:

- competitive junctions: at least one negative candidate;
- isolated junctions: no valid competing negative under the frozen generator.

Do not count isolated scenes as evidence that the ranker solved a hard case.

## Additional negative types

### none_incomplete_junction

Yes, but NOT inside the first local binary JUNCTION candidate ranker.

It belongs to the later scene-level relation task:

`NONE / PAIR / JUNCTION / BOTH`

and should be generated as a separate scene type after the local JUNCTION geometry/CT baseline is frozen.

### all-arm replacement

Do not add to the first baseline. It is likely to create easier negatives and is not required to test whether CT distinguishes a true junction from a close geometry-matched competitor.

Can be added later as a robustness ablation.

### extra-arm negatives

Do not add to the first degree-3 candidate ranker.

They are more naturally a scene/relation-type error mode (invalid extra branch / BOTH-like ambiguity) and should be tested later with the four-class relation head.

## Perturbations

Do NOT mix 30 deg / 45 deg + 1 mm perturbations into the clean candidate generator or training baseline now.

First freeze and train on the clean controlled plan.

Then apply:

- position jitter;
- tangent noise;
- 30 deg + 1 mm;
- 45 deg + 1 mm

as robustness/stress-test copies after the generator and model are frozen.

If perturbation augmentation is later tested for training, treat it as a separate experiment selected only on train/validation.

## Degree-4

Keep all degree-4 junctions in source metadata and audit outputs, but do not include them in the primary learned/evaluated baseline.

CT28 contains only:

- 1 degree-4 junction in train;
- 0 in validation;
- 2 in held-out test.

This is insufficient to tune or validate a degree-4 model fairly.

Primary quantitative experiment: degree-3 only.

After the degree-3 protocol/model is frozen, degree-4 may be shown as a descriptive zero-shot audit. Do not include the two test degree-4 cases in the headline metric.

A larger cohort is needed for a real degree-4 study.

## Immediate implementation checklist

1. Change adaptive cap from 0.9 to 0.45 and record `adaptive_cut` flag.
2. Change hard-negative selection to geometry-first; labels audit-only.
3. Keep 1.5 mm decoy step, 2 mm endpoint margin, max span 16 mm.
4. Keep one-arm + two-arm only; caps 12 + 6.
5. Regenerate train/val only.
6. Re-run geometry sanity check.
7. Inspect competitive-junction count and negative hardness.
8. Freeze protocol only after this review.
9. Then generate JUNCTION CT features on train/val.
10. Only after model/threshold choices are frozen, generate held-out test plan once.

## Next model comparison

On the frozen degree-3 plan:

```text
JUNCTION geometry
vs
JUNCTION CT
vs
JUNCTION geometry + CT
```

Then, only after patient-general JUNCTION_CT exists:

```text
PAIR geometry + PAIR CT
+
JUNCTION geometry + JUNCTION CT
        ↓
NONE / PAIR / JUNCTION / BOTH
        ↓
canonical Graph-LIRA
        ↓
selective repair / abstain
```
