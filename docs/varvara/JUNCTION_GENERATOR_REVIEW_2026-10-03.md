# HISTORICAL REVIEW — superseded

This review records the 2026-10-03 generator-development decision point. Later train/validation work changed the decision: the controlled generator is now frozen and generator tuning is no longer the active task. For current instructions use `docs/varvara/CURRENT_TASK.md`.

# CT28 JUNCTION generator review and next decisions — 2026-10-03

This note reviews Varvara's current `build_junction_plan.py` using only train/validation data. Held-out test junction candidates have not been generated or inspected.

## 1. Current draft reproduces its stated counts

With:

- target cut = 2.0 mm
- max arm fraction = 0.9
- max span = 16 mm
- decoy step = 1.5 mm
- endpoint margin = 2.0 mm
- max one-arm negatives = 12 / positive
- max two-arm negatives = 6 / positive

the uploaded generator reproduces:

- 144 train/val true junctions;
- 143 degree-3 + 1 degree-4;
- 143 degree-3 positives in the selected plan;
- 1,666 selected negatives;
- 1,094 one-arm negatives;
- 572 two-arm negatives;
- 7.9% positives;
- 20 selected negatives using the same-label fallback;
- raw generated negative pool = 143,116.

The code keeps test out of the development plan.

## 2. The current negative family is too easy for geometry

I ran a train-only geometry sanity check on the nine existing `jg_*` features and evaluated on validation.

This is a generator diagnostic, not a paper result.

Current selection:

- LogisticRegression: validation AUROC 0.9951; top-1 positive rank 96.4% on competitive validation junctions.
- HistGradientBoosting: validation AUROC 0.9993; top-1 positive rank 100%.

This means the current one-/two-arm replacement pool, after selection, is almost perfectly separable by geometry alone. That is a problem for the intended JUNCTION+CT experiment: CT would have almost no meaningful ambiguity left to resolve.

Important correction: simply removing the different-label preference does **not** solve this. I repeated the train/val audit using geometry-distance-first selection; HGB remained essentially perfect (AUROC ~0.99994, top-1 100%). Therefore the issue is the negative family itself, not only the tie-breaking key.

### Decision

Do not freeze the negative protocol yet.

Keep the present generator as a useful baseline/audit generator, but add a new **geometry-adversarial hard-negative mining stage** on train/val:

1. generate the full span-valid negative pool;
2. train a geometry-only ranker on train only;
3. for train hard-negative mining use out-of-fold geometry scores;
4. for validation use the frozen train geometry ranker;
5. retain the highest-scoring false junction candidates per true junction;
6. keep labels only as truth/audit metadata, not as inference features;
7. report how many junctions actually have a competitive false candidate.

This directly asks the intended question: can CT distinguish true junctions from candidates that geometry itself finds plausible?

The old historical 16 mm span remains unchanged.

## 3. Adaptive cut

Current rule:

```text
cut = min(2.0 mm, 0.9 * shortest incident arm)
```

Twelve train/val junctions receive a cut <2 mm. These are associated with short inter-junction arms, so the exposed endpoint can approach a neighbouring branch point.

The 0.9 factor is a new choice, not a recovered historical rule.

### Recommendation

Do not freeze 0.9 yet.

Use the more conservative development rule:

```text
cut = min(2.0 mm, 0.45 * shortest incident arm)
```

and add an `adaptive_cut` / `close_junction` audit flag.

The 0.45 rule is deliberately below the midpoint of a short shared arm, so two neighbouring hidden-junction regions cannot cross each other. It is a new safety convention and must be documented as such.

On train/val this gives 24 adaptive-cut junctions. Inspect those 24 before freezing.

## 4. Decoy step and endpoint margin

Keep for the first protocol:

- decoy step = 1.5 mm;
- endpoint margin = 2.0 mm.

There is currently no evidence that changing these improves the scientific validity of the candidate pool. Treat them as generator discretization parameters, not model hyperparameters.

## 5. One-arm / two-arm negatives and caps

Keep one-arm and two-arm replacements as the basic local JUNCTION negative families.

Keep the current caps provisionally:

- <=12 one-arm / positive;
- <=6 two-arm / positive.

The 7.9% positive rate is close enough to the historical naturally imbalanced pool (~5.3%) that forcing 1:1 is not justified.

However, the current candidate availability is uneven:

- 47/143 degree-3 positives have zero selected negatives;
- 90/143 have the full 18;
- the rest have partial pools.

Therefore all ranking metrics must distinguish:

- **competitive junctions** — at least one valid false candidate exists;
- **isolated junctions** — no false candidate exists under the frozen generator.

Do not count isolated scenes as evidence that a ranker solved an ambiguous case.

## 6. Additional negative types

### none_incomplete_junction

Yes, but later.

It belongs naturally to the scene-level relation problem:

`NONE / PAIR / JUNCTION / BOTH`

rather than the first local exact-junction candidate ranker.

### all-arm replacement

Do not add to the first baseline. It is likely to create mostly easy negatives and does not target the remaining ambiguity.

Use only as a later robustness ablation if needed.

### extra-arm negatives

Also later, in the scene/relation-type stage. They model an invalid extra branch and are closer to relation-existence/type ambiguity than to the first local degree-3 candidate ranker.

## 7. Perturbations

Do not mix 30 deg / 45 deg + 1 mm stress into the clean baseline generator now.

Order:

1. freeze clean train/val generator;
2. train/freeze geometry, CT and geometry+CT models;
3. then evaluate position jitter, tangent noise, 30 deg + 1 mm and 45 deg + 1 mm as robustness/stress tests.

Training augmentation with perturbations can be a separate later experiment, selected only on train/validation.

## 8. Degree-4

Keep degree-4 in metadata and audit outputs, but exclude it from the primary learned quantitative experiment.

CT28 contains only:

- 1 degree-4 in train;
- 0 in validation;
- 2 in held-out test.

That is not enough to tune or validate a degree-4 model without using test as development data.

Primary model: degree-3 only.

After the degree-3 protocol is frozen, degree-4 can be shown as a descriptive zero-shot audit, not mixed into the headline metric.

## 9. Immediate implementation checklist

1. Keep held-out test untouched.
2. Change adaptive cut to the conservative 0.45 rule and add audit flags.
3. Keep max span 16 mm, decoy step 1.5 mm, endpoint margin 2 mm.
4. Keep one-arm + two-arm generation and 12 + 6 caps as the basic pool.
5. Add geometry-adversarial hard-negative mining using train-only/OOF geometry scores.
6. Regenerate train/val.
7. Re-run geometry diagnostic.
8. Freeze the generator only when the retained negatives are genuinely competitive and the protocol is documented.
9. Then compute JUNCTION CT features on train/val.
10. Compare JUNCTION geometry vs CT vs geometry+CT.
11. Only after all choices are frozen, generate held-out test once.

## 10. Next scientific step

```text
JUNCTION geometry
vs
JUNCTION CT
vs
JUNCTION geometry + CT
```

Then:

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
