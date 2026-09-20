# Binary-lumen shape context as a relation-existence proxy

Date: 2026-09-20

## Why this experiment exists

The 800-case ImageCAS-X geometry benchmark showed that candidate recall is already high, while the main failure is deciding whether a valid repair relation exists at all.

No matched original CCTA volumes are present in the uploaded ImageCAS-X bundle, so CT-conditioned learning is still blocked. As an intermediate experiment, this study asks a narrower question:

> Does spatial context from the already-available broken binary vessel mask improve PAIR / JUNCTION existence decisions beyond geometry-score distributions alone?

This is **not CT evidence** and must not be presented as image-conditioned clinical validation.

## Data and split

Same frozen ImageCAS-X benchmark as the large-scale geometry study:

- 800 patients;
- official split: 560 train / 80 validation / 160 test;
- 10,615 controlled scenes;
- primary stress: 30 deg tangent perturbation + 1 mm endpoint jitter;
- secondary stress: 45 deg + 1 mm.

The binary ImageCAS-X vessel mask is sampled after virtual deletion of every synthetic gap / junction source represented by the scene. Anatomical segment labels are not used as model features.

## Features

Two binary existence heads are trained independently:

- pair exists?;
- junction exists?

Inputs:

1. geometry-only OOF score-distribution features from the frozen local pair/junction rankers;
2. binary-lumen mask-shape features sampled around endpoints, candidate corridors and fitted junction regions;
3. their concatenation.

The train geometry features are out-of-fold with respect to the official 560-patient train split.

## Relation-existence ranking

### Pair presence

| Variant | test30 AUROC | test30 AUPRC | test45 AUROC | test45 AUPRC |
|---|---:|---:|---:|---:|
| geometry | 0.85434 | 0.66038 | 0.81664 | 0.61042 |
| mask only | 0.90506 | 0.78468 | 0.88656 | 0.76371 |
| geometry + mask | **0.92877** | **0.82532** | **0.90014** | **0.78370** |

### Junction presence

| Variant | test30 AUROC | test30 AUPRC | test45 AUROC | test45 AUPRC |
|---|---:|---:|---:|---:|
| geometry | 0.91004 | 0.86281 | 0.86615 | 0.80155 |
| mask only | 0.90001 | 0.85691 | 0.87729 | 0.82746 |
| geometry + mask | **0.93144** | **0.89489** | **0.89455** | **0.84966** |

This is the strongest direct evidence so far that nonlocal spatial vessel context helps the **existence** stage, which was identified as the main geometry-only bottleneck.

## Frozen structural operating-point selection

Pair and junction thresholds are selected using **only the official validation patients** and both validation stress settings.

For the safety-oriented operating point, require structural false-scene rate <= 5% on both val30 and val45, then maximize the worst validation exact rate.

Selected thresholds:

- geometry-only: pair 0.97, junction 0.875;
- geometry + mask: pair 0.96, junction 0.90.

### Test results

| Variant | test30 exact | test30 false | test30 incomplete | test45 exact | test45 false | test45 incomplete |
|---|---:|---:|---:|---:|---:|---:|
| geometry | 51.46% | 7.67% | 40.87% | 47.46% | 6.45% | 46.09% |
| geometry + mask | **56.50%** | **6.50%** | 37.01% | **51.60%** | **5.98%** | 42.42% |

Thus, under the same validation-side safety budget, mask shape context increases exact structural resolution by about 5 points while also lowering the observed test false rate relative to the matched geometry-presence baseline.

## Comparison with the canonical four-class geometry relation head

The canonical relation-type system (NONE / PAIR / JUNCTION / BOTH) previously achieved:

- test30 exact 55.46%, false 10.92%;
- test45 exact 51.37%, false 8.05%.

At the 5%-validation-budget geometry+mask presence operating point:

- test30 exact 56.50%, false 6.50%;
- test45 exact 51.60%, false 5.98%.

Paired patient-cluster bootstrap, geometry+mask presence minus canonical relation head:

| Eval | Metric | Difference | 95% cluster CI |
|---|---|---:|---:|
| test30 | exact | +1.04 pp | [-0.52, +2.58] |
| test30 | false | **-4.43 pp** | **[-5.68, -3.20]** |
| test30 | incomplete | +3.39 pp | [+1.88, +4.93] |
| test45 | exact | +0.24 pp | [-1.03, +1.47] |
| test45 | false | **-2.07 pp** | **[-3.12, -1.03]** |
| test45 | incomplete | +1.84 pp | [+0.47, +3.24] |

Interpretation: the mask-presence formulation does not establish a statistically clear exact-rate improvement over the canonical relation head, but it substantially reduces false structural repairs at the cost of more incomplete / abstaining decisions.

That tradeoff is aligned with the safe-repair objective.

## Important failure analysis

The gain is not uniform.

At test30:

- mixed scenes improve strongly in exact rate, but false mixed decisions also increase;
- NO-REPAIR scenes become safer;
- pair-only repair recall becomes very conservative;
- degree-4 junctions remain a severe failure mode.

For the 30 degree-4 test scenes, the 5%-budget mask-presence system has:

- exact 0%;
- false 63.33%;
- incomplete 36.67%.

Therefore **binary mask shape context is not a solution to high-degree branching**.

The mask features also help RCA much more than LAD/LCX, so the residual anatomical ambiguity remains exactly where the earlier geometry audit predicted it would.

## Scientific interpretation

This experiment changes the next-step hypothesis in a useful way.

The positive result is not "the mask solves Graph-LIRA." It does not.

The supported conclusion is:

> Relation existence benefits from spatial vessel context that is not captured by local geometric score distributions alone.

That strengthens the motivation for the next matched-CT experiment:

```text
geometry scores
+ local / cross-section CT representation
-> pair-presence and junction-presence evidence
-> joint structural Graph-LIRA
-> perturbation-consistency risk control
-> repair / abstain
```

The image model should therefore target **existence and branch identity**, not merely replace the local ranker.

## Limitations

- The mask is derived from the clean ImageCAS-X anatomical segmentation and then synthetically broken. It is cleaner than a real predicted segmentation.
- No CCTA intensities are used.
- Natural segmentation failures are not represented.
- The test set has already been inspected; no further publication-facing threshold tuning should be performed on it.
- Degree-4 behavior is poor and must be treated as a frozen hard stratum.
- A matched original ImageCAS CT cohort is still required before making image-conditioned claims.

## Machine artifacts

Canonical artifacts for this experiment are stored under:

`results/ccta_graph_lira_safe_repair/2026-09-20/mask_shape_context_*.csv`

and scripts under:

`scripts/research/ccta_graph_lira_safe_repair/mask_shape_context_*.py`.
