# Graph-LIRA: 800-case ImageCAS-X structural benchmark checkpoint

Date: 2026-09-20

This checkpoint supersedes the two-patient benchmark as the primary **geometry-only structural stress test**. The older 921/953 experiments remain useful controlled evidence, but they were substantially easier and must not be presented as representative of the full dataset.

## 1. Data actually used

The uploaded research bundle contains 800 ImageCAS-X anatomical cases:

- 1600 centerline VTK files: left + right for every case;
- 800 multi-label coronary segmentations;
- official split lists: 560 train / 80 validation / 160 test;
- `Descriptors.xlsx` metadata.

Raw medical files must not be committed to Git. The bundle SHA256 is:

`a0da8d1b23861217f1681d7b911bf7fbe9763066b4d7f576ea33e6741f8781c4`

A canonical SHA256 manifest with 2405 entries was generated locally.

No original CCTA volumes are present in this bundle. Therefore this checkpoint evaluates **anatomical / geometric structural reasoning**, not CT image evidence.

## 2. Dataset anatomy audit

Across 800 patients:

- 4,686 annotated branch points;
- 4,641 degree-3 junctions;
- 45 degree-4 junctions in 44 patients;
- median 14 centerline polylines per case;
- median 1,230.5 centerline points per case.

Degree-4 junctions are distributed across the official split: 28 train, 1 validation, 16 test. Their anatomical labels are mainly LM (35), with LAD (4), LCX (4), and RCA (2).

Segment presence is broad: LAD / LCX / RCA occur in all 800 cases; LM in 789; D1 in 788; R-PDA in 744; R-PLA in 736; OM1 in 694; D2 in 475; OM2 in 301; IM in 196; Other in 88; L-PLA in 70; L-PDA in 46.

One annotation anomaly was isolated: scan 272 contains a small `L14` segment with label 0. It must be treated as unknown / excluded from anatomical segment analyses rather than silently mapped to a named branch.

A spot geometry audit on scans 921, 953, 956, 957, and 960 found 100% centerline points within the vessel mask and 100% exact centerline-to-segment-label agreement after the established coordinate conversion.

## 3. Frozen controlled scene protocol

A total of 10,615 synthetic-but-anatomically-grounded scenes were generated from real annotated centerlines:

| split | scenes |
|---|---:|
| train | 7,463 |
| validation | 1,028 |
| test | 2,124 |

Scene types are deliberately heterogeneous:

- `pair_only`: one true 3–7 mm gap;
- `junction_only`: a lost real bifurcation / junction;
- `mixed`: a true pair and a true junction in the same scene;
- `none_orphan_pair`: an endpoint whose true partner is withheld;
- `none_incomplete_junction`: an incomplete junction candidate set where no valid reconstruction should be forced.

Thus the system must solve not only pair ranking, but also **PAIR / JUNCTION / NO-REPAIR** existence.

Candidate generation is intentionally simple and label-free:

- pair distance <= 12 mm;
- junction maximum endpoint span <= 16 mm.

Under test stress at 30 deg tangent noise + 1 mm endpoint jitter and 45 deg + 1 mm, true pair and true junction candidate recall remained 100%. Candidate generation is therefore not the current bottleneck under this protocol.

## 4. Local geometry ranking is useful but not sufficient

At test 30 deg + 1 mm:

- pair local AUROC: 0.9064;
- pair AUPRC: 0.1916;
- true pair top-1: 69.02%; top-2: 94.02%; top-3: 99.15%; top-5: 99.57%;
- junction local AUROC: 0.9446;
- junction AUPRC: 0.4441;
- true junction top-1: 75.45%; top-2: 88.50%; top-3: 92.97%; top-5: 97.32%.

This is a key diagnosis: the correct hypothesis is usually present near the top of the candidate list, but absolute geometry-only scores are poorly calibrated for deciding whether a valid repair exists at all.

## 5. Conservative geometry-only operating point

Validation selected a conservative pair / junction threshold pair. On the 160-patient test set:

### 30 deg + 1 mm

- joint structural exact: 52.02%;
- false-scene rate: 12.34%;
- incomplete-but-nonfalse: 35.64%.

Patient-cluster bootstrap 95% intervals:

- exact: 50.54–53.49%;
- false: 10.92–13.84%;
- incomplete: 34.19–37.04%.

### 45 deg + 1 mm

- exact: 47.46% (95% cluster CI 46.11–48.85%);
- false: 9.89% (8.57–11.25%);
- incomplete: 42.66% (41.32–43.96%).

The lower false rate at the harder stress level does **not** mean the problem became easier. The system becomes more conservative and leaves more scenes unresolved.

The overall exact number must not be interpreted as repair recall. At 30 deg + 1 mm, the conservative operating point is heavily driven by correct NO-REPAIR decisions:

- `pair_only`: exact 0%, false 6.88%, incomplete 93.13%;
- `junction_only`: exact 42.38%, false 13.77%;
- `mixed`: exact 0%, false 11.49%, incomplete 88.51%;
- `none_incomplete_junction`: exact 84.63%;
- `none_orphan_pair`: exact 96.88%.

This exposes the central geometry-only tradeoff: lowering thresholds recovers repairs but rapidly creates false bridges in NO-REPAIR scenes.

## 6. Explicit relation-type head: PAIR / JUNCTION / BOTH / NONE

A new scene-level relation-type head was trained from **out-of-fold local-score distributions on the official 560-patient train split**. It never uses the held-out test labels to construct its training features.

A HistGradientBoosting relation head with a validation-selected confidence threshold 0.85 was chosen. It predicts one of NONE, PAIR, JUNCTION, BOTH.

### Test 30 deg + 1 mm

- relation-type accuracy: 61.49%;
- structural exact: **55.46%**;
- false scene: **10.92%**;
- incomplete: 33.62%.

Compared with the conservative base joint optimizer:

- exact: +3.44 percentage points;
- false: -1.41 points;
- incomplete: -2.02 points.

Paired patient-cluster bootstrap 95% intervals for the differences:

- exact gain: **+1.76 to +5.17 points**;
- false-rate change: **-2.79 to -0.05 points**;
- incomplete change: **-3.82 to -0.24 points**.

### Test 45 deg + 1 mm

- structural exact: **51.37%**;
- false scene: **8.05%**;
- incomplete: 40.58%.

Paired changes versus the base system:

- exact +3.91 points, 95% cluster CI +2.43 to +5.45;
- false -1.84 points, CI -3.10 to -0.60;
- incomplete -2.07 points, CI -3.58 to -0.61.

This is a real incremental improvement, but it does not solve the task. At 30 deg, exact by scene type is still only:

- pair-only 14.69%;
- junction-only 31.55%;
- mixed 42.57%;
- NO-REPAIR incomplete-junction 90.78%;
- NO-REPAIR orphan-pair 95.63%.

The remaining bottleneck is therefore not candidate recall; it is reliable relation existence / identity under ambiguity.

## 7. Perturbation consistency still works as a selective safety layer

For the relation-type system, 15 additional perturbation reruns use +0.8 mm endpoint noise and +20 deg tangent noise.

On validation, the maximum-coverage threshold with <=1% false among accepted is combined consistency >= 0.60:

- validation coverage 76.36%;
- validation false among accepted 0.892%.

Frozen on test:

### Test 30 deg + 1 mm, gate >= 0.60

- coverage: **74.34%** (1,579 / 2,124 scenes);
- false among accepted: **0.823%** (13 scenes);
- exact among accepted: 57.88%;
- incomplete among accepted: 41.29%.

Patient-cluster bootstrap 95% intervals:

- coverage 72.49–76.20%;
- false 0.432–1.262%;
- exact 56.37–59.38%.

At stricter test risk-coverage points (reported descriptively, not re-selected on test):

- consistency >= 0.85: coverage 57.91%, false 0.081%, exact 63.82%;
- >= 0.90: coverage 51.13%, false 0.092%, exact 67.31%;
- >= 0.95: coverage 38.37%, 0 observed false, exact 71.53%.

### Test 45 deg + 1 mm, frozen gate >= 0.60

- coverage: **81.87%** (1,739 / 2,124);
- false among accepted: **0.863%**;
- exact among accepted: 51.93%;
- incomplete: 47.21%.

Again, higher coverage under harder perturbation is not evidence of better repair; many predictions are stably conservative / incomplete.

Perturbation consistency is therefore a useful **risk-control signal**, not a correctness oracle.

## 8. Explicit binary “repair exists?” heads

Cross-fitted score-distribution models were also tested separately for pair-presence and junction-presence.

Validation-selected HGB heads achieved on test30:

- pair presence AUROC 0.8238, AUPRC 0.6132;
- junction presence AUROC 0.8770, AUPRC 0.7908.

But the full structural system remained around 50.0% exact and 12.38% false. The presence heads contain useful signal but are not sufficient with geometry-only inputs.

## 9. Multi-event graph optimization

A separate 480-scene test contains several simultaneous possible repairs (`multi_pair`, `multi_mixed`, `multi_none`).

At the validation-selected very conservative 0.99 / 0.99 thresholds, independent, greedy, and global methods collapse to the same behavior because almost all uncertain repairs are rejected.

At an explicitly exploratory high-recall 0.8 / 0.8 operating point, global graph matching shows the structural benefit more clearly.

### Test30 high-recall

- independent: exact 28.13%, false 40.00%;
- greedy: exact 31.67%, false 31.67%;
- global: exact **33.13%**, false **27.71%**.

Global vs greedy:

- exact +1.46 points;
- false -3.96 points.

### Test45 high-recall

- greedy: exact 27.29%, false 29.17%;
- global: exact 27.50%, false 26.04%.

Global reasoning reduces mutually inconsistent false structure when several repairs compete, but these high-recall false rates are too large for a safe operating point. This remains exploratory evidence for the graph layer, not a clinical result.

## 10. What the 800-case benchmark changes

The large-scale data force a more precise research claim.

The useful result is **not** that geometry-only Graph-LIRA already solves coronary repair. It does not.

The current evidence supports four narrower conclusions:

1. proximity-based candidate generation has essentially complete recall under the controlled stress protocol;
2. local geometry ranks the true pair / junction surprisingly well;
3. global structural reasoning reduces conflicts when multiple repair hypotheses compete;
4. selective perturbation consistency can sharply reduce false automatic repairs, but many accepted / rejected cases remain incomplete.

The major unresolved problem is **relation existence and branch identity**: deciding whether the candidate set actually contains a valid pair / junction instead of a plausible but wrong connection.

This is exactly where CT image context should be introduced.

## 11. Next experiment — frozen before image-model development

Do not tune geometry further on the inspected ImageCAS-X test set.

The next image-conditioned experiment should keep candidate generation, patient splits, graph optimizer, and uncertainty protocol fixed, then compare local evidence:

```text
geometry only
vs radial 2.5-D CT context
vs candidate-aligned 3-D CT tube
vs full cross-section CNN encoder -> sequence context
vs ANZA local encoder -> sequence context
```

The target is not merely higher pair AUROC. The target is:

- better PAIR / JUNCTION / NO-REPAIR discrimination;
- higher repair-scene exact rate at a fixed false-repair budget;
- improved hard high-degree junction performance;
- better risk-coverage after the same perturbation-consistency gate.

A matched original ImageCAS CCTA volume for an ImageCAS-X anatomical scan is still required for this question. The uploaded 27 MB bundle intentionally contains no CT volumes.

## 12. Scientific limitations frozen now

- gaps are controlled synthetic deletions from annotated centerlines, not natural segmentation failures;
- endpoints / tangents are generated from annotated geometry plus explicit perturbations, not yet extracted from a real predicted segmentation pipeline;
- the 800-case test set has now been inspected repeatedly and should not be used for another round of publication-facing hyperparameter selection;
- patient-cluster bootstrap is used for uncertainty because many scenes come from each patient;
- `0 observed false` at a selective operating point is not proof of zero population risk;
- image-conditioned claims remain blocked until matched CT + ImageCAS-X anatomy is available.

## 13. Repair-aware correction: selective coverage is dominated by NO-REPAIR

A later audit separated **accepted-scene correctness** from **actual successful repair**. This changes how the selective results must be interpreted.

At the strict relation-type operating point (confidence/consistency gate `0.70`) on test30:

- `pair_only`: coverage `66.25%`, but exact among accepted `0%`; all accepted cases are incomplete;
- `junction_only`: coverage `43.58%`, exact among accepted only `2.45%`, incomplete `96.63%`;
- `mixed`: coverage `45.27%`, exact among accepted `29.85%`;
- `none_incomplete_junction`: coverage `88.90%`, exact among accepted `100%`;
- `none_orphan_pair`: coverage `92.50%`, exact among accepted `100%`.

The same pattern persists at test45. Therefore high selective coverage / low false rate is currently driven mostly by **stable correct NO-REPAIR decisions**, not by successful automatic repair.

This is an important scientific correction: perturbation consistency is a strong safety signal, but geometry-only selective Graph-LIRA is **not yet a high-recall repair system**.

Machine artifact:

- `results/ccta_graph_lira_safe_repair/2026-09-20/large_scale_repair_aware_diagnostics.csv`.

## 14. Bottleneck decomposition: relation existence is the main remaining problem

To separate candidate ranking from existence/identity, an oracle was evaluated that knows whether the scene truly contains PAIR / JUNCTION / BOTH / NONE, while the existing local geometry ranker still chooses the candidate.

At test30, among the `1,216` repair-needed scenes:

- actual relation-type system exact: `28.45%`;
- oracle presence + top-1 geometry candidate: `70.81%`;
- oracle presence + top-2: `89.47%`;
- oracle presence + top-3: `94.57%`;
- oracle presence + top-5: `97.94%`.

At test45:

- actual: `18.50%`;
- oracle + top-1: `62.75%`;
- oracle + top-2: `82.40%`;
- oracle + top-3: `89.97%`;
- oracle + top-5: `95.81%`.

Thus candidate generation/ranking is comparatively strong. The dominant error source is deciding **whether a valid relation exists and what structural action is present**, especially under ambiguity.

This freezes the next scientific target:

> stop tuning geometry thresholds; add local spatial/image evidence specifically to PAIR / JUNCTION / NO-REPAIR existence and branch identity, while keeping the candidate generator, graph optimizer and patient split fixed.

The next intermediate experiment may use a **broken binary-lumen mask as shape context** (with the deleted relation removed before feature extraction) to test whether spatial context helps existence discrimination without using anatomical segment labels as model input. This is a proxy experiment, not CT evidence. The final image-conditioned experiment still requires the matched original ImageCAS CCTA volumes.

The paired strict selective comparison also remains useful but must not be called repair recall: relation-type gating increases overall exact yield by about `+7.96` points on test30 and `+7.77` on test45 at a similar false-yield level, but that gain includes many correct NO-REPAIR scenes.

Machine artifact:

- `results/ccta_graph_lira_safe_repair/2026-09-20/large_scale_selective_paired_bootstrap.csv`.
