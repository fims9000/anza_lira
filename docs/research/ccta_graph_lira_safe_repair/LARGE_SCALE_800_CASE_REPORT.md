# Graph-LIRA: 800-case ImageCAS-X structural benchmark

Date: 2026-09-20

This is the primary **geometry-only structural benchmark** for the CCTA Graph-LIRA branch. The earlier 921/953 experiments remain useful controlled evidence but were easier and must not be presented as representative of the full dataset.

## Data

The research bundle contains:

- 800 ImageCAS-X anatomical cases;
- 1600 centerline VTKs, left + right for every case;
- 800 multi-label coronary segmentations;
- official patient split: 560 train / 80 validation / 160 test;
- Descriptors.xlsx.

Bundle SHA256:

`a0da8d1b23861217f1681d7b911bf7fbe9763066b4d7f576ea33e6741f8781c4`

No original CCTA intensities are present, so this benchmark tests anatomical / geometric structural reasoning rather than CT evidence.

## Anatomy audit

Across 800 patients:

- 4,686 annotated branch points;
- 4,641 degree-3 junctions;
- 45 degree-4 junctions in 44 patients;
- median 14 centerline polylines per case;
- median 1,230.5 centerline points per case.

Degree-4 junctions: 28 train, 1 validation, 16 test. Their labels are mostly LM (35), then LAD (4), LCX (4), RCA (2).

LAD / LCX / RCA occur in all 800 cases. Scan 272 contains a small `L14` / label-0 anomaly and is treated as unknown for anatomical segment analyses.

Spot audits on 921, 953, 956, 957 and 960 gave 100% centerline-inside-mask and exact segment-label agreement after the established coordinate conversion.

## Frozen scene protocol

10,615 controlled scenes were generated:

| split | scenes |
|---|---:|
| train | 7,463 |
| validation | 1,028 |
| test | 2,124 |

Scene types:

- pair_only;
- junction_only;
- mixed;
- none_orphan_pair;
- none_incomplete_junction.

The task is therefore explicitly **PAIR / JUNCTION / NO-REPAIR**, not just pair ranking.

Candidate generation is simple and label-free:

- pair distance <= 12 mm;
- junction maximum endpoint span <= 16 mm.

At both test30 (30 deg tangent noise + 1 mm endpoint jitter) and test45 (45 deg + 1 mm), true pair and true junction candidate recall remained 100%.

## Local ranking diagnosis

At test30:

- pair AUROC 0.9064;
- pair AUPRC 0.1916;
- pair top-1 69.02%, top-2 94.02%, top-3 99.15%, top-5 99.57%;
- junction AUROC 0.9446;
- junction AUPRC 0.4441;
- junction top-1 75.45%, top-2 88.50%, top-3 92.97%, top-5 97.32%.

The correct hypothesis is usually present near the top. The main geometry-only bottleneck is therefore relation **existence / identity / calibration**, not candidate recall.

## Conservative geometry system

Test30:

- exact 52.02%;
- false-scene 12.34%;
- incomplete 35.64%.

Test45:

- exact 47.46%;
- false 9.89%;
- incomplete 42.66%.

The lower false rate at harder stress reflects increased conservatism / incompleteness, not an easier task.

At test30 the conservative system is particularly weak on actual repair scenes:

- pair_only exact 0%;
- junction_only exact 42.38%;
- mixed exact 0%;
- none_incomplete_junction exact 84.63%;
- none_orphan_pair exact 96.88%.

## Explicit relation-type head

A scene-level HGB head, trained from out-of-fold train-score distributions, predicts:

- NONE;
- PAIR;
- JUNCTION;
- BOTH.

Validation-selected confidence threshold: 0.85.

Test30:

- relation-type accuracy 61.49%;
- structural exact 55.46%;
- false 10.92%;
- incomplete 33.62%.

Paired patient-cluster bootstrap versus the conservative base:

- exact +3.44 pp, 95% CI [+1.76,+5.17];
- false -1.41 pp, CI [-2.79,-0.05];
- incomplete -2.02 pp, CI [-3.82,-0.24].

Test45:

- exact 51.37%;
- false 8.05%;
- incomplete 40.58%.

The relation head is an incremental improvement, but the repair problem remains unsolved.

## Selective perturbation consistency

15 additional perturbation reruns use +0.8 mm endpoint noise and +20 deg tangent noise.

Validation selects combined consistency >= 0.60 under <=1% false among accepted.

Frozen test30:

- coverage 74.34% = 1,579 / 2,124;
- false among accepted 0.823% = 13;
- exact among accepted 57.88%;
- incomplete 41.29%.

Frozen test45:

- coverage 81.87%;
- false among accepted 0.863%;
- exact among accepted 51.93%;
- incomplete 47.21%.

This is risk control, not proof of high automatic repair recall. Many accepted scenes are stably conservative / incomplete.

## Repair-aware correction

At stricter relation-type consistency gate 0.70 on test30:

- pair_only coverage 66.25%, exact among accepted 0%;
- junction_only coverage 43.58%, exact 2.45%;
- mixed coverage 45.27%, exact 29.85%.

Accepted NO-REPAIR scenes are almost entirely correct.

An oracle that tells the system whether a repair relation exists shows the remaining opportunity. On repair-needed test30 scenes:

- current relation system exact 28.45%;
- oracle relation presence + existing geometry top-1 exact 70.81%;
- oracle presence + top-3 94.57%.

At test45:

- current 18.50%;
- oracle top-1 62.75%;
- oracle top-3 89.97%.

This is the decisive diagnosis: **PAIR / JUNCTION / NO-REPAIR existence and branch identity are the main bottleneck.**

## Multi-event graph optimization

In a 480-scene exploratory multi-event benchmark, high-recall 0.8/0.8 operation shows a structural benefit:

Test30:

- independent exact 28.13%, false 40.00%;
- greedy exact 31.67%, false 31.67%;
- global exact 33.13%, false 27.71%.

Test45:

- greedy exact 27.29%, false 29.17%;
- global exact 27.50%, false 26.04%.

Global reasoning reduces mutually inconsistent false structure but is not safe enough by itself.

## Consequence for the research direction

Do not spend another iteration tuning geometry thresholds on the inspected test set.

Freeze:

- official patient splits;
- candidate generator;
- graph optimizer;
- uncertainty protocol;
- anatomy-based hard strata.

The next local-evidence model should target **relation existence and branch identity**, while Graph-LIRA remains the structural decision layer.

The intended matched-CT comparison remains:

```text
geometry only
vs radial 2.5-D CT context
vs candidate-aligned 3-D CT tube
vs full cross-section CNN encoder -> sequence context
vs ANZA local encoder -> sequence context
```

Primary evaluation:

- PAIR / JUNCTION / NO-REPAIR discrimination;
- repair-scene exact at fixed false-repair budget;
- high-degree junction behavior;
- patient-cluster uncertainty;
- risk-coverage after the same perturbation-consistency gate.

## Limitations

- gaps are controlled synthetic deletions, not natural segmentation failures;
- endpoints / tangents derive from annotated geometry plus explicit noise, not a predicted segmentation pipeline;
- the test set has already been inspected and must not be used for another publication-facing tuning loop;
- 0 observed false is not proof of zero population risk;
- CT-conditioned claims remain blocked until a true original ImageCAS CT is matched to an ImageCAS-X anatomical case.
