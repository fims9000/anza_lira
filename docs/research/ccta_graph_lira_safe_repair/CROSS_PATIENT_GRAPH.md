# Cross-patient Graph-LIRA stress benchmark

Date: 2026-09-20.

Status: controlled geometry experiment on real ImageCAS-X centerlines. This is **not** a natural-gap clinical validation and it does not use CT intensity evidence.

## Goal

Test whether the structural conclusions found on scan 921 survive a patient-level transfer to scan 953, and vice versa.

The benchmark creates controlled scenes from real labelled coronary centerlines:

- one hidden bifurcation / junction from the anatomical centerline graph;
- one independent ordinary hidden gap;
- endpoint position and tangent perturbations;
- a geometry-only pair classifier;
- an analytic variable-degree junction score;
- either independent/sequential decisions or a single joint structural optimization.

The held-out patient's labels are not used to fit the pair classifier or geometry thresholds.

The joint optimizer uses fixed exploratory constants:

- pair term weight: `1.0`;
- degree bonus: `0.35`.

These are frozen from the earlier exploratory branch and are not retuned on the held-out patient.

## Data used

- scan 921 ImageCAS-X left/right labelled centerlines;
- scan 953 ImageCAS-X left/right labelled centerlines.

Generated scene counts in this run:

- 921: 447 scenes from 5 usable branch points;
- 953: 648 scenes from 8 usable branch points.

The reduced usable count reflects the scene-construction constraints, not the total number of branch points present in the files.

## Primary stress setting: 30 deg tangent error + 1 mm endpoint jitter

### Train 921 -> test 953

| Method | Exact scene | False scene | Branch recall | Ordinary-gap recall |
|---|---:|---:|---:|---:|
| local independent pairs | — | 53.70% | — | 95.37% |
| sequential junction -> pair | 87.04% | 8.80% | 96.30% | 87.04% |
| joint Graph-LIRA | **94.44%** | **1.85%** | **98.92%** | **94.44%** |

Pair-ranking AUROC on held-out 953: `0.98269`.

### Train 953 -> test 921

| Method | Exact scene | False scene | Branch recall | Ordinary-gap recall |
|---|---:|---:|---:|---:|
| local independent pairs | — | 54.36% | — | 95.30% |
| sequential junction -> pair | 80.54% | 9.40% | 94.97% | 85.91% |
| joint Graph-LIRA | **88.59%** | **2.68%** | **96.92%** | **93.96%** |

Pair-ranking AUROC on held-out 921: `0.98136`.

## Perturbation-consistency gate

For the 30 deg + 1 mm stress setting, each held-out scene is re-evaluated 15 times after an additional approximately 0.8 mm endpoint perturbation and 20 deg tangent perturbation.

The consistency score is the fraction of reruns that produce the same structured decision as the unperturbed joint Graph-LIRA output.

### Train 921 -> test 953

| Consistency threshold | Coverage | Accepted | False among accepted | Exact among accepted |
|---:|---:|---:|---:|---:|
| 0.80 | 78.70% | 170 | 0 / 170 | 98.24% |
| 0.90 | 60.19% | 130 | 0 / 130 | 99.23% |
| 0.95 | 44.44% | 96 | 0 / 96 | 98.96% |

### Train 953 -> test 921

| Consistency threshold | Coverage | Accepted | False among accepted | Exact among accepted |
|---:|---:|---:|---:|---:|
| 0.80 | 69.80% | 104 | 0 / 104 | 99.04% |
| 0.90 | 53.69% | 80 | 0 / 80 | 100% |
| 0.95 | 34.23% | 51 | 0 / 51 | 100% |

These zero-false counts are finite-sample observations in this controlled benchmark, not proof of zero clinical false-link risk.

## Heavier stress

At 45 deg + 1 mm:

- train 921 -> test 953: joint exact `87.96%`, false `6.94%`; sequential false `12.96%`; local independent false `69.44%`;
- train 953 -> test 921: joint exact `85.91%`, false `12.08%`; sequential false `17.45%`; local independent false `85.91%`.

Geometry therefore still degrades under strong direction noise. This is exactly the residual regime where CT image evidence is expected to be useful.

## Interpretation

The result strengthens three working hypotheses:

1. the benefit of joint graph reasoning is not confined to one patient's coronary tree;
2. the main failure of independent pair decisions is structural incompatibility, not simply poor pair AUROC;
3. perturbation consistency is a useful selective-repair signal across patient transfer.

It does **not** establish performance on natural segmentation failures because the gaps and perturbations are controlled from ground-truth centerlines.

## Machine artifacts

- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_summary.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_risk_coverage.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_settings.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_protocol.json`
- per-scene table: archived separately in the same result directory.

Exact exploratory source is archived under:

- `scripts/research/ccta_graph_lira_safe_repair/ccta_graph_lira_cross_patient.py.gz.b64`
