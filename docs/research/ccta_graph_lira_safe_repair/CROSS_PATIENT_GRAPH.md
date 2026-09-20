# Cross-patient Graph-LIRA stress benchmark

Date: 2026-09-20.

Status: controlled geometry experiment on real ImageCAS-X centerlines. This is **not** a natural-gap clinical validation and it does not use CT intensity evidence.

Canonical implementation/result version: **V2**.

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

These are frozen from earlier exploratory work and are not retuned on the held-out patient.

## V2 reproducibility correction

The first exploratory implementation seeded perturbation-consistency reruns partly from the global scene index. Adding another stress configuration therefore changed the random perturbations used for an already-existing scene, even though the base scene and the structural result were unchanged.

V2 fixes this by defining a stable scene identity:

```text
case : branch_index : repetition : angle : jitter
```

and deriving every perturbation seed from this identity plus the perturbation repetition index.

Verification:

- rerunning V2 with the 60-degree stress configuration present or absent produced the **same 30-degree risk/coverage table**;
- all summary rows up to 45 degrees were also identical.

Therefore V2 is the canonical cross-patient result set. The older `cross_patient_graph_*.csv` artifacts remain as provenance but their perturbation-consistency tables are superseded by `cross_patient_graph_v2_*.csv`.

Exact archived source:

`scripts/research/ccta_graph_lira_safe_repair/ccta_graph_lira_cross_patient_v2.py.gz.b64`

Source SHA256:

`6beb28bb28a6687260763eb30e40e71c8bccf8e6fb1422156692b12b1dcf2aad`

## Data used

- scan 921 ImageCAS-X left/right labelled centerlines;
- scan 953 ImageCAS-X left/right labelled centerlines.

With all four stress settings enabled, V2 contains:

- scan 921: 596 generated scenes from 5 usable branch points;
- scan 953: 864 generated scenes from 8 usable branch points.

The primary 30-degree evaluation itself contains 149 held-out scenes for 921 and 216 for 953.

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

The important point is that pair AUROC is already high while independent local decisions remain structurally unsafe. The gain therefore comes from solving the **joint configuration**, not simply from obtaining a good pair ranker.

## Perturbation-consistency gate

For the 30 deg + 1 mm setting, each held-out scene is re-evaluated 15 times after an additional approximately 0.8 mm endpoint perturbation and 20 deg tangent perturbation.

The consistency score is the fraction of reruns that produce the same complete structured decision as the unperturbed joint Graph-LIRA output.

### Train 921 -> test 953

| Consistency threshold | Coverage | Accepted | False among accepted | Exact among accepted |
|---:|---:|---:|---:|---:|
| 0.60 | 92.13% | 199 | 0 / 199 | 96.48% |
| 0.80 | 77.78% | 168 | 0 / 168 | 98.21% |
| 0.90 | 63.43% | 137 | 0 / 137 | 99.27% |
| 0.95 | 43.06% | 93 | 0 / 93 | 100% |

### Train 953 -> test 921

| Consistency threshold | Coverage | Accepted | False among accepted | Exact among accepted |
|---:|---:|---:|---:|---:|
| 0.60 | 88.59% | 132 | 0 / 132 | 93.18% |
| 0.80 | 68.46% | 102 | 0 / 102 | 98.04% |
| 0.90 | 44.97% | 67 | 0 / 67 | 98.51% |
| 0.95 | 33.56% | 50 | 0 / 50 | 100% |

At threshold 0.50 the reverse direction still contains one accepted false scene (1 / 136). In the current two-patient benchmark all observed false structural decisions fall below consistency 0.60.

**This does not authorize choosing 0.60 as the final operating threshold.** The threshold grid has already been inspected on these held-out patients. A publication-grade operating threshold must be chosen on future calibration patients and then frozen before confirm/test patients are opened.

The zero-false counts above are finite-sample observations, not proof of zero clinical false-link risk. Exact finite-sample uncertainty is documented separately in `SELECTIVE_RISK_UNCERTAINTY.md`.

## Heavier stress

### 45 deg + 1 mm

- 921 -> 953: joint exact `87.96%`, false `6.94%`; sequential false `12.96%`; local independent false `69.44%`.
- 953 -> 921: joint exact `85.91%`, false `12.08%`; sequential false `17.45%`; local independent false `85.91%`.

### 60 deg + 1.5 mm

- 921 -> 953: joint exact `75.00%`, false `18.98%`; sequential exact `64.35%`, false `27.31%`; local independent false `92.59%`.
- 953 -> 921: joint exact `69.80%`, false `28.19%`; sequential exact `68.46%`, false `28.19%`; local independent false `96.64%`.

Pair AUROC also falls to about `0.9223` and `0.9147` in the two transfer directions at the 60-degree setting.

This is the geometry ceiling we wanted to expose: when endpoint direction itself becomes unreliable, graph constraints help but cannot recover missing image evidence.

## Branch-level failure localization

At 30 deg + 1 mm:

- held-out 953 D1: 60 / 60 exact;
- held-out 953 RCA: 30 / 30 exact;
- held-out 953 LAD degree-3: exact `86.67%`, false `6.67%`;
- held-out 953 LCX degree-3: exact `89.39%`, false `3.03%`;
- held-out 921 LAD degree-3: exact `96.61%`, false `0%`;
- held-out 921 LCX degree-3: exact `93.33%`, false `3.33%`;
- held-out 921 LCX degree-4: exact `63.33%`, false `6.67%`, incomplete-but-nonfalse `30.0%`.

The degree-4 LCX stratum is the clearest remaining geometry-only weakness. Its mean perturbation consistency is `0.647` and median `0.60`.

## Interpretation

The result strengthens four bounded working hypotheses:

1. the benefit of joint graph reasoning is not confined to one patient's coronary tree;
2. a high pair-ranking AUROC does not make independent repair decisions structurally safe;
3. perturbation consistency is a useful selective-repair signal across patient transfer;
4. the residual failure regime is concentrated around difficult branch identity / higher-degree junctions and under strong tangent corruption, which gives a precise target for future CT evidence.

It does **not** establish performance on natural segmentation failures because the gaps and perturbations are controlled from ground-truth centerlines.

## Canonical machine artifacts

- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_summary.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_risk_coverage.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_settings.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_protocol.json`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_branch_strata_30deg.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_failure_cases_30deg.csv`

Older `cross_patient_graph_*.csv` files are retained only for provenance.
