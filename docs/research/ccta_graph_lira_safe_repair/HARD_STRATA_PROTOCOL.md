# Frozen hard strata for the next CT-conditioned experiment

Date: 2026-09-20.

Status: **pre-image-context freeze**.

The purpose of this document is to stop us from choosing the interesting anatomical cases after seeing the future CT-model results.

The strata below are defined from anatomy and the geometry-only ambiguity audit on scans 921 and 953. They are **not** defined from failures of a future CNN, ANZA encoder, Transformer or Mamba model.

## Why a hard-stratum protocol is needed

The canonical V2 cross-patient Graph-LIRA experiment shows that average pair AUROC is not enough. Pair AUROC is around 0.98 in the primary transfer setting, yet independent pair decisions still create false structural links in more than half of controlled multi-gap scenes.

The ambiguity audit also shows that plausible wrong continuations are anatomically concentrated. Across the top directed wrong-branch relations recorded on scans 921 and 953, the most common relations include:

- LM -> IM: 130;
- LM -> LAD: 128;
- R-PDA -> R-PLA: 105;
- LAD -> D1: 88;
- R-PLA -> R-PDA: 88;
- LAD -> D2: 84;
- LCX -> OM1: 83.

These counts are controlled-geometry candidate events, not natural clinical error frequencies.

## Frozen anatomical strata

### S1 — proximal left hub

Members: `LM, LAD, LCX, IM`.

Reason: several branches originate or run close to the left-main bifurcation/trifurcation region, so a locally plausible direction can correspond to the wrong anatomical continuation.

Observed top-pair audit count in the two available labelled hearts: `422`.

### S2 — diagonal ambiguity

Members: `LAD, D1, D2`.

Reason: diagonal branches are the dominant wrong-branch alternatives around LAD in the current audit.

Observed top-pair audit count: `249`.

### S3 — circumflex / marginal ambiguity

Members: `LCX, OM1, OM2, IM`.

Reason: LCX–OM and IM–OM relations create repeated geometrically plausible alternatives.

Observed top-pair audit count: `261`.

### S4 — distal right bifurcation

Members: `RCA, R-PDA, R-PLA`.

Reason: R-PDA and R-PLA are particularly strong mutual decoys in scan 953.

Observed top-pair audit count: `301`.

### S5 — high-degree junction

Definition: `true junction degree >= 4`.

Reason: this is a structural class, not a named-vessel class. In the canonical V2 patient-transfer benchmark the held-out degree-4 LCX junction is the clearest remaining weakness of geometry-only Graph-LIRA.

At `30 deg + 1 mm` for train-953 -> test-921:

- exact scene rate: `63.33%`;
- false scene rate: `6.67%`;
- incomplete but non-false: `30.0%`;
- branch recall: `89.17%`;
- ordinary-gap recall: `90.0%`;
- perturbation consistency mean: `0.647`;
- perturbation consistency median: `0.60`;
- only `10 / 30 = 33.33%` of scenes survive consistency `>= 0.80`.

Among those 10 accepted scenes at `>= 0.80`, no false structural link was observed, but exactness was only `80%`. At `>= 0.90` only 2 / 30 scenes remain, so that threshold is too selective to tell us much about this stratum.

This stratum is therefore mandatory in the CT-context evaluation.

## Primary evaluation once matched CT is available

Every local representation must use the same frozen candidate scenes and Graph-LIRA decision layer.

Compare:

```text
G0  geometry only
I1  radial 2.5-D CT
I2  candidate-aligned 3-D tube
I3  full cross-section CNN encoder -> sequence context
I4  full cross-section ANZA encoder -> sequence context
```

Sequence aggregation starts simple. A Transformer/Mamba comparison is only justified if the full-spatial token representation exposes a remaining sequence-context failure.

## Metrics per stratum

For each of S1-S5 report:

- pair AUROC / AUPRC;
- joint-structure exact rate;
- false structural-link rate;
- incomplete / abstention rate;
- perturbation-consistency risk/coverage;
- recovery at the same false-link budget;
- junction-degree accuracy for S5;
- path success only conditional on the correct accepted relation.

The headline comparison is not average AUROC. It is:

> additional recovery at a fixed false-link risk, especially in the hard anatomical strata.

## No-retuning rule

The stratum definitions in this file are frozen before matched CT is opened for model development.

Do not:

- drop a stratum because a model performs badly on it;
- add a new named stratum because a model happens to fail there;
- select consistency thresholds on the held-out patient;
- change the candidate geometry separately for image and geometry baselines.

The current two held-out patients have already been inspected while developing the structural pipeline, so they must not later be described as untouched final clinical confirm data. Future publication-grade confirm patients must be separate.

Exploratory failure views may still be produced, but they must be labelled exploratory and kept separate from the frozen primary strata.

## Machine artifacts

- `results/ccta_graph_lira_safe_repair/2026-09-20/hard_strata_summary.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/wrong_branch_pairs_aggregated.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_branch_strata_30deg.csv`
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_failure_cases_30deg.csv`
