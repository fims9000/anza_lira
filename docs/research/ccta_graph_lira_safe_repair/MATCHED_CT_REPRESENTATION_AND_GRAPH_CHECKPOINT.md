# Matched CCTA representation + Graph-LIRA checkpoint

Date: 2026-09-21  
Branch: `research/ccta-graph-lira-safe-repair`

## Scope

This checkpoint continues the six-patient exact-matched ImageCAS / ImageCAS-X pilot:

- train: 953, 964
- validation: 957, 966
- test: 980, 984

The raw CCTA volumes remain outside Git. Alignment/provenance is frozen in `MATCHED_CT_6CASE_CHECKPOINT.md` and `matched_ct_alignment_6case.csv`.

Two questions were tested:

1. which CT representation is most useful on the frozen geometry-matched wrong-branch relation stress;
2. whether that CT signal can already be inserted safely into the frozen four-class PAIR / JUNCTION / BOTH / NONE Graph-LIRA relation layer.

No test threshold was tuned.

## Representation comparison

All representations use the same patient split and the same balanced hard relation set. The operating threshold for every representation is selected on matched validation patients only by maximizing recall subject to FPR <= 5%.

Held-out test:

| representation | AUROC | AUPRC | recall | FPR | TP / FP |
|---|---:|---:|---:|---:|---:|
| geometry | 0.9873 | 0.9881 | 0.0213 | 0.0000 | 1 / 0 |
| radial 2.5-D | 0.9244 | 0.9487 | 0.6170 | 0.0000 | 29 / 0 |
| geometry + radial 2.5-D | 0.9488 | 0.9577 | **0.6383** | **0.0000** | **30 / 0** |
| coarse 3-D tube + PCA | 0.7175 | 0.7630 | 0.2128 | 0.0213 | 10 / 1 |
| geometry + coarse 3-D tube + PCA | 0.9176 | 0.9126 | 0.2766 | 0.0213 | 13 / 1 |
| small cross-section CNN | 0.8551 | 0.8664 | 0.1915 | 0.0000 | 9 / 0 |
| geometry + cross-section CNN | 0.8995 | 0.8954 | 0.1915 | 0.0000 | 9 / 0 |

Validation selected the same qualitative ordering: geometry + radial 2.5-D recovered 51/59 positives at FPR 2/59, while the raw 3-D and CNN representations were less stable.

### Interpretation

The useful image signal is currently simple local lumen continuity / contrast, not a high-capacity representation. With only two matched training patients, the coarse 3-D tube and CNN overfit / fail to transfer. ANZA should not be tested as a large architecture replacement at this stage; if tested, it must be a compact explicit ablation against the radial representation.

The geometry AUROC remains higher than the CT models. The CT advantage is specifically at the conservative validation-frozen operating point: geometry is so confident that its threshold becomes extremely restrictive on held-out patients, whereas radial CT evidence preserves substantially more true relations without observed false positives in this finite sample.

`0/47` false negatives is an observed count, not a population-risk claim.

## Frozen canonical Graph-LIRA baseline reproduced

The full 800-case geometry-only pipeline was independently regenerated from the archived checkpoint source and the full ImageCAS-X centerline package.

The regenerated four-class relation head re-selected exactly the canonical validation setting:

- model: HGB
- relation confidence `tau = 0.85`

Full official results reproduced:

- val30 exact 54.96%, false 8.95%
- test30 exact 55.46%, false 10.92%
- test45 exact 51.37%, false 8.05%

This confirms that the matched-CT experiment is being compared against the same frozen geometry baseline rather than a reconstructed approximation.

## First real-CT insertion into the four-class relation layer

A conservative auxiliary CT presence layer was built only from the matched six cases. The canonical full-800 geometry candidate models and the frozen four-class relation head remained unchanged.

The first rule allowed CT to **add** a missing PAIR or JUNCTION relation while retaining canonical geometry for candidate identity and compatibility.

On the matched subset:

| method | split | exact | false | incomplete |
|---|---|---:|---:|---:|
| frozen canonical | val | 42.86% | 10.71% | 46.43% |
| frozen canonical | test | 53.33% | 10.00% | 36.67% |
| CT add-only | val | 50.00% | 14.29% | 35.71% |
| CT add-only | test | 56.67% | **23.33%** | 20.00% |

The add-only rule therefore fails the safety objective. A small exact-rate gain is not acceptable when false structural repair more than doubles on held-out matched test patients.

A second validation-only hysteresis search allowed CT to veto low-presence relations and add only high-presence relations. It looked substantially better on the two validation patients, but still failed to transfer:

- validation-selected <=10% false setting: exact 57.14%, false 3.57%;
- held-out test: exact 50.00%, false 20.00%.

A validation setting matched to canonical validation risk gave:

- validation exact 60.71%, false 10.71%;
- held-out test exact 56.67%, false 20.00%.

### Central conclusion

Real CCTA intensity **does contain useful local continuation evidence**, and radial 2.5-D is currently the strongest representation. However, with only two matched train and two matched validation patients, the scene-level calibration needed to inject that evidence into the Graph-LIRA PAIR / JUNCTION / NONE decision layer is not stable across patients.

Therefore:

- do **not** promote the six-case CT hybrid as an improved Graph-LIRA model;
- do **not** tune more thresholds on scans 980 / 984;
- do **not** spend time on a larger CNN or ANZA until patient-level calibration is better supported;
- the highest-value next step is to add more original ImageCAS CTs that already have ImageCAS-X labels, while retaining the official patient split.

This is a useful negative result: the bottleneck has shifted from “is CT informative?” to “is the CT relation-presence calibration patient-general enough to preserve low false-link risk?”

## Next executable step

Use additional original ImageCAS CTs from the already-downloaded Kaggle 801-1000 multipart block, prioritizing cases that physically reside in the same downloaded `.z04` volume. Expand matched train/validation/test patient counts without changing the official split, then rerun:

1. geometry + radial 2.5-D relation evidence;
2. four-class relation presence calibration;
3. frozen `tau=0.85`;
4. frozen perturbation-consistency gate `0.60`;
5. patient-level bootstrap / risk-coverage.

Only after calibration stabilizes across more patients should ANZA and learned cross-section encoders be revisited.
