# Expanded matched-CCTA plan — 28 patients from 801-1000.z04

Date: 2026-09-21  
Branch: `research/ccta-graph-lira-safe-repair`

## Why this expansion is needed

The six-case exact-matched CCTA pilot established that radial 2.5-D intensity context is locally informative, but direct insertion into the frozen Graph-LIRA relation layer did not preserve the low-false objective across patients. The remaining bottleneck is therefore patient-general calibration, not proof that CT contains signal.

No further threshold tuning is allowed on held-out scans 980 / 984.

## Same-disk cohort

The already-downloaded Kaggle `801-1000.z04` volume contains a contiguous ImageCAS ID range that can be joined directly to ImageCAS-X annotations. Restricting the expansion to IDs 953–984 gives 28 exact anatomical cases while preserving the official split:

- train: 17 patients;
- validation: 5 patients;
- test: 6 patients.

Patients:

- train: 953, 955, 956, 959, 960, 963, 964, 967, 969, 970, 971, 975, 976, 977, 979, 982, 983;
- validation: 957, 961, 965, 966, 974;
- test: 954, 958, 972, 973, 980, 984.

This keeps the original ImageCAS-X patient partition intact.

## Frozen geometry-matched relation plan

The exact same controlled stress definition used by the six-case real-CT pilot is retained:

- true within-branch gap: 4 mm;
- positive sampling step: 2 mm;
- hard-negative endpoint sampling step: 1.5 mm;
- candidate distance: 2.5–6.5 mm;
- wrong branch must come from a different polyline and different anatomical segment;
- anchor alignment >= 0.6;
- decoy alignment >= 0.6;
- axis consistency >= 0.4;
- one-to-one positive / hard-negative matching uses geometry only.

Frozen balanced examples generated before reading any new CT intensity:

- train: 379 positive + 379 hard negative;
- validation: 134 + 134;
- test: 167 + 167.

Thus the expanded local relation study has 1,360 examples across 28 independent patients.

## Image representation

Do not escalate model capacity in this expansion.

Use the current strongest representation:

- candidate-aligned radial 2.5-D CCTA profile;
- 17 positions along the candidate corridor;
- centerline HU profile;
- 8 angular samples on radii 1, 2 and 3 mm;
- robust HU clipping to [-300, 1200];
- the same radial summary family used in the previous matched-CCTA pilot.

Compare:

1. geometry only;
2. radial CT only;
3. geometry + radial CT.

The classifier remains the same lightweight `StandardScaler + LogisticRegression(C=1, class_weight=balanced)` baseline for this calibration check.

## Frozen evaluation

Threshold selection:

- validation patients only;
- maximize recall subject to FPR <= 5%.

Test thresholds must not be changed after seeing the six held-out test patients.

Primary outputs:

- patient-level AUROC / AUPRC;
- recall and FPR at the frozen validation operating point;
- patient heterogeneity;
- patient-cluster bootstrap;
- CT alignment manifest / SHA256;
- subsequent insertion into the already frozen Graph-LIRA relation layer.

The canonical Graph-LIRA geometry components remain frozen:

- candidate generator;
- candidate identity / compatibility models;
- four-class relation head;
- relation confidence `tau=0.85`;
- perturbation consistency gate `0.60`.

## Local execution boundary

Raw CT remains on the user's workstation and must not be uploaded to Git.

A compact local runner was prepared that reads the CT entries directly from the already-downloaded `801-1000.z04 + 801-1000.zip`, verifies each CT grid against frozen ImageCAS-X geometry, extracts only radial relation features, evaluates the three lightweight baselines, and emits a compact ZIP with no raw CCTA.

Runner bundle SHA256:

`3e215e4e1bded4dbb824142869701a509cb806b882013fc80eccb1649b622d21`

Core extractor SHA256:

`c5c0d287bdf30bc4879c03398037afd6381b9b0d63a341b2079311d66e1ce1de`

Frozen pair-plan SHA256:

`4629bd082457bdb14a3c7653e6fc8b1592d648d91f087e14b522c38f0ccca523`

Expected CT geometry SHA256:

`309ba692c82cef14dcb80bb5863d49775b35840c83fb36847b865da8c4a83359`

The split-ZIP reader was unit-checked on a synthetic five-disk archive and reproduced raw entry bytes exactly across the z04 -> final-zip boundary.

## Exact resume action

Run the compact feature extractor locally against the existing Kaggle z04 archive and return only:

`GraphLIRA_CT_expanded_radial_results.zip`

Then:

1. audit all 28 CT / ImageCAS-X grids;
2. compare geometry vs radial CT vs geometry+CT under the frozen validation rule;
3. run patient-level / patient-cluster uncertainty;
4. refit only the image-conditioned relation-presence layer;
5. inject it into frozen Graph-LIRA;
6. apply the frozen `tau=0.85` and consistency `0.60` policy;
7. decide whether calibration is now stable enough to justify an ANZA ablation.
