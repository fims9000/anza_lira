# Expanded matched-CCTA 28-case experiment

This directory freezes the next matched-CCTA calibration experiment for the Graph-LIRA safe-repair line.

## Cohort

Official ImageCAS-X patient split is preserved:

- train (17): 953, 955, 956, 959, 960, 963, 964, 967, 969, 970, 971, 975, 976, 977, 979, 982, 983
- validation (5): 957, 961, 965, 966, 974
- test (6): 954, 958, 972, 973, 980, 984

The frozen plan contains 1,360 balanced geometry-matched relation examples:

- train 758 = 379 positive + 379 hard negative
- validation 268 = 134 + 134
- test 334 = 167 + 167

## Frozen local comparison

1. geometry only
2. radial 2.5-D CCTA only
3. geometry + radial 2.5-D CCTA

Use the same lightweight StandardScaler + LogisticRegression(C=1, class_weight=balanced) model. Select the decision threshold on validation patients only by maximizing recall subject to FPR <= 5%.

## Frozen downstream policy

Do not retune:

- canonical Graph-LIRA geometry candidate logic
- four-class relation head
- relation confidence tau = 0.85
- perturbation consistency = 0.60

## Files

- `expected_ct_geometry.csv`: exact CT grid expected for every patient before HU extraction
- `pair_plan_counts.csv`: frozen split / class counts
- exact plan generator: `../../../scripts/research/ccta_graph_lira_safe_repair/build_expand_pair_plan.py.gz.b64`
- exact CT extractor: `../../../scripts/research/ccta_graph_lira_safe_repair/extract_radial_features_z04.py.gz.b64`

The full research rationale, download instructions, execution command, output contract and decision tree are in:

`../../../docs/research/ccta_graph_lira_safe_repair/EXPANDED_CT_28CASE_EXECUTION_AND_ROADMAP.md`

## Raw data

Raw medical data are intentionally excluded from Git.

Required external archive members:

- `801-1000.z04`
- `801-1000.change2zip` renamed to `801-1000.zip`

Official source: https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas

## Current status

Preflight has been executed. Source compiles, 1,360 rows / 28 patients / split isolation / class balance are verified. Full extraction is blocked only because the two raw Kaggle split-archive pieces are not mounted in the current execution environment.
