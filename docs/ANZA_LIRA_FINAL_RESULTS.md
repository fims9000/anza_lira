# ANZA-LIRA CRACKS Structural Stability V1.1

## Research question

Does reciprocal determinant-one B3 improve topology robustness over free-determinant B2 and ordinary consistency B1?

## Frozen design

Section-disjoint CRACKS, train-only normalization, 12 from-scratch runs, clean calibration, one-shot development, five perturbation families at three severities. White remains unknown.

## Development

- Clean Dice B3-B2: `{'bootstrap_seed': 20260819, 'ci_lower': -0.0003250028522967524, 'ci_upper': 0.007112184750602715, 'estimate': 0.003072599339701858, 'resamples': 10000}`
- Clean clDice B3-B2: `{'bootstrap_seed': 20260819, 'ci_lower': 0.00042862144456512924, 'ci_upper': 0.005268534779574142, 'estimate': 0.0027377191953559844, 'resamples': 10000}`
- Shifted clDice B3-B2: `{'bootstrap_seed': 20260819, 'ci_lower': 0.00042930075849295077, 'ci_upper': 0.004099964460582737, 'estimate': 0.002183114125314378, 'resamples': 10000}`
- Topology-drop ratio B3/B2: `{'bootstrap_seed': 20260819, 'ci_lower': 0.8684347216670577, 'ci_upper': 1.154913397552947, 'estimate': 0.9965658537356187, 'resamples': 10000}`
- B3 geometry: `{'axis_mae_deg': 15.350085726872653, 'd_mae': 0.06767894891696337, 'd_quantiles': {'q05': 0.21271397769451142, 'q25': 0.24421477317810059, 'q50': 0.27644574642181396, 'q75': 0.29800066351890564, 'q95': 0.31399683356285096}, 'determinant_residual_mean': 1.0372653331131619e-08, 'm_quantiles': {'q05': 0.0, 'q25': 0.0, 'q50': 0.0, 'q75': 0.0, 'q95': 0.0}, 'pixels': 76633, 'variant': 'B3'}`

## Confirm and expert

Confirm and expert remained unopened because access is conditional on the development decision.

## Final status

`STOP_ANZA_STABILITY_NO_INCREMENTAL_VALUE`
