# Synthetic evaluator audit (2026-08-17)

The original synthetic test artifacts remain immutable. Their segmentation and
trace measurements, checkpoint provenance, and test-open provenance remain
valid. Model-labelled continuation values for B0/B1/C0 and the `P >= 0.5`
routing readout are partially invalidated for mechanism claims.

Evaluator v2.1 separates three evidence families:

- Family A contains model-comparable visible segmentation and latent trace
  measurements.
- Family B contains threshold-free route measurements only for a model that
  actually emits transport state. It reports top-1 hit, true probability mass,
  MRR, average precision, normalized entropy, excess over chance, and
  topology-constrained X/T/Y assignments.
- Family C is one diagnostic named
  `geometry_only_minimum_angle_heuristic`. It explicitly records
  `uses_generator_branch_geometry=true` and is not a neural-model result.

X uses the best of three perfect matchings. T selects one pair. Y selects a hub
and its two incident continuation pairs. Prediction construction receives only
the declared topology, branch IDs, and route scores; generator continuation
truth is used afterward for scoring.

False bridge keeps coverage threshold 0.50 as primary. Thresholds 0.25 and
0.75 are validation-only sensitivity analyses and cannot select a test
threshold. If every method remains at 1.0 after the contract audit, the verdict
is `FALSE_BRIDGE_ENDPOINT_SATURATED_NONDISCRIMINATIVE`.

Original `test[0:2000]` corrected results may only be labelled
`POSTHOC_REANALYSIS_NOT_CONFIRMATORY`. A replacement receipt must freeze the
evaluator, checkpoint hashes, and validation-selected visible thresholds before
opening the disjoint `test[2000:4000]` stream. That result is labelled
`REPLACEMENT_CONFIRMATION_AFTER_EVALUATOR_AUDIT`, never the original
preregistered test.
