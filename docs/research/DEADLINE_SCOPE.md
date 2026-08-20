# ANZA-LIRA deadline scope

The deadline result includes Setting A crowd-to-expert same-section
reconstruction and the corrected synthetic evaluator with replacement
confirmation. Settings B and C are deferred as `NOT_RUN_DEADLINE_SCOPE` and are
not used in submitted claims.

The four primary model families are U-Net, Deformable U-Net, the previous
ANZA-LIRA operator, and the frozen mode-resolved transport model. Their results
use seeds 41, 42, and 43. Seeds are averaged within each seismic section before
section-level aggregation. Paired comparisons form model deltas for the same
section and seed, average those three seed deltas within section, and bootstrap
section IDs at least 10,000 times.

The no-replay, no-fuzzy, and no-directional variants are seed-42-only
ablations. They cannot be promoted post hoc to the primary model. Qualitative
examples are selected from expert section IDs by SHA-256 rank with the frozen
salt `anza-v2-qual-20260817`, before inspecting model errors.

Corrected route metrics apply only to models that expose transport outputs.
The generator-conditioned minimum-angle rule is reported separately as
`geometry_only_minimum_angle_heuristic`. A saturated false-bridge endpoint is
reported as non-discriminative rather than tuned on test data.
