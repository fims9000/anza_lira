# Negative results that matter

Date: 2026-09-26

This is not a dump of every failed experiment. These are the negative results that materially determine the current architecture.

## 1. Direct CT add-only insertion was unsafe

Six-patient matched-CCTA pilot:

Canonical geometry Graph-LIRA on held-out test:

- exact: 53.33%
- false structural scene: 10.00%
- incomplete: 36.67%

CT add-only:

- exact: 56.67%
- false structural scene: 23.33%
- incomplete: 20.00%

So CT local evidence was useful, but naïvely adding CT-positive relations increased false structural repair too much.

Validation-only hysteresis did not fix transfer:

- selected validation <=10% false rule: val false 3.57%;
- held-out test false: 20.00%.

**Consequence:** do not reproduce the old add-only/hysteresis rule as the final integration. Build a patient-general CT-conditioned relation model.

Source:
`results/ccta_graph_lira_safe_repair/2026-09-21/ct_graph_hybrid_summary.csv`
`results/ccta_graph_lira_safe_repair/2026-09-21/ct_graph_hysteresis_selected.csv`

## 2. More complex image representations did not beat radial 2.5-D on the tiny matched cohort

Six-case representation comparison at the validation-defined low-FPR operating point:

- radial 2.5-D recall: 61.70%, FPR 0%;
- geometry + radial: 63.83%, FPR 0%;
- coarse 3-D tube + PCA: 21.28%, FPR 2.13%;
- geometry + coarse 3-D tube: 27.66%, FPR 2.13%;
- small cross-section CNN: 19.15%, FPR 0%.

**Consequence:** bigger representation capacity was not justified with only two training patients. First increase patient-general evidence and solve calibration / relation structure. Then compare CNN vs ANZA cleanly.

Source:
`results/ccta_graph_lira_safe_repair/2026-09-21/ct_representation_compare_test.csv`

## 3. Binary broken-mask context helped mainly as a veto / safety cue

On the large controlled geometry benchmark, broken-mask occupancy context could reduce false structural decisions, but often increased incompleteness / under-repair.

Example at the 5% operating family, test30:

Geometry:
- exact 51.55%
- false 6.78%
- incomplete 41.67%

Geometry + mask:
- exact 53.86%
- false 5.93%
- incomplete 40.21%

This was useful evidence that local spatial context matters, but it did not solve branch identity or automatic repair.

**Consequence:** occupancy context is not a substitute for CCTA intensity and not the main next direction.

Source:
`results/ccta_graph_lira_safe_repair/2026-09-20/mask_veto_summary.csv`

## 4. Candidate ranking is not the dominant bottleneck

In the 800-case geometry benchmark, candidate generation had 100% true-candidate recall under the controlled stress protocol.

For repair-needed test30 scenes:

- actual relation-type system exact: 28.45%;
- oracle relation presence + existing geometry top-1: 70.81%;
- oracle presence + top-2: 89.47%;
- oracle presence + top-3: 94.57%.

Test45:

- actual: 18.50%;
- oracle presence + top-1: 62.75%;
- top-2: 82.40%;
- top-3: 89.97%.

**Consequence:** stop spending iterations on candidate geometry. The main problem is relation existence / identity: whether PAIR / JUNCTION / BOTH / NONE is actually present.

Source:
`results/ccta_graph_lira_safe_repair/2026-09-20/large_scale_repair_aware_diagnostics.csv`

## 5. High selective coverage did not mean high repair recall

The geometry-only selective system was very good at stable NO-REPAIR decisions, but many accepted repair-needed scenes remained incomplete.

At strict relation gate 0.70 on test30:

- pair_only coverage 66.25%, exact among accepted 0%;
- junction_only coverage 43.58%, exact among accepted 2.45%;
- mixed coverage 45.27%, exact among accepted 29.85%;
- accepted NO-REPAIR strata were nearly always correct.

**Consequence:** report repair-needed exact separately from overall selective accuracy. Do not let correct abstention / NO-REPAIR dominate the claim.

## What can be ignored for now

Do not spend time reconstructing every old threshold grid or every weak sequence experiment.

The above five negative results are the ones that directly justify the current task:

**build patient-general PAIR_CT + JUNCTION_CT evidence, then a CT-conditioned relation-type model, then evaluate it inside frozen Graph-LIRA.**
