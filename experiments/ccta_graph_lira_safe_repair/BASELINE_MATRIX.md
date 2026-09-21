# Baseline matrix for coronary connectivity repair

Date: 2026-09-21
Status: frozen comparison design before expanded matched-CCTA test results

| ID | Baseline | Uses CCTA intensity | Global graph | Can abstain | Purpose |
|---|---|---:|---:|---:|---|
| B0 | no repair | no | no | yes (always) | zero-forced-link reference |
| B1 | nearest endpoint | no | no | optional | trivial geometry |
| B2 | distance + tangent direction | no | no | threshold | local geometry |
| B3 | OGMC-style geometric matching | no | yes-ish matching constraints | threshold | external-style geometry/reconnection |
| B4 | regularized-walk-style | yes | path graph | threshold | CorSegRec-family comparison |
| B5 | radial 2.5-D CCTA | yes | no | threshold | image-only local evidence |
| B6 | geometry + radial 2.5-D | yes | no | threshold | local hybrid |
| B7 | canonical Graph-LIRA | no | yes | yes | frozen internal baseline |
| B8 | CT-conditioned Graph-LIRA | yes | yes | yes | target method |

## Fairness rules

1. Candidate generator must be identical wherever the method permits it.
2. Patient train/validation/test split remains fixed.
3. Hyperparameters and thresholds are selected on validation only.
4. Test patients never select thresholds.
5. Report failure to repair separately from false repair.
6. Report patient-cluster uncertainty.
7. External-inspired approximations must be called "-style" unless exact authors' code/equations are reproduced.
8. No method receives anatomical ground-truth segment identity as an inference feature.
9. Raw ImageCAS-X labels may be used to define controlled evaluation truth.
10. Path construction is evaluated separately from relation identity.

## Primary operating-point comparison

Primary table should prioritize the validation-frozen safety operating point rather than AUROC ranking:

| method | coverage | repair-needed exact | false structural repair | false among accepted | exact among accepted |
|---|---:|---:|---:|---:|---:|

Secondary:

- AUROC / AUPRC for pair/relation evidence;
- Dice / clDice only for segmentation-stage methods;
- path success conditional on correct endpoint relation;
- LAD / LCX / high-degree-junction strata.

## Statistical unit

The patient is the primary independent resampling unit.

Do not compute confidence intervals by treating dozens of candidate relations from the same heart as independent patients.

Use patient-cluster bootstrap for final uncertainty.
