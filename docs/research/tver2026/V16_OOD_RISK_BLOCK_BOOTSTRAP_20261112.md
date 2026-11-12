# V16 OOD risk: paired block-bootstrap reanalysis (12 Nov 2026)

Source: previously executed `/mnt/data/tver_v15/V16_OOD_SHAPE_DECISIONS.csv` (720 decisions; 240 source scenes, 3 counterfactual worlds per scene). **Do not count the 720 worlds as independent samples.** Bootstrap resamples 48 groups of five source scenes, stratified by both geometry families, with 12,000 replications; RNG seed 20261112. No re-training, threshold edits, or additional test exposure.

| Metric | V16 | 95% block bootstrap interval |
|---|---:|---:|
| False bridge / negative scenes | 8/240 = **3.33%** | **1.25–5.42%** |
| Wrong candidate / positive scenes | 11/480 = **2.29%** | **1.04–3.75%** |
| Correct hard continuation | 69/276 = **25.00%** | **15.91–34.77%** |
| V16 minus V15 false-bridge rate | +2.92 pp | **+1.25…+5.00 pp** |
| V16 minus V15 hard recall | +13.77 pp | **+7.14…+20.71 pp** |

Detailed source-family split:

- **Triangle deformation:** V16 hard links 66/132 vs V15 26/132, V16 false bridges 4/120 vs V15 1/120, V16 wrong branches 0/240 vs V15 17/240.
- **Two-lobed deformation:** V16 hard links 3/144 vs V15 5/144, V16 false bridges 4/120 vs V15 0/120, V16 wrong branches 11/240 vs V15 1/240.

## Interpretation

The previously specified empirical risk ceilings (false bridges ≤2%, wrong branches ≤1%) are **violated on the OOD dataset**. V16 genuinely recovers more links in the triangle-shift scenario, but has near-zero hard recall and increased wrong-branch risk with two-lobed shifts. Improvement in mean hard recall does **not** demonstrate general deformation robustness or controlled risk.

This is a *post-test diagnostic reanalysis* of one synthetic generator, **not** independent preregistered confirmation, and never constitutes geological or real CRACKS validation. See V17 real-F3 adjacent-section pilot in `scripts/cracks_real_neighbor_path_v17.py` for a separate non-E2E exploratory experiment.