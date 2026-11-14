# Tver V24 — executed, audited, matched-marginal switched-neighbor NO_LINK test
**14 November 2026. Experimental source is a synthetic 2.5D line renderer, NOT observed seismic fault identities and NOT full trained historical ANZA–LIRA.**

Original V23 [PR #28](https://github.com/fims9000/anza_lira/pull/28) found 312/320 correct positive OOD links with 0/160 false acceptance on a generator where negative candidates have unusually large z-displacements. To expose this potential shortcut, V24 [PR #30](https://github.com/fims9000/anza_lira/pull/30) predeclared a *new* generator of NO_LINK scenes whose two real-neighbor marginals are **each pixel-for-pixel identical to an alternative positive world**. The complete central 2D view and supplied endpoint geometry are also bitwise identical across all three source worlds. This synthetic NO_LINK corresponds to **inconsistent source/target membership between adjacent slices**, not a natural geological fault negative.

## Controlled protocol and comparison
- [Frozen pre-test specification](./V24_SWITCHED_NEIGHBOR_HARD_NOLINK_FROZEN_PROTOCOL_20261114.md) fixed independent seed base 202673010, 160 nuisance groups ×3 worlds =480 TEST cases. TRAIN and CAL were original frozen V23 200 and 100 source bases. V24 TEST was not rendered until all original TRAIN HGB fits and original/new CAL thresholds were frozen. No TEST-based threshold tuning or population selection.
- For each NO_LINK world, **z−1 is exactly a positive LINK A neighbor, while z+1 is exactly positive LINK B neighbor** (or reverse by base parity); a policy examining one neighbor cannot identify the negative from that neighbor alone. The two positive worlds remain coherent across both neighbors.
- Candidate source/endpoint positions are still **oracle-supplied**. The original V23 mean-neighbor source-conditioned matcher is unchanged. The new **V24 consensus** evaluates per-candidate agreement of the source strand with each candidate on both neighbors and takes the **minimum**, not the mean:
\[
S_j=\min_{z\in\{-1,+1\}}\{\min(E^{source}_z,E^{candidate}_{j,z})\}.
\]
- A link j is accepted only if \(S_j\) exceeds a CAL-selected absolute evidence threshold and exceeds the other candidate by a CAL-selected margin; otherwise output **NO_LINK**. The risk CAL objective is empirical false acceptance ≤2% of CAL negatives and wrong branch ≤1% of CAL positives; not a guaranteed population risk.
- Comparisons: old mean-neighbor V23 matcher, two independent one-neighbor verifiers, central pixel method, learned 2D HGB and learned 2.5D HGB trained on original V23 TRAIN with **the same observed 2.5D feature family**, with fixed CAL gates. The 2.5D HGB was **not retrained on switched V24 NO_LINKs**; this is an OOD inductive-bias comparison, *not an equal-new-negatives training comparison*.

## Executed original V24 TEST

| Policy | Correct positive accepted /320 | Wrong branch accepted /320 | False NO_LINK accepted /160 | 3-way total accuracy |
|---|---:|---:|---:|---:|
| Original V23 source-conditioned mean | 311 | 0 | **83** | 80.83% |
| Same-features trained HGB 2.5D (V23 TRAIN) | 286 | 0 | **40** | 84.58% |
| V24 two-neighbor **minimum-consensus** | **312** | **0** | **1** | **98.125%** |
| Left-neighbor-only | 291 | 0 | **148** | 63.12% |
| Right-neighbor-only | 312 | 0 | **155** | 66.04% |
| Central image | 23 | 23 | 23 | exactly 33.33% |
| Learned central 2D HGB | 0 | 0 | 0 | exactly 33.33% |

New consensus minus original mean 3-way accuracy **+17.292 pp**, paired nuisance-base bootstrap 95% percentile CI **[+14.792,+19.792] pp**. Versus same-feature trained 2.5D HGB (trained on *earlier* negatives) **+13.542 pp**, 95% nuisance-base CI **[+10.833,+16.250] pp**, 3000 replicates and **160** independent nuisance bases rather than 480 dependent worlds.

The V24 code's test-risk condition was met (312/320 correct, 1/160 false, 0/320 wrong; bootstrap lower-bound delta vs HGB >0). But a one-sided 95% Clopper–Pearson **upper bound** for 1/160 false accepted is **2.93%** *even under independent negative Bernoulli examples*; section/patient-like nuisance dependence would require a more cautious group analysis. Empirical ≤2% is not a universal guarantee. The single registered false acceptance is base **202673144**, synthetic NO_LINK accepted as candidate 1; retain the failure figure without repair.

## Independent audit and numerical repeat
- Successful first real execution [CI 43592395849](https://github.com/fims9000/anza_lira/actions/runs/43592395849).
- Successful independent per-case audit + independent repeat [CI 43592538043](https://github.com/fims9000/anza_lira/actions/runs/43592538043): \`V24_INDEPENDENT_MATCHED_MARGINAL_CASE_AUDIT_PASS\`. Checks all 480 rows, three-world 2D equality, exact one-neighbor positive marginals from the opposite true worlds, per-method TEST predictions from original CAL thresholds, 160-group paired CIs and strict positive/negative rates.
- Cross-run comparison of these two GitHub CI executions: **0 changed decisions among all 480 cases for all seven methods**, and observed score drift **0** in case-level CSV. This does **not** imply future CPU/GPU or library-version invariance; V23's naive neighbor baseline displayed significant machine-precision CAL threshold sensitivity (posthoc disclosed in [V23 report](./V23_25D_SYNTHETIC_EXECUTED_RESULTS_20261114.md)).
- Runner-local/CI aggregate artifacts carry complete synthetic case labels and decisions, CAL lock and code commit. No medical/private data.

## Scientific significance without overclaiming
**Supported only on this synthetic generator:** the original source-conditioned V23 *mean* failed badly (83/160 fake bridges) when NO_LINK examples mimicked true neighboring-slice marginals. A new **cross-slice global-consistency constraint** substantially reduced overconnection with essentially unchanged positive-link recovery. Naively pooling a candidate's per-slice support is not enough when support is inconsistent across slices. This is a useful falsifiable architecture lesson for a future LIRA verification stage.

**Not supported:** the historical ANZA→SBPP→LIRA pipeline being better than published 3D CNN/UNet/Transformer *segmentation* baselines; those models were **not** run on the V24 LINK/NO_LINK task. All endpoints and candidate branches are generator-supplied, whereas a complete pipeline must *detect* them. Real F3 CRACKS mask studies V21/V22 provide distinct evidence about observed pixels, not ground-truth physical fault identity. The GAN-like "switched neighbor" NO_LINK is synthetic and intentionally rewards a two-neighbor consistency bias; genuine fault connectivity can split/change with depth. Very strong cross-domain generalization is not established.

**Next falsification needed:** train a capable same-input 2.5D HGB and CNN **also on the new switched-negative regime**, not only on the easy V23 negatives, while retaining a completely disjoint TEST. Otherwise the +13.5pp contrast may reflect training-distribution mismatch rather than intrinsic superiority. For publication, phrase "on unseen switched synthetic negatives, our nontrainable physical consistency prior outperformed a learner fitted to the older negative generator," not "ANZA–LIRA beats state of the art."
