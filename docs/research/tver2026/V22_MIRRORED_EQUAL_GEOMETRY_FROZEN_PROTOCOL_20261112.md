# V22 preregistered mirrored-endpoint diagnostic on genuine CRACKS F3 seismic images

Scientific scope: **synthetic matched-geometry counterfactuals only**. Designed *after* seeing V21 real results, hence exploratory even if test metrics are computed just once. It is NOT validation of true geological negative links, ANZA segmentation or learned LIRA E2E.

## Pre-fixed case source
- Use the existing V21 immutable per-case CSV from successful run 43297677785; require exact SHA256 `c30b1279e8e114d8c11c656f7f20ebfeacc237ca163a304f19b41a9389b7d237` and use **only positive V21 rows**.
- Reconstruct the source expert connected trace and visible tangent from V19 original interior closest-approach index and the V21 fixed `gap` and `shift`, without changing the original case selection. Verify reconstructed e0/true endpoint exactly agrees with pinned V21 row.
- Set true endpoint equal to original expert-connected continuation. Construct a *synthetic fake endpoint* as the exact reflection of the true endpoint across the line through source visible endpoint parallel to its 6-sample visible tangent; reflect the true target tangent by the same isometry. This gives identical source-to-target distance, heading cosine and absolute tangent compatibility, *by construction*. No pixels from hidden expert centerline are used for scoring.
- Exclude a case iff the fake endpoint lies outside the seismic image or the true-to-fake target separation is <5 px. Report eligibility among **all 58** positive V21 cases and number of source sections. The current V21 CSV indicates about 40 cases meet the minimum-separation-only criterion, but exact V22 eligibility must be counted.
- Randomize positive candidate presentation index deterministically using SHA256(section, V21 pair, direction, gap, `V22`), not source label parity.

## Frozen models
- Both geometrical metrics are intentionally tied; report maximum numerical differences to verify. Geometry-only baseline selects the first candidate, which should be approximately chance after order randomization; **do not optimize tie-breaking**.
- Images are actual central F3 image, genuine immediate neighboring sections -1/+1, and fixed distant-section sham controls (+107 and +211 IDs in cyclic list), already used in V17/V21. All sampling at original pixels, no training and no parameter fit.
- Freeze an absolute horizontal Sobel-gradient proxy (Gaussian sigma 1, normalized by image P95 as in V17). Sample at 48 equally spaced points along each straight bridge excluding 8% at each end. Policies: maximum central mean, maximum genuine neighbor mean, maximum combined central+neighbor mean, and maximum far mean. Also evaluate maximum similarity of the candidate's mean central gradient support to mean gradient on 12 visible source-trace pixels before e0. No scalar mixing, risk threshold, abstention or parameter search.
- Report accuracy per policy and section-level 10,000 bootstrap 95% percentile intervals vs 50% tie baseline and adjacent-minus-far, seed 20261112. All individual score values and case labels saved for independent re-count. Exact binomial intervals are **descriptive**; per-case independence is false, so cluster CI is preferred.
- Primary claim only: whether the genuine image signal contains information to distinguish the annotated original continuation from an *artificial reflection* when geometry is exactly equal. This **does not prove** that the fake endpoint is a real observed fault or that ANZA/LIRA resolves naturally branching faults. If neighboring advantage is absent or uncertain, report the null/negative result and stop retuning this benchmark.

## Leakage and publication boundaries
- Expert annotation determines source and true target **only in case construction and metric evaluation**; choices depend on visible endpoint/tangent and raw seismic pixel gradients. True endpoint is of course one provided candidate; the hidden connected path and correct-index label never enter policy scoring.
- All selected sections originate from one F3 volume and many examples share sections. Do not use V22 to claim external-domain generalization or control of natural false-bridge rate.
