# V18 pre-execution design: longer real-F3 gaps and curved-path challenge

**Fixed before running V18 on CRACKS**; the prior V17 real-CRACKS result (47 nearly straight gaps, straight baseline best 47/47) was already inspected. Thus V18 is a **post-V17 exploratory redesign**, not untouched confirmatory evidence.

## Cohort generation without prospective GT tuning

- Official public CRACKS v2 (Zenodo 13926822) images and **expert** blue connected skeletons only. Sections with missing adjacent seismic image are excluded; no crowd annotation used as oracle.
- Enumerate independent unbranched expert skeleton paths; among all paths take at most the 3 **longest per section**, selected without looking at curvature. For each line create a centered missing annotation segment of **24, 36, or 48 skeleton samples**, only if >=15 visible samples remain before and after the gap. No extra tuning after seeing output.
- First/last surviving annotated points and their *visible-only* 6-point tangent estimates are deliberately oracle-supplied input; the hidden expert segment is never used for path construction or inference scores. Original seismic image intensities **remain visible**; this is an annotation-gap test, not missing-seismic-pixel reconstruction or E2E.
- All eligible cases are reported with section, gap length, straight-path mean distance and 2px precision, Hermite baseline, candidate scores and decisions. To isolate the V17 limitation, an explicitly **ground-truth-conditioned challenge subgroup** is pre-defined as straight precision within 2 pixels < 0.90. Its number and fraction among all eligible gaps must be shown; it is not an unbiased sample of natural faults.
- Evaluate five **fixed** Hermite-normal offsets {-4,-2,0,+2,+4} pixels; a central-only, genuine-neighbor-only, central+neighbors and central+two-far-shuffled control choose their candidate by a frozen horizontal-gradient proxy, no labels or threshold fitting. Deterministic Hermite and straight are standalone baselines.
- Primary comparison on ALL eligible cases: mean candidate-to-hidden-expert distance in pixels. Secondary: candidate precision within 2 px; predefined challenge-subgroup metrics; paired section-group bootstrap, 10,000 resamples, seed 20261112; compare methods to straight/Hermite and shuffled controls. Correlated gaps in the same section must not be treated as independent.
- FAIL/admit limitations if eligible challenge cases <10 or fewer than 5 sections; if reported benefit vs geometry includes zero within cluster CI; or if shuffled controls perform equally well. No claim of better real fault segmentation, candidate extraction, natural negative-pair rejection, or complete ANZA→SBPP→LIRA→path E2E.

## Claim boundary

This is an **exploratory conditional path-selection** study with intentionally oracle-supplied endpoints and no valid negative pair identities. The historical ANZA/LIRA checkpoints remain unavailable. Neighboring F3 images are correlated views from one geological volume, not independent external domains.
