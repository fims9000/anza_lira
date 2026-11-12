# V20 pre-execution protocol: real-background controlled competing endpoints

Designed **after seeing V17–V19**, before opening V20 outcomes. Status: exploratory, NOT a fully independent confirmatory test.

## Fixed source and cohort

- Exact V19 real expert-annotation component-pair CSV: SHA256 c9a0b0a31f93c9731c57dca6536af87227bc22fc96c0192afd80c0c576198482.
- Use each distinct pair of expert-connected blue skeletons with interior minimum separation <=20 px, from V19, both directions (A source, B decoy and vice versa). Require genuine central and ±1 F3 seismic images with matching image dimensions.
- For source path of length N, with V19 nearest interior index i, place one synthetic 12-sample annotation gap at a = clip(i-6,15,N-12-15), b=a+12. Visible source endpoint = path[a-1]; positive target = path[b]; decoy endpoint = closest interior location on *other* expert-connected path. The labels belong to the **annotation identity defined by construction**, not verified geological identity. No hidden gap point enters score features.
- Use same fixed case list for all methods, report exclusions, all denominator counts and unique sections. Multiple directed cases from one annotated pair/section are **correlated**.

## Fixed models (no new trainable parameters)

Observed features from both candidate endpoints and a visible-only source tangent: Euclidean gap length d, forward heading cosine c, absolute downstream tangent cosine k. For a source 12-point gap, fixed geometry risk:
    g = (d/13 - 1)^2 + 0.6 * (1 - c) + 0.2 * (1 - k).
Smaller is better. Compare:
1. nearest endpoint (minimum d);
2. heading-only (maximum c);
3. geometry risk g;
4. geometry minus 0.10 times normalized central-slice horizontal gradient sampled along the straight candidate bridge;
5. geometry minus 0.10 times mean of two real adjacent-slice gradient supports;
6. geometry minus 0.10 times mean of two distant-section sham gradient supports.

Every rule chooses one of the TWO candidates, never NO_LINK. No thresholds, learning, model checkpoints, held-out optimization, or per-case oracle choice. Tie-break deterministically by candidate index. The same geometry rules apply to true and decoy candidates.

## Frozen outcomes and interpretation

- Primary: correct annotated-component continuation choice among two endpoints; error proportion (wrong branch). Report denominator of all directed cases, unique annotated pairs and sections.
- Secondary: hard distance-ambiguous subgroup where the decoy's distance from source is <=1.2 times the true candidate distance. This subgroup is explicitly defined using ground truth and cannot be treated as an unbiased test population; report fraction and group count.
- Paired bootstrap, resampling entire expert sections 10,000 times with seed 20261112; contrast geometry+adjacent vs geometry and geometry+far sham; report 95% percentile interval without multiple-comparison adjustment.
- If fewer than 20 directed cases or 5 sections, report insufficient. If geometry+adjacent fails to outperform plain geometry or sham, preserve this negative result. Winning synthetic annotation pair identity would not prove geological instance identity or full ANZA→SBPP→LIRA segmentation and candidate generation.
- V19 spatial sections are from one F3 volume and are not external validation. All choices are conditional on expert-provided visible endpoints and candidate geometry; no natural negative-link ground truth or abstention risk is tested.
