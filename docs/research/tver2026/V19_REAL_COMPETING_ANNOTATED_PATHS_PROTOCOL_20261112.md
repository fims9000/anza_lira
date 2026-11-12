# V19 prospective exploratory audit: competing annotated paths on real CRACKS

Status: protocol fixed **before V19 real-archive execution**. V17 and V18 outcomes were already viewed; V19 is a new exploratory hypothesis, **not** an untouched confirmation of ANZA-LIRA.

## Motivation

Real CRACKS v2 expert-annotation gaps were easy to interpolate: V17 straight candidate was oracle-best in 47/47; V18 deliberately longer centerline deletions still had just 1/165 cases where straight-line precision within 2 px fell below 90%. Thus *route recovery with oracle true endpoints* does not measure the target difficulty: deciding which potentially adjacent endpoints belong to the same annotated connected track.

## Frozen analysis

1. Load all expert color masks (blue confident, green uncertain); skeletonize blue excluding a 5-pixel halo of green, then enumerate independent unbranched paths with at least 36 samples. No learning or threshold tuning.
2. For each pair of **distinct expert connected paths on one section**, compute minimum Euclidean distance between the interior portions at least 15 samples away from each endpoint, using KD-tree or exactly equivalent nearest-neighbor distance.
3. Report counts and unique sections at fixed thresholds **6, 12, 20, 32, 48, 64 px**, plus closest-pair distances and location, denominator all eligible path pairs.
4. A pair is considered a *potential synthetic ambiguous-occlusion case* only if both paths have interior points with >=15 visible samples on both sides, their interior distance <=20 px and both permit 12-sample centered artificial gaps. Do not build case labels from unobserved geological identity or claim that two separated expert components are physically distinct faults.
5. Store per-pair cases and all counts with SHA256 in GitHub Actions artifact. Every recorded label is only the identity of the **annotated connected component**, not independently confirmed geologic source.
6. If <10 suitable paired cases or <5 distinct expert sections, report failure of benchmark suitability; do not fabricate crossings or call a simple geometry filter validated.

No real ANZA/LIRA checkpoints or full prediction pipeline is involved. V19 is a dataset-suitability audit to decide whether actual CRACKS annotations can support meaningful positive/negative *synthetic* connectivity controls.