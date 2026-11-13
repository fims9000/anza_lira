# V25 post-V24 exploratory component-first link candidates (13 Nov 2026)

## Design motivation and status

V23 on genuine F3 images had 29/44 annotated anchor pairs already connected. Six remaining paths had both anchors detected but disconnected; just 1/6 had a candidate connecting the relevant components. V24 extended terminal radius to 68 px and added virtual interior ports but still yielded only 1/6 directly accessible pairs, and 30/44 connected under both new selectors with a modest increase in explicitly false ORANGE pixels.

**V25 is designed after all these results have been inspected. It is post hoc exploratory; the same 40 expert slices cannot provide new untouched confirmatory significance.**

## Frozen method

- Identical HGB predictor, crowd-only training sections 340–392, crowd-only CAL 311–327, exact threshold 0.35; expert masks unavailable until model decisions have completed.
- From binary image-derived dense segmentation, find 8-connected predicted components. For each component with ≥12 pixels and principal-axis eigenvalue ratio ≥2.5, place at most **two regional ports** at extrema of its first principal axis; outward tangents are the PCA axis signs. No oracle line positions, endpoints or labels.
- Compare `region_only` (PCA regional ports) vs `region_hybrid` (existing trace-terminal ports plus PCA regional ports). For each predicted port, propose at most 12 other-component endpoints within 3–96 px with directed tangent error ≤85 degrees. Do not propose already-connected component pairs.
- Score all proposed bridges with the V23 model-derived image support `score=−geometry+1.3×mean P(bridge)`. Retain fixed acceptance gate −0.3 and at most 120 links. Require one-to-one port matching, no induced component-level cycles and no more than **2 new links per originally predicted connected component**. Failed path search or failed gate produces NO_LINK.
- Path is deterministic Dijkstra in a 6px corridor of *predicted model probability*, never expert labels.
- Compare with unchanged V23 raw, geometry, context baselines **on exactly the same full 40 real expert sections**. Record blue-pixel tolerance recall, explicit-ORANGE false predicted fraction, annotation-only connectivity of two gold evaluation points, candidate upper bound and aggregate counts with section bootstrap.
- Do not claim false-link physical-geology identity, original neural ANZA/LIRA checkpoint performance, or external survey generalization. Results cannot be elevated to the existing manuscript as independent evidence solely because a metric improves.

## Interpretation

Candidate upper bound is an *evaluation-time* diagnostic using known expert annotation paths, and cannot be fed back to infer candidates. PCA regional port selection is image-only. Improvement requires more connected expert-positive anchor pairs *and* acceptable additional explicit negatives; increased candidates alone is not enough.
