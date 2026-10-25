# S30: candidate identity and ambiguity gating (limited milestone)

S29 established that curvature-constrained 2D photo evidence can recover manually injected test lines on H50Z backgrounds, but lacks genuine track labels. A physically incorrect path and a valid path can share local contrast/curvature. S30 adds an **unverified-hypothesis gate**, not a new trained AZConv model.

The gate operates on the unchanged S28 `candidate_paths` representation: a centerline, an uncalibrated score and a fractional local signal. It identifies redundant hypotheses via bidirectional sample overlap (both paths must cover at least 87% of each other within 3px); it flags distinct competing branches within an 8% relative-score gap for human review. A lone weak path is `LOW_EVIDENCE_REVIEW`. A seemingly unambiguous path is still `SINGLE_PROVISIONAL_REVIEW`, never an automatic physics link.

All paths, including duplicates, remain available for review with their original identifiers. No synthetic or model output is transformed into genuine expert decisions. Strict input checks prevent malformed polylines and invalid scores.

The deterministic S30 contract cases test path deduplication, fork abstention, weak-evidence abstention and a lone hypothesis. **These are mathematical demonstrations, not held-out physical tests**. The heuristics and thresholds are not statistically calibrated and cannot be reported as lowering false joins on real tracks.

**Next scientific requirement:** test actual competing S29 paths against human-confirmed line and same/different relation annotations, using photo-separated or run-separated validation. When expert JSON is supplied, measure path identity precision, track continuity, missed branches and false physics links. Until that evidence exists, all suggestions remain review-only.
