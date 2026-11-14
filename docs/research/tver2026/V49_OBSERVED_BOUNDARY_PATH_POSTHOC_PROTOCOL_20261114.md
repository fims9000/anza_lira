# V49 — observed-boundary anchor path tracing; path-stage experiment after failed V47

**Frozen on 14 November 2026 before V49 path inference.** Source is original real Thebe, 120 artificially interrupted REL-CAL LINK gaps (VAL1 sections 40–99). No natural instance linkage labels and no external TEST7 access. Rel-CAL was already repeatedly viewed, so this is exploratory, **not** a confirmatory test.

## Problem with prior decoder
The coherent V46 cascade restores 75/120 true candidates and ranks the correct pair top1 in 45/120. In the actual V47 geometric test, original straight path meets 2px-tolerance hidden-centerline precision/recall>=0.8 in only 2/120, original maximin in 0/120; neither succeeds under the acceptance gate. Pair endpoint correctness allows 8px deviations; the hidden path metric allows only 2px, which might explain some mismatch. Previous experiments do not isolate whether the particular route decoder is at fault.

## Frozen alternatives, no oracle use for path generation
- Keep exactly V46 original trained ANZA checkpoint, label-blind q70 adaptive dense fallback and 12D paired-difference ranker trained on the SAME augmented TRAIN1. For acceptance, keep the same V45-style source-fold OOF scene gate; do not refit it to path results.
- Measure original straight and original maximin route from V47 **without alteration**.
- New candidate decoder BOUNDARY-STRAIGHT: using ONLY post-intervention predicted probability outside the known intervention rectangle, find one deterministic strong-probability anchor near its boundary on each side of the *predicted* source→target pair (max 16px from the line's boundary crossing and 24px from that candidate endpoint), with source/target distinct. Connect source→anchor→anchor→target; metric uses path pixels INSIDE hidden rectangle.
- New decoder BOUNDARY-HERMITE: same two observed anchors; estimate local line tangent from the principal axis of high predicted probability outside the intervention mask within 10px of each anchor (no hidden centerline), orient towards the opposite anchor, interpolate cubic Hermite with tangent length 0.30 of anchor distance and rasterize densely.
- Fixed anchor scoring and thresholds are defined in the source before real case evaluation. No CAL-selected tuning, no supervised path-data fitting, no using expert centerline, endpoints, or group IDs as algorithm input.
- Evaluate all candidate-correct top1 cases exactly as V47, with original hidden-centerline path precision+recall >=.8 at 2px tolerance. Report each method's top1-conditional and all-120 success, and correct accepted paths under the SAME V46 frozen gate risk protocol. No method may be described as outperforming if only a posthoc number is better.
- A source-level case CSV must record whether anchor search changed original endpoints and all successes/failures, with immutable original V47 source/control digest, without raw seismic source arrays in artifact.

## Stop/go
A geometric improvement on previously inspected Thebe REL_CAL is **only** a new hypothesis. The protocol-locked VAL1 30..39 mini-holdout was opened in V48 and must NOT be used to choose V49 path hyperparameters or claim a second independent test. A true field-independent success still requires a new annotated volume, natural identity negatives, and a frozen risk-controlled end-to-end test.