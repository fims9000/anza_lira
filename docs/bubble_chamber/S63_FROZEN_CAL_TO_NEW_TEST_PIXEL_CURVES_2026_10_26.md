# S63 — frozen image-only candidate extraction with calibration-only selection

2026-10-26; branch `research/bubble-chamber-graph-lira`. Written BEFORE running S63 calibration and fresh test. No expert GT exists for the native H50Z tracks.

## Research objective

Can pixel-derived polyline extraction improve S62's recoverable artificial three-piece links (73/384 over 96 S62 scenes) without a material increase in artificial decoy support, without access to generator geometry at inference?

## Data isolation

Source visualMetro film 070 audited 145 complete source frames and S37 whole-frame deterministic split train 79 / CAL 30 / TEST 36. CAL template curves exclusively derive from CAL frames; only TEST templates derive from TEST frames. Both source photos `H50Z_030_01_005` and `...006` are from the same real physical run and are unannotated, so this does not validate unseen photo-run generalization. Only two 192x192 target-photo background banks derived from 39 private crops, no raw photographs in Git.

Predeclared S63 scene seeds: CAL source `005`: 20266301,20266302; CAL source `006`: 20266311,20266312; each 12 scenes. Fresh TEST source `005`: 20266303,20266304; source `006`: 20266313,20266314; 24 scenes each. The `SCENARIOS` cycle is unchanged. Artificial scene generation and evaluation used as in frozen S62. Ground-truth polylines are available ONLY after image-only prediction for evaluation. No source-derived coordinates, truth identities, or simulator internals passed into extraction.

## Frozen comparisons and fixed budgets

- `s39_original`: exactly existing `extract_ridges(ridge_evidence(gray),max_candidates=32)`.
- `junction_less`: same S39 fixed 3.5% ridge mask; skeleton with only degree >=5 junction pixels excised, without a dilation. All other selection identical.
- `darkness_skeleton`: top 3.5% raw local-contrast score, same 3x3 closing/skeleton as S39 and degree >=4 junction points plus dilation.
- `ridge_dark_fusion`: fixed ridge/contrast blend after each map's own 99th-percentile normalization, weights 0.65/0.35; select top 3.5% pixels; cut only degree >=5 junction points without dilation.

All return up to 32 image-derived curves. No increase in per-image candidate count cap and no neural pretraining. Score-based sort and `_trace_component` length criteria same as S39. Tests must verify predictions invariant to any evaluation-only identity fields.

## CAL-only selection, frozen before seeing TEST

Pool all 48 CAL scenes per arm; compare recoverable synthetic true joins and synthetic decoy-supported fragment count. Variant is eligible only if its CAL true joins improve by at least `max(5,ceil(0.15*baseline_CAL_true_joins))` relative to original AND its CAL artificial decoy-supported count increases by at most 2. If multiple qualify choose fewest decoy supports, then most true links, then lexical method name. If none qualify, retain original. TEST runs ALL FOUR arms for scientific transparency, but no post-TEST switching.

S62 proxy acceptance: 40% generator-polyline pixels within 4px of some extracted image line, with both fragment endpoints supported. This is an upper bound on linkable synthetic fragments, not end-to-end graph correctness. Record full-path possible, synthetic decoy, candidate count and scene-level bootstrap. Never present as actual particle detection accuracy, physics certainty, or evidence of actual restored links.
