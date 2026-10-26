# S76 — preregistered separation of shape association from endpoint error (2026-10-26)

**Status:** frozen BEFORE the S76 generator is run. This is a diagnostic experiment, not a new detector, classifier, threshold selection or license to identify real particles. Prior S75 and S75C TEST results are already known and are only motivation; S76 uses independent **synthetic scene draws** with newly reserved seeds, not independent H50Z physical runs.

## Scientific question

How many pixel-derived lines follow an injected synthetic fragment's **shape** but fail to match the source's **two endpoints**, and how often do those candidate pieces participate in a proposed graph join? The S69 evaluator requires endpoint proximity before it assigns synthetic identities, causing S75C's `selected_synthetic_unmatched_or_ambiguous` category to confound source-shape mismatch with truncated/extended endpoint geometry. S76 separates these two tests **without changing any S69 official synthetic correctness score**.

## Frozen evaluation-only definitions

The `match_by_shape` function accepts *post-inference* image-predicted polylines and seven synthetic ground-truth generator fragments; neither is exposed to the detector or graph solver. For each predicted/source line pair:

- Require at least 65% of the **predicted curve samples** within 3.5 px of the *single* source curve, and at least 50% of source samples within 4.5 px of that *same predicted curve*. Match is **endpoint-blind**; no endpoint position, tangent direction or chosen graph port may influence this assignment.
- Perform a Hungarian one-to-one matching with a no-match option. When the top two candidate identities differ in joint coverage by ≤0.10, mark the predicted line ambiguous rather than assert identity. Two predicted fragments cannot both be assigned the same source fragment; duplicates are diagnosed as extras, not automatically merged.
- **Only after** shape identities are frozen, calculate orientation-invariant minimum `max(endpoint_distance_0, endpoint_distance_1)` and number of matched source fragments having both endpoints within 8 px. Record shape-matched-but-endpoint-far counts and distances by direction; never reassign shape identity based on endpoint distance.
- Evaluate every already-selected graph join by separating `(a) shape-unmatched/ambiguous source`, `(b) shape-matched non-neighbor identity`, `(c) correct synthetic neighbor IDs but selected port wrong, where the opposite predicted endpoint is within 8 px`, `(d) correct IDs but one/both source gap endpoints not within 8 px of either predicted endpoint`, `(e) synthetic-correct neighbor IDs and both selected gap endpoints within 8 px`. If selected gap port is farther but the opposite port also farther, classify **endpoint unlocalized**, not wrong port choice. Synthetic neighbor source gap ports come from the S69 orientation-invariant `reference_facing_ports` helper; no assumptions about stored point order.
- A synthetic 3-piece path is complete only with two correct selected bonds and no conflicting selected connection touching any of its three predicted nodes. The **original S69 official evaluator is called on the exact same inference object** and must retain its own unchanged output; differences are diagnostic, not substituted into original S75 published accuracy.

## Frozen image benchmark

- Source measured visualMetro 070 whole-frame split: 79/30/36; source images H50Z `005` and `006` from **one physics run**, 39 private 512 px crops, source frames are NOT pixel ground truth for H50Z.
- S72 fixed direction-specific graph variants from `S72_PUBLIC_FROZEN_CAL_2026_10_26.json` (SHA-256 **1ed7e58667aaab1a7d5ba91252db64b38bf8d3da517396b9e18e9e6bb484a952**). S56 original source-photo TRAIN/CAL scorer and frozen cutoff, no TEST access for training.
- Fresh 96 synthetic scenes: target 006 from source 005 seeds **20267603/20267604**; target 005 from source 006 seeds **20267613/20267614**, 24 scenes each, `SCENARIOS` cycle. Their source track templates come from original 36 held-out visualMetro TEST frames, but scene draws are new. No S74/S75/S75C TEST scene seeds are reused.
- Three fixed pixel conditions per exact same scene: `clean`, `blur_sigma_11`, `blur_sigma_16`; three predeclared S75 preprocessors: `identity`, `coherent_dog`, `local_blackhat`. All 9 arms are only diagnostics. No training or threshold selection may use S76 results.
- Summary metrics: shape-matched six true synthetic fragments, decoy, endpoint-near vs shape-only endpoint-far matches, all join error categories, original S69 complete/correct links, S76 shape-only upper potential, per-image candidate counts, scene-bootstrap paired contrasts. Each scene contributes once per condition/arm; bootstrap scenes, never individual sampled track pixels.
- Fixed 1200 paired bootstrap draws, seed **20267699** offset by scenario index/direction. Report differences as exploratory conditional-to-these-scene statistics, **not** physical-run uncertainty.

## Privacy and final gate

No raw H50Z photo pixels, extracted visualMetro coordinate files, private per-scene predicted geometry, synthetic source polylines or trained weights may enter GitHub. Only source code, tests, frozen protocol, safe aggregate JSON and experiment ID `ANZA-LIRA-BC-S76-20261026` may be committed. Native physical track GT count remains 0. **No model is promoted.** Future method proposals must be evaluated on further fresh synthetic scenes and, eventually, independently labeled physics runs.
