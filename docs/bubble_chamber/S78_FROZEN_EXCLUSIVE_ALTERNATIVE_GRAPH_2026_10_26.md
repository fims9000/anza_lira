# S78: Preregistered non-destructive competing 2D endpoint hypotheses

**26 Oct 2026 — FROZEN BEFORE S78 CAL and independent TEST.** Scientific question: can S77's short-gap pixel evidence propose **alternative endpoint connections without deleting or merging any original detected line**, with global mutual exclusivity between incompatible hypotheses and graph-cycle/port constraints? The aim is reversible human-inspectable *machine* alternatives, **not** physical track identity. This project has 0 native verified H50Z particle links.

## Fixed inference arms

Input is exclusively 192x192 uint8 photo-backed pixels, the frozen per-source S56 geometry-only scorer and cutoff, and the frozen S72 per-direction image curve extractor. Underlying unchanged S68/S69/S72/S77 modules are read-only. **Source fragment identities and injected polylines enter only post-hoc S69 evaluation**.

- `baseline`: exactly original S72 graph inference.
- `alternative_strict`: preserve the same baseline polylines and all original S56 graph edge candidates; add S77 short-gap-strict pixel-supported endpoint hypothesis **only when** the undirected pair passed the frozen S59 strict_s39 gate AND the proposed endpoint ports differ from the original scored edge's ports. If the proposed pair/ports are identical, retain the baseline version (do not double-count evidence).
- `alternative_relaxed`: same rule with S77 short-gap-relaxed pixel proposals.
- New edge ranking score is a **heuristic, not a calibrated probability**: `min(0.999, max(geometry_score + 0.06, 0.68 + 0.24*pixel_gap_score))`, 0 <= pixel_gap_score <= 1; if proposed S77 pixel score is outside [0,1] reject it. Baseline S56 edge score unchanged. Only proposals with score above frozen S56 CAL cutoff can enter optimization; max 32 polylines and max 32 added gap alternatives per image; no new pixels, no source-proxy geometry.
- Each alternative has a provenance record pointing to two original machine polyline indices, the two endpoint ports, and the pixel-evidence arm; no sensitive image hashes or pseudo-physical IDs. Original polylines remain unchanged byte-for-byte and in original order.
- The new bounded MILP solver chooses a weighted **forest of competing edges**. Constraints: at most one selected edge per physical machine-polyline port, at most one alternative per undirected machine pair (even if ports differ), no graph cycles. Perform counterfactual stability margin >=0.04 for each chosen edge. Solver failure or timeout => **empty proposed joins**. If no eligible alternative, return the exact untouched S72 baseline graph decision.
- Global forest assumptions are toy 2D synthetic graph restrictions only. They need not hold for real physical crossings or loops.

## Frozen data and experimental IDs

- visualMetro film 070 operator geometry 145 complete frames, whole-frame S37 split 79 TRAIN /30 CAL/36 TEST; **not** H50Z true particle identities.
- 39 authentic 512px H50Z crops (21 from photo 005, 18 from photo 006) from **one** physical camera run. All scenes are artificial line insertions over genuine photo backgrounds, never real expert GT.
- S72 original source-CAL SHA-256 `1ed7e58667aaab1a7d5ba91252db64b38bf8d3da517396b9e18e9e6bb484a952`, S37 source-frame split SHA-256 `448f21ca3a709010ed972a596ed046cadff0643112905fd1d029c91f8b14faef`. The original S56 source model refit on TRAIN as before, frozen cutoff verified against S72 CAL.
- Fresh **S78 CAL** 16 synthetic scenes per source photo (seeds 20267801 and 20267811), both `clean` and `blur_sigma_11`. Frozen CAL gate: > = +2 correct artificial joins at blur 1.1, <= baseline other-join count under clean AND blur, >= baseline full chains both conditions, <= 1 lost correct join clean, solver failure count == 0. If none pass, retain `baseline` for that direction. Tie-break fewer blurred other joins, more blurred correct, then strict before relaxed.
- Fresh **S78 TEST**, 24 synthetic scenes per seed, 4 seeds `20267803`, `20267804`, `20267813`, `20267814` = 96 independent synthetic scenes, each evaluated under `clean`, `blur_sigma_11`, `blur_sigma_16`. Both directions are reported separately; results are not a third independent physical run. All arms reported even if CAL rejected them. Paired whole-scene bootstrap 1200 replicates with fixed seed from 20267890.
- Stable experiment ID: `ANZA-LIRA-BC-S78-20261026`. Public outputs: protocol/code hashes, per-direction CAL decisions, per-seed aggregates and bootstrap intervals, counts of original vs alternative proposals, rejected conflicts, and native labels count 0. Private image bytes, per-scene line coordinates, visualMetro source frame IDs, weights and individual photo hashes must NOT be committed.

## Decision

Promote neither experimental mode to default unless its frozen CAL gate passed **and** unseen TEST exhibits more confirmed synthetic joins, no higher synthetic noncorrect joins/any-error scenes, no fewer full synthetic chains in both photo directions and no failures. Passing synthetic gates still does **not** authorize real H50Z automatic physical joining. Preserve all former reports as history; no silent retrospective S77 tuning.
