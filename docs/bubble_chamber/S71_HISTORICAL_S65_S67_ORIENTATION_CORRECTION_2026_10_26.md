# S71 — corrected retrospective synthetic scoring for S65, S66 and S67

**26 October 2026.** Research experiment ID `ANZA-LIRA-BC-S71-20261026`. Historical erratum only — no new synthetic TEST scenes and no physical particle ground truth. Previous S65–S67 reports are retained unchanged.

## What was wrong

Prior to S69, the evaluation-only synthetic end-port audit assumed a fixed order of points inside generator-created fragments. S56 independently reverses those arrays, so the historical evaluator marked many correct synthetic joins wrong. S69 introduced an evaluation-only order-invariant reference endpoint check. S70 demonstrated the severity on historical S64. S71 repeats the original CAL/TEST seeds of S65–S67 with the original frozen image detector, S56 geometry model and threshold, image-domain protections and solver.

**Scientific invariant:** the original scorer and the repaired scorer assess each *identical already-produced prediction*, and the historical evaluation continues to feed the frozen benchmarking code. On each graph, the number of predicted curves, source-identity associations, decoy matches, candidate pairs, proposed edges and solver status are identical in both audits. All original public CAL and TEST outputs were reproduced exactly; S67's original per-seed bootstrap intervals were also checked for equality. Only the end-port correctness and dependent synthetic path statistics change.

## Corrected TEST results, each study 96 synthetic-on-H50Z scenes

| Experiment | Arm | Correct synthetic links, old → corrected | Other synthetic-selected links, old → corrected | Complete synthetic 3-piece paths, old → corrected |
|---|---|---:|---:|---:|
| S65 | Original S39 | 15 → **52** | 88 → 51 | 1 → **13** |
| S65 | Historical CAL-selected endpoint filter | 6 → **27** | 39 → 18 | 0 → **5** |
| S66 | Original S39 | 18 → **47** | 73 → 44 | 3 → **11** |
| S66 | Strict pixel-guided extension | 16 → **51** | 91 → 56 | 1 → **12** |
| S66 | Relaxed pixel-guided extension | 19 → **55** | 93 → 57 | 1 → **12** |
| S67 | Original S39 | 8 → **44** | 78 → 42 | 2 → **11** |
| S67 | Crossing-aware continuity stitch | 8 → **43** | 72 → 37 | 2 → **11** |

These are *different, historically observed synthetic TEST scenes* for the three experiments, and must not be combined as though they were a common paired sample. Their authentic H50Z photographic backgrounds are derived from only **two photos of one physical run**, and the underlying visualMetro film-070 measured shapes are not ground truth for these pictures.

## Corrected CAL selection decisions (CRITICAL)

**S65 now changes its scientific conclusion at the calibration-selection level.** Recomputing the original threshold-selection policy from corrected CAL scores selects **NO endpoint filtering** (`None`, original S39) in BOTH directions:

- Source photo 005: historical `0.7` → corrected `None`. CAL unfiltered synthetic-correct 19; historical cutoff 0.7 corrected 8, below the fixed 60% retention requirement.
- Source photo 006: historical `0.4` → corrected `None`. CAL unfiltered synthetic-correct 16; historical cutoff 0.4 corrected 12, but the required reduction in noncorrect links cannot be achieved from only two corrected negative baseline links (maximum reduction 2, minimum requirement 3).

The 27 correct / 18 noncorrect / 5 complete S65 filter figures are a **retrospective evaluation of the historical filter**, not performance of a filter newly selected by corrected CAL. The correct CAL-only default is unfiltered S39: 52 correct, 51 other, 13 full paths on the historical TEST. The historical S65 inference and original frozen CAL JSON remain unchanged and reproducible.

**S66:** corrected CAL selects original S39 for both directions, as before. Relaxed tracing appears to add +8 correct joins and +1 complete synthetic path on its historical TEST, but also +13 noncorrect joins (55/57 vs original 47/44). Strict tracing yields 51/56 and 12 paths. Thus the former pessimistic absolute counts were wrong, yet the strict no-extra-errors deployment criterion still fails. No default change.

**S67:** corrected CAL selects original S39 for both directions, as before. On historical TEST, crossing-aware stitching has 43 correct vs 44 original, 37 noncorrect vs 42 and the same 11 complete paths. It does not raise reconstruction completeness. No default change.

The public aggregate contains per-direction breakdowns, per-seed paired 1,200-scene-bootstrap 95% intervals, the exact source SHA-256 locks and protocol checksum. The intervals describe uncertainty **conditional on generated artificial scenes**; the camera runs are not independently replicated.

## Reproduction & review

```bash
python -m pytest -q tests/test_s65_*.py tests/test_s66_*.py tests/test_s67_*.py tests/test_s69_*.py tests/test_s70_*.py tests/test_s71_*.py
python scripts/research/bubble_chamber/s71_reaudit_s65_s67_historic.py \
  --visualmetro /PRIVATE/1mH2Data/S1 \
  --photo-tiles /PRIVATE/source_39_H50Z_crops \
  --out /PRIVATE/S71_RETROSPECTIVE_MACHINE_ONLY.json
```

- Source: repository `fims9000/anza_lira`, **only** branch `research/bubble-chamber-graph-lira`.
- The original source and S69 evaluator are verified against exact frozen SHA-256s before benchmark execution. All original public S65/S66 CAL and TEST records must match bit-for-bit decoded JSON (S67 original published totals and per-seed paired CIs must match).
- The evaluator is shadowed only in the research replay process; no inference code is modified. A failure must restore the original audit callback.
- Private H50Z tiles/BMP/7z, source coordinates, generated per-scene labels, photo pixels, weights, predicted polylines and frame-level IDs do **not** enter public source or aggregates.
- **Physical particle identity is UNKNOWN; no native H50Z expert GT and no authorization for automatic 3D/physics track reconstruction.**

Next research stage: a brand-new synthetic investigation with orientation-invariant evaluator from the start, stronger *instance-aware* line separation at crossings, ideally with independent physical runs for domain validity. Do NOT use S71's known historical TEST figures to retune the previous methods.
