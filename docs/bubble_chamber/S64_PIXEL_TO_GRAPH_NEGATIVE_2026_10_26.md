# S64 | From authentic photo pixels to image-derived 2D path hypotheses: held-out negative study

**2026-10-26 · `research/bubble-chamber-graph-lira` · no human GT / no authorized native automatic particle joins.**

## Research question

Can the S56 geometry-only global forest perform when the **seven simulator fragment coordinates are not exposed** and polylines must be extracted from a real-photograph-backed raster first?

Previous S56/S60 measured graph recovery given seven already-known noisy polylines. S62/S63 measured an optimistic union-of-predicted-curves availability bound. Neither measured the full chained sequence `pixels -> distinct polylines -> candidate pair gate -> S56 geometry scorer -> global MILP abstain -> entire assembled three-piece synthetic path`. S64 closes that methodological gap.

## Frozen method

See `S64_FROZEN_PIXEL_TO_PATH_PROTOCOL_2026_10_26.md`, frozen prior to running TEST. 145 visualMetro **film-070** source frames, frame-disjoint 79/30/36 split. 64 source TRAIN and 64 source CAL three-piece scenes per direction, original unmodified S56 logistic geometry scorer, CAL-only negative-FPR cutoff (0.844309 for 005->006 and 0.853921 for 006->005), original margin `.04`, S59 strict pair gate. Separate 005->006 / 006->005 transfer to opposite authentic H50Z tile backgrounds, all photos belong to one physical run. TEST seeds `20266403, 20266404, 20266413, 20266414`, 24 scenes each, **96 new held-out synthetic scenes**.

All 96 images contain two artificially drawn 3-piece traces plus one synthetic decoy. Every image has four synthetic true links. Inference accepts **only image pixels**; no generator polylines, source indices or labels. `s39_original` and `junction_less` extract curves blind from identical pixels, pass through identical pair gates and frozen S56 scorer. For comparison only, `S56 oracle` receives the 7 source-provided noisy polylines from the simulator (known to be artificially privileged). Original neural ANZA architecture, pretrained networks and H50Z native labels are not touched.

After decisions, an EVALUATOR maps predicted lines one-to-one to generator proxies, requiring compatible overlap AND both fragment endpoints. An incorrectly selected endpoint is counted as NOT a recovered true link even if the predicted fragments lie near a common source curve. Union-of-lines potential from S62 is a separate optimistic quantity, not evidence of recovered identity.

## Held-out S64: totals across four seeds, 96 scenes

| Comparison arm | Selected candidate joins | Synthetic-correct joins | Not established as synthetic-correct | Completed three-piece synthetic paths | Scenes with >=1 noncorrect join |
|---|---:|---:|---:|---:|---:|
| S56 graph using supplied synthetic noisy fragments (oracle input) | 102 | **99** | 3 | **30** | 3 |
| **S64** conservative S39 pixel-extracted curves | 88 | **9** | **79** | **0** | 46 |
| **S64** S63 junction-less pixel-extracted curves | 94 | **10** | **84** | **0** | 48 |

Among S39's 79 non-correct selected pairs, **1** connected identifiable non-adjacent injected pieces, **34** included an unmatched or ambiguous extracted piece, and **44** used an endpoint inconsistent with the known *artificial* joining port. The corresponding counts for junction-less were **1 / 38 / 45**. Synthetic-only positive recall = 9/384 (2.34%) and 10/384 (2.60%); the graph with pre-supplied noisy source pieces obtains 99/384 (25.78%). These are NOT real physical track recalls or measured false join rates.

- S39: 483 image-derived polylines across 96 scenes, of which 197 could be associated one-to-one with an injected true fragment under the strict declared rule. Junction-less: 493 candidate lines, 200 corresponding associations.
- S62-style optimistic union proxy: 83/384 S39 joins and 89/384 junction-less joins conditionally available. The actual end-to-end correct joins (9/10) are much lower; **union proxy is not a substitute for segment identity**.
- The global MILP solver reported no failures. Scene-bootstrap conditional intervals, per each fixed seed, are in the accompanying aggregated JSON; pixels and crops are not used as independent resampling units. No claim about uncertainty across independent physical camera runs can be made.
- Prespecified synthetic safety gate: **FAIL / do not promote junction-less** (+1 synthetic true join comes with +5 noncorrect joins and 0 additional whole paths).

## What this falsifies

1. AUC / recovery with oracle source segments does **not** measure full image-to-graph performance.
2. Raising image-derived fragment counts alone has no established whole-path benefit here.
3. The main limiting issues are segmentation of individual fragment extent, correct placement of endpoints/ports, and rejection of spurious/duplicated background candidates, **before** or alongside a graph relation score.
4. Neither S64 arm may identify a real photograph streak as a particular particle or automatically fill an undocumented gap.

## Reproduce

```bash
python -m pytest -q tests
python scripts/research/bubble_chamber/s64_benchmark_pixel_to_path.py \
  --visualmetro /PRIVATE/1mH2Data/S1 \
  --photo-tiles /PRIVATE/source_39_H50Z_crops \
  --out /PRIVATE/S64_RERUN_PUBLIC.json \
  --private-dir /PRIVATE/S64_PRIVATE_SYNTHETIC_PREVIEW
```

Only aggregate JSON and code are committed. Preview PNGs contain true H50Z pixels and stay private. A safe post-hoc evaluator is deliberately separated from `infer_from_pixels(gray, scorer, cutoff)` and is never imported by the inference function. 186 unit/regression tests passed at S64 completion; no manual labeling was requested or fabricated.

**Next S65:** CAL-only pixel-domain endpoint rejection / safety gate, followed by *fresh, never-reused* synthetic held-out seeds. Do not use any S64 TEST truth to fit S65.
