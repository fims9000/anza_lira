# S77 — pixel-supported short-gap instance repair did not improve end-to-end recognition

**Date:** 26 October 2026  
**Experiment:** `ANZA-LIRA-BC-S77-20261026`  
**Branch:** `research/bubble-chamber-graph-lira`  
**Decision:** no deployment/promotion; the frozen S72 source arm stays the default. This is a controlled synthetic-on-real-background study, not independently labeled physics.

## Motivation and method

The S76 diagnostic separated two errors that were conflated earlier: an image-derived polyline can resemble an injected line's *shape* while its endpoints are not localized, and the graph can propose an incorrect connection even if the line is detected. S77 asked whether two already-detected short curves could be reconstructed as one by bridging **only small, pixel-supported and direction-consistent gaps**. Unlike S66, it does not freely extend an endpoint into open space; unlike S67, it does not connect arms across a generic crossing by declaring physical identity.

The source-blind predictor takes only the 192×192 uint8 photo-backed pixel image, frozen S72-selected line proposals, and the previously fitted S56 source-only scorer. Two new proposal policies were frozen before CAL: `short_gap_strict` (gap ≤5.5 px, ≤18° directional deviation, stronger minimum darkness) and `short_gap_relaxed` (gap ≤8.5 px, ≤25°, weaker minimum darkness). A unique continuation is required, with a competitor-margin veto and third-line veto. Merged curves are **research candidates only**. The source polylines of injected tracks never enter either arm's inference.

The corrected S69 official synthetic correctness evaluator and separate S76 shape-only/endpoint evaluator consume generator geometry strictly *after* all predictors finish. Neither evaluator nor frozen S39/S56/S68/S69/S72 source code was changed.

## Data and frozen protocol

- H50Z: 39 real, private 512×512 crops from photographic images `005` and `006` **of the same physical run**, 21+18. No human-dense GT on them.
- Synthetic sources: 145 complete film-070 visualMetro frames, whole-frame split **79 TRAIN / 30 CAL / 36 TEST**. Their measured 2D curves do **not** belong to photographed H50Z events.
- Preexisting frozen S72 CAL: `1ed7e58667aaab1a7d5ba91252db64b38bf8d3da517396b9e18e9e6bb484a952`; graph arm is `s68_branch_endpoint_refine` for 005→006 and `s39_original` for 006→005.
- New **CAL only**: synthetic scene seeds 20267701 and 20267711, 16 scenes per source; clean and blur σ=1.1 conditions. Decision made before TEST: `baseline` in both directions, because neither new arm met the prespecified CAL simultaneous safety, specificity and reconstruction gates.
- New **TEST**: seeds 20267703/04 and 20267713/14, 24 scenes each, **96 new synthetic scenes**, 384 artificially true links and 192 whole artificial 3-piece chains, tested on clean, blur σ=1.1, and σ=1.6 images. Different scene seeds than S75/S76, but same source frame split and two photographic backgrounds.
- The saved CAL JSON is immutable for the TEST run. Its SHA-256 is embedded in TEST results. TEST was repeated on the same scenes; the aggregate JSON is **byte-for-byte identical**, not a statistical reproducibility guess.

## Results: full image→instances→candidate links→graph

Each image condition has 96 artificial scenes. "Other" selected joins means incorrect or unmatched relative to **artificial** source truth; it is not a measured physical false-join count.

| Pixel condition | Method | Correct artificial joins | Other joins | Full 3-piece synthetic paths | True synthetic fragments shape-matched | Short-gap repairs |
|---|---|---:|---:|---:|---:|---:|
| Clean | **S72 baseline** | **54** | 33 | **14** | **260** | 0 |
| | S77 strict | 47 | 31 | 12 | 246 | 10 |
| | S77 relaxed | 37 | **21** | 8 | 214 | 31 |
| Blur σ=1.1 | **S72 baseline** | **15** | 24 | **4** | **133** | 0 |
| | S77 strict | 12 | 21 | 3 | 124 | 8 |
| | S77 relaxed | 10 | **18** | 2 | 118 | 16 |
| Blur σ=1.6 | **S72 baseline** | **11** | 18 | **1** | **113** | 0 |
| | S77 strict | **11** | 16 | **1** | 112 | 5 |
| | S77 relaxed | 8 | **14** | **1** | 101 | 13 |

**Mechanistic explanation:** by joining distinct weak pixel proposals into longer curves, the predictor often **reduces the number of independently identifiable fragments**. Although the remaining graph may propose fewer incorrect pairs, that comes at a significant reduction in synthetic true identities and complete paths. The S76 evaluator substantiates this distinction: on clean scenes S72 yields 260 shape matches, relaxed S77 only 214, a loss of 46. At moderate blur there are 133 vs 118 shape matches; the algorithm has not solved the missing-instance problem.

## Native H50Z smoke (no synthetic tracks)

Exactly one fixed **central 192×192** image window per each of all 39 genuine private H50Z 512×512 crops was processed without source labels. This is an offline machine-only feasibility check, **not a validation metric**:

| Method | Machine candidate curves across 39 real windows | Proposed graph joins, all `UNKNOWN` | Pixel-supported short-gap repairs |
|---|---:|---:|---:|
| Baseline S72 | 83 | 3 | 0 |
| S77 strict | 80 | 2 | 3 |
| S77 relaxed | 79 | 2 | 4 |

The fact that the algorithm can form 3–4 native short-gap *hypotheses* says nothing about whether they belong to the same particle. The corresponding private PNG overlays are withheld from public Git. A fixed center window was used so results could not be selected after visual inspection.

## Verdict / next research direction

1. **Do not promote S77.** CAL rejected both variants, untouched TEST confirms the trade-off; original S72 retained.
2. A hard merge is too irreversible for instance-ambiguous bubble-chamber images. Better next step: **preserve original fragments and propose local alternative continuations without deleting existing instances**, enforcing alternative-hypothesis exclusivity in the graph. A stronger version would estimate per-pixel instance membership or confidence on synthetic insertions while testing new backgrounds without silently accepting synthetic GT as native physics truth.
3. This work does not produce human-confirmed physical links, mass, charge, stereo trajectories or 3D tracks. No calibration for independent physical runs exists, no native automatic physics joining is authorized.

## Reproduction and receipts

```bash
python -m pytest -q tests/test_s77_*.py
python scripts/research/bubble_chamber/s77_calibrate_then_test_short_gaps.py cal \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/H50Z_CROPS \
  --out /PRIVATE/S77_CAL.json
python scripts/research/bubble_chamber/s77_calibrate_then_test_short_gaps.py test \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/H50Z_CROPS \
  --calibration /PRIVATE/S77_CAL.json --out /PRIVATE/S77_TEST.json
python scripts/research/bubble_chamber/s77_native_unlabeled_centre_audit.py \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/H50Z_CROPS \
  --out /PRIVATE/S77_NATIVE.json
```

Public-safe aggregates and hashes are stored in `S77_PUBLIC_FROZEN_CAL_2026_10_26.json`, `S77_PUBLIC_FRESH_TEST_2026_10_26.json`, `S77_PUBLIC_NATIVE_UNLABELED_SMOKE_2026_10_26.json`, and `EXPERIMENT_IDS_MACHINE_ONLY.json`. Scene-specific coordinates, original photos and individual film-070 frame IDs remain **PRIVATE**.
