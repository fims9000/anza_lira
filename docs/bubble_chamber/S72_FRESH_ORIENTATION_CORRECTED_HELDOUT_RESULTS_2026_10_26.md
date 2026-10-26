# S72: prospective synthetic-on-H50Z test with correct evaluator from the start

**26 October 2026. Experiment ID `ANZA-LIRA-BC-S72-20261026`.** The frozen protocol was written before CAL and TEST execution. This is new synthetic-seed evidence (96 scenes), but **not a new independent physical H50Z camera run**.

## Why S72 exists

S71 found strong orientation-dependent bias in historical S65–S67 evaluation. S72 tests whether the fixed, already-existing S68 branch splitting and pixel-only endpoint refinement can outperform S39 when S69's orientation-independent audit is used for **both** CAL selection and new TEST, not merely retrospectively. Original S56 geometry scorer, S68 pixel detectors, CAL cutoff, strict pair gate, graph MILP and all numeric parameters are unchanged.

## Frozen data, selection and access boundary

- H50Z: exactly 39 private 512px crops of only TWO real photos from ONE physical run; no expert verified native H50Z particle identities.
- Shape source: visualMetro film 070 complete 145 frames, frozen whole-frame 79 TRAIN / 30 CAL / 36 TEST. The shape data do **not** correspond to real H50Z particle ground truth.
- CAL synthetic scene seeds `20267201` (source 005) and `20267211` (source 006), 32 scenes each. TEST seeds `20267203`, `20267204`, `20267213`, `20267214`, 24 scenes each (96 scenes in total). No reuse of S64–S71 scene seeds.
- Full model and all image predictions are computed from image pixels alone. Artificial GT identity and synthetic segment endpoint location enter **only** the evaluation stage, via unmodified S69 orientation-invariant evaluator.
- S68's original predeclared CAL-only policy requires >=2 more correct artificial joins than S39, no increase in noncorrect joins or scenes with false joins, and no loss of complete 3-piece synthetic paths.
- CAL JSON saved **before** TEST and validated by SHA-256. TEST evaluates all four original methods, but selection is determined independently for each H50Z source photo direction by CAL alone.

## S72 CAL (64 source-photo synthetic scenes)

| Source | Method | CAL correct synthetic joins | CAL noncorrect | Complete paths |
|---|---|---:|---:|---:|
| 005 | S39 | 14 | 21 | 2 |
| 005 | S68 branch+endpoint refinement | **17** | **16** | **4** |
| 006 | S39 | 14 | 9 | 2 |
| 006 | S68 branch+endpoint refinement | 15 | 7 | 2 |

**Frozen choices:** 005→006 uses `s68_branch_endpoint_refine`; 006→005 uses `s39_original`. Reverse-direction candidate did not satisfy the CAL **minimum +2 correct joins**, although it had fewer noncorrect proposals.

## NEW TEST (96 synthetic scenes across both directions)

| Arm | Synthetic-correct joins | Other selected joins | Complete synthetic 3-piece paths | Synthetic scenes containing any noncorrect join |
|---|---:|---:|---:|---:|
| S39 original | **56** | 52 | 12 | 40 |
| S67 crossing continuation | 49 | 45 | 10 | 37 |
| S68 topological branch splitting | 52 | 45 | 12 | 34 |
| S68 branch split + endpoint refinement | 53 | **35** | **13** | **31** |
| **Frozen per-direction CAL-selected hybrid** | **56** | **41** | **13** | **35** |

The selected hybrid consists of the S68 refined variant for 005→006 (33 correct, 22 other, 9 paths), and original S39 for 006→005 (23 correct, 19 other, 4 paths). The original all-direction S39 benchmark was 56 correct, 52 other, 12 paths. Thus the selected hybrid reduced other synthetic proposals by 11, kept the correct link count unchanged and improved the synthetic complete-path count by 1.

**Paired-scene uncertainty by newly locked TEST seed:** in 005→006, the refined-vs-S39 noncorrect-link mean differences were −0.292 and −0.167 per artificial scene; 95% scene-bootstrap intervals `[−0.542, −0.083]` and `[−0.333, −0.042]`. Correct-link deltas were 0 in both source-005 seeds, with intervals `[−0.125, +0.125]` and `[0,0]`. This is conditional evidence in two H50Z photographs, **not independent-camera physical confidence**. See the public TEST JSON for all four arm/seed intervals and the separate photo direction.

## Gate / deployment verdict

This is a **conditional positive result for a CAL-selected synthetic demonstration**, not a physics production improvement. The global S68 refined arm alone lost 3 correct joins (56→53) while reducing 17 noncorrect, so universal replacement is NOT justified. The per-direction CAL-selected hybrid preserves this TEST's correct joins and raises complete paths by only 1, which is too little to establish a robust physics-generalization claim. The real native H50Z track identity correctness is still unknown and no automatic 3D or physical joining is authorized. Preserve native S39 as conservative default.

## Reproduction and provenance

```bash
python -m pytest -q tests/test_s72_fresh_protocol.py
python scripts/research/bubble_chamber/s72_fresh_corrected_branch_benchmark.py cal \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/source_39_H50Z_crops \
  --out /PRIVATE/S72_CAL_FROZEN.json
python scripts/research/bubble_chamber/s72_fresh_corrected_branch_benchmark.py test \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/source_39_H50Z_crops \
  --calibration /PRIVATE/S72_CAL_FROZEN.json --out /PRIVATE/S72_TEST.json
```

Exact source locks, CAL SHA and per-seed bootstrap intervals are recorded in `S72_PUBLIC_FROZEN_CAL_2026_10_26.json` and `S72_PUBLIC_FRESH_TEST_2026_10_26.json`. Public outputs contain only synthetic aggregated metrics, protocol hashes and experiment IDs; not original photo pixels, 7z, operator source coordinates, model weights, or per-image link labels.
