# S70 — correction of the historical S64 pixel-to-graph evaluation (2026-10-26)

**Status:** reproducible retrospective erratum; historical S64 model predictions and thresholds are unchanged. The prior S64 aggregate is retained for provenance, not silently overwritten.

## Why the correction was necessary

The S64 post-hoc synthetic scorer assumed a fixed order of stored points. The frozen scene generator reverses source polylines independently. A true bond may therefore connect `[0]` rather than `[-1]` for a fragment, and the old scorer labeled it incorrect. S69 verifies this mechanism with exact reversal unit tests and introduces an **evaluation-only** orientation-invariant reference endpoint check. Synthetic truth never enters pixel inference or candidate generation.

## Exact retrospective replay

Four original S64 seeds (20266403/04 and 20266413/14), 24 scenes each, **96 artificial scenes** in total. This is a **known historical test set, not an additional independent test**. Native photographs are only unlabeled backgrounds from one H50Z run; 145 source visualMetro film-070 measured frames are NOT matching picture GT.

The original S64 predictor, the frozen geometry-only S56 scorer, and image-to-curve detector were not modified. An evaluation-only wrapper ran old and new scorers on the **same completed inference**. We required the old metrics including confidence intervals, source identities matched, selected edge counts, fixed source thresholds, synthetic oracle and union proxy bounds to match the original frozen S64 JSON exactly. **All checks passed.**

| Method | Correct synthetic joins: historical → corrected | Synthetic noncorrect joins: historical → corrected | Complete three-piece paths: historical → corrected |
|---|---:|---:|---:|
| S39 original | **9 → 53** | 79 → 35 | **0 → 13** |
| S64 junction-less | **10 → 55** | 84 → 39 | **0 → 15** |

Both methods generated exactly the same numbers of proposed edges as in the original evaluation: 88 for S39 and 94 for junction-less, respectively. Only the correctness audit changed. Original source integrity SHA-256 and original summary hash are enforced in `scripts/research/bubble_chamber/s70_reaudit_s64_historic.py`.

By direction (S39 corrected): 005→006 **35** correct / 16 noncorrect / 9 complete paths; 006→005 **18** correct / 19 noncorrect / 4 complete paths. Junction-less corrected: 005→006 **37/19/10**, reverse **18/20/5**.

## Interpretation and restrictions

1. The S64 conclusion that the full image-to-graph path produced **zero** completed synthetic 2-D chains was based on a defective evaluator. It is retracted; corrected counts are **13 and 15** in the historical synthetic experiment.
2. The detectors still recover only about **53/384** (S39) and **55/384** (junction-less) possible true artificial joins; there is still substantial room for pixel-level segmentation and endpoint improvement.
3. Junction-less gains two synthetic correct joins and two complete paths compared with S39 but also generates four more noncorrect joins. It **does not pass a strict no-increase-of-errors deployment gate**.
4. No user/physicist annotated real H50Z line identity exists. **Native verified physical joins remain zero.** None of these rates are clinical/physics reconstruction accuracy; no real 3-D momentum, direction or particle mass is claimed.
5. S70 is retrospective; new independent confirmation would require new image sources, not reused artificial S64 scenes. Do not tune S68 or S39 on S70 figures.

## Exact provenance

- Experiment ID: `ANZA-LIRA-BC-S70-20261026`
- Frozen protocol: `docs/bubble_chamber/S70_FROZEN_S64_RETROSPECTIVE_REEVALUATION_2026_10_26.md`
- S70 reproducible runner: `scripts/research/bubble_chamber/s70_reaudit_s64_historic.py`
- Frozen protocol SHA-256: `eda0c9c75b7e28b2523978450f2bb10d1af3607b3f170b8290fa0bba41a2d783`
- Corrected S70 runner SHA-256: `6b1e7b0b70839f3099bad933cb0ddd22f61b32a965a4bb2a2be437a96426fd14`
- Original S64 benchmark SHA-256: `ad31fef03bc40fc70606830527300873ae013967e6ac19836a115e1b95ed55c7`
- Original S64 evaluation SHA-256: `c148e55dc9a94f2841e34d2f5e1f3f6b8c41a1ebd345152d2ca21ee7c3ff824c`
- Corrected S69 evaluator SHA-256: `3e4ad80c0fc544b7b231b6a7b7dcdc3501484830a98d6d19c133dc0eb1c5b495`
- Original S64 public aggregate SHA-256: `0cf514960cc47651a0b0b5579d0cd728d108f8d639eaf0ee6dc89565f33c3301`
- S70 public-safe retrospective aggregate SHA-256: `ab523bad98f68cf7a6b5fe0e0460d65ac915088d45c7534b4b34b3a22898f357`
- Full source tests: S35–S69 214 PASS; S70 3 focused PASS, and the S70 corrected S64 historical replay reproduced identical aggregate SHA on two executions.

No source photo pixels, measured coordinates, individual source frame identities, original 7z files or checkpoint weights have been committed.
