# S43-S44: automated H50Z track-hypothesis quality and same-coordinate structure research

**Date:** 2026-10-25. **Branch:** `research/bubble-chamber-graph-lira`. **Native physical track GT:** 0. Research deliverable with runnable code and private visual demo, **not** physics-validated particle tracking.

## Key finding 1: synthetic scratch scores are not reliable particle identity

S43 fits a feature-level discriminative baseline (balanced L2 logistic regression) to *injected synthetic-origin labels*. Two original H50Z BMPs contribute real, unlabeled backgrounds. Film-070 visualMetro measured 2D curves are inserted programmatically; they do not belong to those photographs. Frozen frame-disjoint source folds: **72 train / 37 calibration / 36 test**. Directions are photo 005->006 and reverse, but there is only **one physical H50Z run**.

| Metric on synthetic strokes only | 005->006 | 006->005 |
|---|---:|---:|
| Pooled AUROC | 0.981 | 0.978 |
| Simulated curve acceptance after q90 negative calibration | 0.969 | 0.979 |
| All simulated scratch acceptance | 0.083 | 0.097 |
| **Matched-stroke fake continuation accepted** | **0.500** | **0.583** |

**Critical negative control:** the apparently excellent pooled AUROC is misleading. Easy scanner-like scratch families have different geometric distributions (including length), while a near-track decoy drawn with the same renderer as the positive line is accepted 50-58% of the time. No scratch-removal gate is justified. Public metrics are aggregate and model scores attached to the native H50Z proposals are labeled **synthetic-origin similarity, not probability of a physical track**.

The separately documented **S43B post-hoc sensitivity** adds translated measured-curve counterfactuals and a length/sinuosity feature ablation. It is NOT an independent confirmation set. The generated clone paths still have a centerline-length distribution shifted by line-placement noise relative to measured-source fragments (see `S43B_POSTHOC_HARD_NEGATIVE_PUBLIC.json`), so its AUROC cannot rescue a real-scene accuracy claim. The failure of this control to remove simulator shortcuts is explicitly retained.

## Key finding 2: exact large-BMP registration enables a new automatic artifact cue

Two original native 8-bit BMPs of sizes 9020x26151 and 9268x25324 pixels were read through read-only row-mapped access, with **39/39 extracted source crops re-verified by pixel SHA-256**. We registered them using the six common fiducial positions f1-f6, never using putative particle tracks. Mean registration residual: **1.507px**, maximum **2.707px**. Independently, 12 known fiducial arm controls achieved mean **0.993** counterpart support after the geometric transform.

A matching coordinate anomaly is visible: f8 in 005 is spatially close to f7 in 006 (affine prediction error ~2.59px), while the labels are different. This is a *hypothesis about index/crop coverage*, **not** automatic renaming of fiducials.

| Registered-machine diagnostic on real H50Z pixels | Count |
|---|---:|
| Original S39 line hypotheses, kept unchanged | 567 |
| Both-photo line-support score >=0.45 at same registered pixels | 132 |
| Within-window shifted-path control >=0.45 | 46 |
| Machine hypotheses with known fiducial risk | 56 |
| Same-location persistent candidates outside known fiducial risk | 78 |
| Native candidates erased; physical auto-links performed | **0; 0** |

Difference (direct minus shifted control) across 39 whole crops has **exploratory scene/crop bootstrap 95% interval [0.08196, 0.27053]**. This is a diagnostic for registered image structure and is **not** native scratch precision. Fixed calibration marks, shared backgrounds, scanner patterns, repeated physical lines and chance overlap can all contribute. The two BMPs do not establish independent physical events or independent runs.

## Operational result

- All 567 original candidate lines, 182 S39 graph hypotheses and S40 explicit fiducial uncertainty flags remain available. No pseudo-label is reclassified from unknown to positive or negative, no line is automatically deleted.
- For each of 39 H50Z crops, private machine-only S43 rankings and 2D S44 counterpart support can be displayed alongside native images in the offline demonstration. Persistent appearance is a **warning cue**, not a destructive scratch filter.
- An original AZUNet or U-Net score is **not** used as a native GT teacher. S42's negative result (classical detectors outperform the pilot AZUNet/U-Net on injected lines) remains unchanged.
- VisualMetro book constraints still apply: two scans and sparse 2D curves do not constitute a calibrated 3-view physical reconstruction; the hadroide mass prior is reserved for a confirmed stopping track under the specific paper's assumptions.

## Reproduction (private inputs kept outside Git)

```bash
python scripts/research/bubble_chamber/s43_machine_artifact_trial.py \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/S31_source_39_crops \
  --s39-private /PRIVATE/S39_MACHINE_HYPOTHESES_PRIVATE.json \
  --s39-graph-private /PRIVATE/S39_GRAPH_RELATIONS_PRIVATE.json \
  --s40-graph-private /PRIVATE/S40_GRAPH_FIDUCIAL_GUARDS_PRIVATE.json \
  --output /PRIVATE/s43_result
python scripts/research/bubble_chamber/s43b_length_matched_diagnostic.py \
  --visualmetro /PRIVATE/1mH2Data/S1 --photo-tiles /PRIVATE/S31_source_39_crops \
  --output /PRIVATE/s43b_result.json
python scripts/research/bubble_chamber/s44_audit_same_coordinate_structure.py \
  --bmp-directory /PRIVATE/H50Z_originals \
  --s39-private /PRIVATE/S39_MACHINE_HYPOTHESES_PRIVATE.json \
  --s40-fiducials /PRIVATE/S40_FIDUCIAL_CANDIDATES_PRIVATE.json \
  --out /PRIVATE/s44_result
python -m pytest -q tests
```

**Independent physical confirmation still missing.** The next safe machine-only iteration is to compare geometrically persistent patterns against scanner/frame-specific nuisance features and *down-weight uncertainty only*, with a rigorously fresh test where possible. Do not present any current AUROC as native particle tracking accuracy. Original raw photos, label positions, model files and private visualizations must remain off public GitHub.
