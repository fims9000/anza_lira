# CCTA Graph-LIRA Safe Repair

Status: active research branch, created 2026-09-20.

This branch freezes the current medical/CCTA continuation line without changing `main` or the historical `research/anza-lira-q1-journal` protocol.

## Research question

Can residual connectivity errors in coronary-artery segmentation be repaired with a low false-link rate by separating:

1. candidate generation;
2. local pair / junction evidence;
3. global graph-consistency reasoning;
4. uncertainty-based abstention;
5. path construction after a structural decision has been accepted?

The working pipeline is:

```text
CCTA segmentation
    -> endpoints / candidate fragments
    -> local pair and junction scores
    -> joint Graph-LIRA structural optimization
    -> perturbation-consistency uncertainty gate
    -> confident: repair
    -> uncertain: abstain / review
    -> max-min path construction after acceptance
```

The historical ANZA-LIRA principle is preserved: local geometric plausibility and structural identity are different decisions. The earlier controlled work already showed that pair selection is the sensitive stage, while max-min path construction is reliable once the correct pair is known.

## What is currently supported by machine artifacts

### Perturbation-consistency uncertainty, 30 deg + 1 mm stress setting

The exact uploaded machine artifacts are stored under `results/ccta_graph_lira_safe_repair/2026-09-20/`.

For `combined_consistency` / `baseline_agreement`:

| threshold | coverage | accepted | false among accepted | exact among accepted |
|---:|---:|---:|---:|---:|
| 0.80 | 0.7287 | 325 | 0.00308 | 0.97231 |
| 0.85 | 0.6547 | 292 | 0.00342 | 0.98288 |
| 0.90 | 0.5583 | 249 | 0 / 249 | 0.98795 |
| 0.95 | 0.3991 | 178 | 0 / 178 | 0.99438 |

Important provenance note: the commonly quoted `0 / 249` at threshold `0.90` is supported by `baseline_agreement` / `combined_consistency`. Raw `stability >= 0.90` accepts 250 scenes and contains one false scene (`0.004`). This distinction is frozen here to avoid later misreporting.

### Sequence-context pilot on four CCTA cases

Uploaded cross-patient results compare two compact token-sequence models:

- sequence CNN: mean AUROC `0.927414`, median AUROC `0.937662`;
- sequence Transformer: mean AUROC `0.929432`, median AUROC `0.947155`.

Their mean FPRs are high (`~0.244` for both), so this pilot does **not** support the claim that a Transformer solves the continuation problem by itself. The next sequence model must preserve spatial information in each cross-section rather than compressing each slice to hand-crafted statistics.

## Exploratory results that are retained as research notes, not yet canonical machine artifacts in this branch

The previous exploratory session also found:

- radial 2.5-D and candidate-aligned 3-D tube representations around mean AUROC `~0.959` on the four-case pilot;
- joint Graph-LIRA substantially reduces inconsistent false repairs compared with independent pair decisions;
- variable-degree junction modeling is necessary because a fixed degree-3 assumption fails on real four-arm branching configurations;
- perturbation consistency is substantially more useful for selective repair than a simple top-1 / top-2 score margin.

These claims must be re-run from canonical scripts before they are used as paper numbers. They are preserved so the research path is not lost.

## Chosen direction

The main novelty is not "a larger 3-D network" and not "a Transformer instead of a CNN". The chosen direction is **risk-controlled structural repair**:

- local evidence proposes plausible pair / junction relationships;
- a global graph layer enforces structural compatibility;
- uncertainty controls automatic repair versus abstention;
- image context is added specifically to reduce confident wrong-branch connections that geometry alone cannot reject.

ANZA remains a candidate local encoder / feature source, but it is not assumed to be beneficial until an ablation proves incremental value.

## Immediate next experiment

A matched CCTA + anatomical-branch package for scan `953` is now available locally:

- multi-label coronary mask;
- left and right anatomical centerlines;
- surface mesh;
- a candidate ImageCAS CT / binary-mask pair corresponding to the external ImageCAS mapping.

Before using CT intensities, run a coordinate / registration audit. The current files are not voxel-identical: the multi-label mask is `512 x 512 x 223`, whereas the candidate CT volume is `512 x 512 x 221`; their physical transforms also differ. No CT+branch-label experiment is valid until this mapping is resolved and frozen.

After the alignment audit, the first matched-data comparison is:

```text
geometry-only pair/junction score
vs
CT radial 2.5-D
vs
candidate-aligned 3-D tube
vs
full cross-section encoder -> sequence context
```

all evaluated inside the same joint Graph-LIRA and perturbation-consistency decision layer.

## Reproducibility rules for this branch

- do not commit external raw medical data unless licensing explicitly permits it;
- store raw-data SHA256 hashes and provenance instead;
- split by patient, never by patch across the same patient;
- select thresholds/calibration only on validation/calibration patients;
- report pair ranking separately from false-link operating behavior;
- report risk-coverage / abstention explicitly;
- never silently promote exploratory chat numbers into final paper results;
- every final number must map to a committed script, config and machine artifact.
