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

## Current strongest result: patient-to-patient structural transfer

A controlled stress benchmark now trains/calibrates the geometry model on one labelled coronary tree and evaluates on the other.

Primary setting: `30 deg` tangent error + `1 mm` endpoint jitter.

### Train 921 -> test 953

- pair AUROC: `0.98269`;
- independent local decisions: `53.70%` scenes with a false structural link;
- sequential junction-then-pair: exact `87.04%`, false `8.80%`;
- joint Graph-LIRA: exact `94.44%`, false `1.85%`.

With 15 perturbation reruns and canonical V2 consistency `>= 0.90`:

- coverage: `63.43%`;
- accepted: `137`;
- false among accepted: `0 / 137`;
- exact among accepted: `99.27%`.

### Train 953 -> test 921

- pair AUROC: `0.98136`;
- independent local decisions: `54.36%` scenes with a false structural link;
- sequential junction-then-pair: exact `80.54%`, false `9.40%`;
- joint Graph-LIRA: exact `88.59%`, false `2.68%`.

With canonical V2 consistency `>= 0.90`:

- coverage: `44.97%`;
- accepted: `67`;
- false among accepted: `0 / 67`;
- exact among accepted: `98.51%`.

These are finite-sample controlled centerline stress results, **not** natural-gap clinical validation. Full protocol and machine artifacts are in:

- `docs/research/ccta_graph_lira_safe_repair/CROSS_PATIENT_GRAPH.md`;
- `docs/research/ccta_graph_lira_safe_repair/REPRODUCIBILITY_FIX_V2.md`;
- `docs/research/ccta_graph_lira_safe_repair/SELECTIVE_RISK_UNCERTAINTY.md`;
- `results/ccta_graph_lira_safe_repair/2026-09-20/cross_patient_graph_v2_*.csv`.

The V2 perturbation seeds are keyed to stable scene identities. Older cross-patient risk/coverage files without the `v2` prefix are retained for provenance but are superseded.

## Where the remaining geometry failures are

The branch-stratified audit shows that the hard residual regime is concentrated around LAD/LCX and higher-degree branching.

At `30 deg + 1 mm`:

- held-out 953 D1 and RCA scenes were exact in all generated scenes;
- held-out 953 LAD: exact `86.67%`, false `6.67%`;
- held-out 953 LCX degree-3: exact `89.39%`, false `3.03%`;
- held-out 921 LCX degree-4: exact only `63.33%`, with `30%` incomplete but non-false decisions.

In canonical V2, the held-out degree-4 LCX stratum has mean consistency `0.647`, median `0.60`, and only `10 / 30 = 33.33%` of scenes survive consistency `>= 0.80`. This is the most concrete target for adding CT image evidence.

Across the complete 30-degree V2 run, all observed false structural decisions fall below consistency `0.60`. That is a useful failure-analysis observation, **not** a final threshold choice: 921 and 953 have already been inspected and cannot be used to select the publication operating point.

## Previous perturbation-consistency artifact

In the earlier within-case stress artifact at `30 deg + 1 mm`, `combined_consistency` / `baseline_agreement` gave:

| threshold | coverage | accepted | false among accepted | exact among accepted |
|---:|---:|---:|---:|---:|
| 0.80 | 0.7287 | 325 | 0.00308 | 0.97231 |
| 0.85 | 0.6547 | 292 | 0.00342 | 0.98288 |
| 0.90 | 0.5583 | 249 | 0 / 249 | 0.98795 |
| 0.95 | 0.3991 | 178 | 0 / 178 | 0.99438 |

Important provenance note: the commonly quoted `0 / 249` at threshold `0.90` is supported by `baseline_agreement` / `combined_consistency`. Raw `stability >= 0.90` accepts 250 scenes and contains one false scene (`0.004`).

## Sequence-context pilot on four CCTA cases

Uploaded cross-patient results compare two compact token-sequence models:

- sequence CNN: mean AUROC `0.927414`, median AUROC `0.937662`;
- sequence Transformer: mean AUROC `0.929432`, median AUROC `0.947155`.

Their mean FPRs are high (`~0.244` for both), so this pilot does **not** support the claim that a Transformer solves the continuation problem by itself. The next sequence model must preserve spatial information in each cross-section rather than compressing each slice to hand-crafted statistics.

The previous exploratory session also found radial 2.5-D and candidate-aligned 3-D tube representations around mean AUROC `~0.959`. These numbers remain exploratory until the exact scripts/artifacts are re-run and archived.

## Chosen direction

The main novelty is not "a larger 3-D network" and not "a Transformer instead of a CNN". The chosen direction is **risk-controlled structural repair**:

- local evidence proposes plausible pair / junction relationships;
- a global graph layer enforces structural compatibility;
- uncertainty controls automatic repair versus abstention;
- image context is added specifically to reduce confident wrong-branch connections that geometry alone cannot reject.

ANZA remains a candidate local encoder / feature source, but it is not assumed to be beneficial until an ablation proves incremental value.

## Data status for CT + branch labels

The ImageCAS-X anatomical packages for scans 921 and 953 are valid for geometry research. Scan 953 centerlines and its multi-label mask are internally aligned.

A previously downloaded third-party BDMAP candidate was tested and **rejected** as the corresponding original CT: mask Dice was only `0.0176` and a free rigid ICP fit remained far outside a defensible alignment.

The official ImageCAS-X data description says that each ImageCAS-X patient ID is identical to the original ImageCAS dataset ID, and its benchmark layout expects:

```text
volumes/<scan_id>.img.nii.gz
```

Therefore the next matched image target is the original ImageCAS volume belonging to scan ID `953`, not a row-indexed BDMAP mirror guess.

See `docs/research/ccta_graph_lira_safe_repair/ALIGNMENT_953.md`.

## Immediate next experiment after matched CT is available

```text
geometry-only pair/junction score
vs
CT radial 2.5-D
vs
candidate-aligned 3-D tube
vs
full cross-section CNN/ANZA encoder -> sequence context
```

All local representations must be evaluated inside the **same** joint Graph-LIRA and perturbation-consistency decision layer.

The primary question is whether image evidence reduces false / incomplete decisions specifically in the frozen hard anatomical strata, especially LAD/LCX and degree >= 4 junctions, not whether a larger network gives a higher average AUROC.

The current zero-false accepted counts are not yet evidence for a sub-1% population risk. Exact one-sided 95% binomial upper bounds are still about `1.49%` for 0/199 and `2.24%` for 0/132 at consistency 0.60. Roughly 299 independent accepted cases with zero failures would be needed just to push that bound below 1%.

## Reproducibility rules for this branch

- do not commit external raw medical data unless licensing explicitly permits it;
- store raw-data SHA256 hashes and provenance instead;
- split by patient, never by patch across the same patient;
- select thresholds/calibration only on validation/calibration patients;
- report pair ranking separately from false-link operating behavior;
- report risk-coverage / abstention explicitly;
- never silently promote exploratory chat numbers into final paper results;
- every final number must map to a committed script, config and machine artifact.
