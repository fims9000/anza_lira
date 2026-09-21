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

## Primary benchmark: 800-case ImageCAS-X

The primary geometry/topology benchmark is now the full 800-case ImageCAS-X bundle with the official 560 / 80 / 160 patient split. The older 921/953 transfer study below is retained as historical controlled evidence, but it is no longer the representative headline benchmark.

The 800-case audit contains 10,615 controlled scenes and explicitly tests PAIR / JUNCTION / NO-REPAIR existence. Candidate recall is effectively saturated under the frozen stress protocol, while relation existence / branch identity remains the dominant bottleneck.

Key test30 facts:

- true pair top-3 ranking: `99.15%`;
- true junction top-3 ranking: `92.97%`;
- geometry relation-type system: exact `55.46%`, false `10.92%`;
- perturbation-consistency gate selected on validation at `0.60`: coverage `74.34%`, false among accepted `0.823%`.

Crucially, accepted repair-needed scenes are often still incomplete, so consistency is a safety layer rather than evidence of high repair recall.

Full report: `docs/research/ccta_graph_lira_safe_repair/LARGE_SCALE_800_CASE_REPORT.md`.

## Latest proxy result: binary-lumen spatial context

With matched CCTA intensities still unavailable, a frozen proxy experiment used the synthetically broken **binary vessel mask** as spatial context for pair-presence and junction-presence decisions. This is not CT evidence.

Geometry + mask improves relation-existence ranking:

- pair test30 AUROC `0.8543 -> 0.9288`;
- pair test45 AUROC `0.8166 -> 0.9001`;
- junction test30 AUROC `0.9100 -> 0.9314`;
- junction test45 AUROC `0.8661 -> 0.8946`.

At a validation-defined <=5% structural false budget, geometry+mask gives:

- test30 exact `56.50%`, false `6.50%`;
- test45 exact `51.60%`, false `5.98%`.

Against the canonical four-class geometry relation head, patient-cluster bootstrap shows a clear reduction in false structural repair but no clear exact-rate gain:

- test30 false difference `-4.43 pp`, 95% CI `[-5.68,-3.20]`;
- test45 false difference `-2.07 pp`, 95% CI `[-3.12,-1.03]`.

The safety gain comes with more incomplete decisions, and degree-4 junctions remain a severe failure mode. Full details: `MASK_SHAPE_CONTEXT_PROXY.md`.

## Historical controlled result: patient-to-patient structural transfer

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

## Matched CCTA data status — six-case pilot

The external matched-CT blocker is now resolved for a six-case original ImageCAS / ImageCAS-X pilot.

Exact original ImageCAS CCTA volumes are available for:

- train: `953, 964`;
- validation: `957, 966`;
- test: `980, 984`.

All six pass exact shape, spacing and affine checks against the corresponding ImageCAS-X multi-label segmentation. After the expected VTK LPS -> NIfTI RAS x/y sign conversion, 100% of centerline points are in bounds, inside a non-zero vessel voxel, and match their VTK anatomical segment label at nearest-voxel sampling. CT SHA256 provenance is frozen in the committed alignment artifact.

A first easy candidate-ranking pilot is geometry-saturated and does not support a CT-ranking claim. A harder geometry-matched wrong-branch relation-presence stress is more informative. With the threshold frozen on validation under FPR <= 5%, held-out test behavior was:

- geometry: recall `1/47 = 2.13%`, false `0/47`;
- CT only: recall `22/47 = 46.81%`, false `0/47`;
- geometry + CT: recall `29/47 = 61.70%`, false `0/47`.

This is a small controlled six-patient pilot, not population-level risk evidence or natural-gap clinical validation. The two test patients are heterogeneous: geometry+CT recall is `3/16` on scan 980 and `26/31` on scan 984.

Full checkpoint: `MATCHED_CT_6CASE_CHECKPOINT.md`.

## Immediate matched-CT experiment

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

## Dataset provenance and earlier image-context work

For future continuation, do not reconstruct dataset identity from chat or legacy numeric labels.

- exact ImageCAS / ImageCAS-X repositories, official download links, mirror links, hashes and ID cautions: `DATA_SOURCES.md`;
- four-case CT + binary-mask image-context experiments and corrected BDMAP identities: `FOUR_CASE_IMAGECAS_PILOT.md`;
- local-file hashes and available annotation packages: `DATA_MANIFEST.md`.

The historical four-case scripts used row-derived labels `953/956/957/960`; their canonical source identities are `BDMAP_00015590/15593/15594/15597`. Those row labels must not be joined to ImageCAS-X anatomical IDs.

## Latest checkpoint: broken-mask context is a safety cue, not the missing repair model

The 800-case proxy study has now been pushed one step further.

Using the synthetically broken binary lumen mask as extra context clearly improves PAIR / JUNCTION existence ranking, but the strongest controlled conclusion is **not** that mask context solves repair identity. It behaves primarily as a conservative veto.

A canonical relation-head + mask-presence veto, selected jointly on val30 and val45 at a 5% structural-false budget, reduced test false-scene rate:

- test30: `10.92% -> 5.93%`;
- test45: `8.05% -> 4.99%`.

This safety gain costs exact repair completion:

- test30 exact: `55.46% -> 53.86%`;
- test45 exact: `51.37% -> 49.01%`.

The paired patient-cluster bootstrap confirms both directions: false structural decisions decrease clearly, while exact completion also decreases and incomplete decisions increase.

Subgroup analysis explains why. The mask veto is excellent at rejecting NO-REPAIR scenes, but it suppresses true repair-needed PAIR scenes too aggressively. Degree-4 junctions remain unresolved.

A second joint validation search allowed mask-veto thresholds and perturbation consistency to trade off under <=1% false among accepted on **both** validation stress levels. It selected strong mask vetoes (`pair=0.97`, `junction=0.96`) and no consistency gate, but did **not** produce a reliable gain in repair exact yield on test patients. This reinforces the interpretation that broken-mask occupancy is an existence/safety signal rather than the missing branch-identity model.

Separately, robust two-stress re-selection of the original relation confidence + perturbation-consistency policy selected the same canonical operating point again:

- relation confidence `tau=0.85`;
- consistency `0.60`.

So the frozen selective policy is stable across the tested 30-degree and 45-degree validation stresses and should not be retuned on test.

Full checkpoint: `docs/research/ccta_graph_lira_safe_repair/MASK_CONTEXT_CHECKPOINT.md`.

Current scientific implication: geometry and binary occupancy are near a diminishing-return ceiling for the central repair-identity question. The six-case matched-CCTA pilot now shows that real intensity can recover additional true relations at a validation-frozen low-false operating point, but the sample is too small and patient-to-patient heterogeneity is too large for a final performance claim.


## Latest matched-CT representation / graph checkpoint

The six-case matched-CCTA line has now advanced beyond the first relation-presence pilot.

On the frozen geometry-matched wrong-branch stress, the strongest current local representation is **radial 2.5-D CCTA context**. At a validation-only FPR <=5% operating point, held-out test recall was:

- geometry: `1/47 = 2.13%`, false `0/47`;
- radial 2.5-D: `29/47 = 61.70%`, false `0/47`;
- geometry + radial 2.5-D: `30/47 = 63.83%`, false `0/47`.

A coarse 3-D tube PCA representation and a small cross-section CNN transferred worse with only two training patients. This does not support escalating model capacity yet.

The canonical full-800 four-class geometry relation head was independently regenerated and re-selected the same frozen `HGB, tau=0.85` operating point, reproducing test30 exact `55.46%` / false `10.92%` and test45 exact `51.37%` / false `8.05%`.

The first attempt to inject the six-case CT presence signal into the frozen PAIR / JUNCTION / BOTH / NONE Graph-LIRA relation layer was **not safe**. CT add-only increased matched-test exact from `53.33%` to `56.67%` but false structural decisions from `10.00%` to `23.33%`. A validation-only hysteresis rule looked safer on the two validation patients but still reached `20.00%` false on held-out test.

Therefore the supported conclusion is now more specific: **CCTA intensity is informative locally, but six-patient scene-level calibration does not generalize well enough to preserve the low-false Graph-LIRA objective.** More matched patients are the next requirement; further threshold tuning on scans 980/984 is prohibited.

Full checkpoint: `MATCHED_CT_REPRESENTATION_AND_GRAPH_CHECKPOINT.md`.


## Expanded matched-CCTA execution — 28-patient cohort

The next calibration experiment is now frozen and fully documented. It expands the exact original ImageCAS / ImageCAS-X matched cohort from 6 to **28 patients** while preserving the official patient split:

- train: 17 patients;
- validation: 5 patients;
- test: 6 patients.

The frozen geometry-matched relation plan contains **1,360 examples**:

- train: 379 positive + 379 hard negative;
- validation: 134 + 134;
- test: 167 + 167.

The first representation remains the current strongest simple image baseline: candidate-aligned **radial 2.5-D CCTA**. The comparison is geometry vs radial CT vs geometry+radial CT under the same lightweight logistic model and a threshold selected only on validation at FPR <= 5%.

The canonical Graph-LIRA policy is not reopened:

- four-class geometry relation head remains frozen;
- relation confidence remains `tau=0.85`;
- perturbation consistency remains `0.60`;
- test patients are not used for threshold tuning.

Exact protocol, download requirements, execution commands, output contract, post-run analysis, ANZA decision point and the long-term publication direction are documented in:

- `EXPANDED_CT_28CASE_EXECUTION_AND_ROADMAP.md`;
- `EXPANDED_CT_28CASE_EXECUTION_STATUS.md`;
- `EXPANDED_CT_28CASE_PLAN.md`.

Exact executable source is archived at:

- `scripts/research/ccta_graph_lira_safe_repair/extract_radial_features_z04.py.gz.b64`;
- `scripts/research/ccta_graph_lira_safe_repair/build_expand_pair_plan.py.gz.b64`.

Frozen experiment metadata is under:

- `experiments/ccta_graph_lira_safe_repair/expanded_ct_28case/`.

### Actual execution status

The runner has been compiled, its frozen plan has been checked, and it has been launched in the current research environment. It initializes the expected 28 patient IDs correctly and then stops at the raw-data boundary because this environment does not contain `801-1000.z04`.

The required raw ImageCAS archive pieces are intentionally not committed to Git. The exact external requirement is:

- `801-1000.z04`;
- `801-1000.change2zip`, renamed locally to `801-1000.zip`.

No post-blocker result has been invented. As soon as those raw archive pieces are available to the execution environment, the next action is to rerun the already-frozen extractor unchanged and continue directly into patient-cluster uncertainty and frozen Graph-LIRA integration.
