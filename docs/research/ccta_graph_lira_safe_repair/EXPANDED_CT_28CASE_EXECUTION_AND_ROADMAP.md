# Expanded matched-CCTA Graph-LIRA study: exact execution and research roadmap

Date: 2026-09-21  
Canonical branch: `research/ccta-graph-lira-safe-repair`  
Status: runner and frozen protocol prepared; execution attempted in the ChatGPT research container; raw Kaggle z04/final split files are the only current external blocker.

## 1. What this branch is for

This branch is the durable research line for **risk-controlled connectivity repair in coronary CCTA**.

The project is no longer "try another network and compare AUROC". The central question is now:

> Can real CCTA image evidence improve PAIR / JUNCTION / NO-REPAIR relation decisions beyond geometry while preserving a deliberately low false-link operating point after global Graph-LIRA reasoning and selective abstention?

The frozen pipeline is:

```text
CCTA / segmentation
    -> endpoints and candidate fragments
    -> geometry candidate evidence
    -> image-conditioned relation evidence
    -> PAIR / JUNCTION / BOTH / NONE
    -> joint Graph-LIRA structural optimization
    -> frozen relation confidence tau = 0.85
    -> perturbation-consistency gate = 0.60
    -> confident repair OR abstain/review
    -> max-min path only after a relation is accepted
```

The separation matters. Candidate plausibility, relation identity, global compatibility, uncertainty, and path construction are different decisions and are evaluated separately.

## 2. Why we are expanding the matched CT cohort

The six-case matched ImageCAS / ImageCAS-X pilot already answered one question: real CCTA intensity is informative.

On the hard geometry-matched wrong-branch stress, radial 2.5-D CT preserved substantially more true relations than the geometry operating point at the same validation-frozen low-FPR policy.

However, direct insertion of six-patient CT calibration into the full four-class Graph-LIRA relation layer was not safe: exact repair increased only slightly while false structural decisions increased substantially on held-out matched test patients.

That means the current bottleneck is no longer:

> "Does CT contain useful information?"

It is:

> "Can the CT relation evidence be calibrated across patients well enough that the low-false Graph-LIRA objective survives patient transfer?"

That is why the next experiment increases **patient count before model capacity**.

## 3. Frozen 28-patient cohort

We use only ImageCAS IDs whose raw CT entries start in the same already-known `801-1000.z04` ZIP part, so there is no reason to download the whole 89 GB dataset.

Official ImageCAS-X split is preserved exactly.

### Train — 17 patients

`953, 955, 956, 959, 960, 963, 964, 967, 969, 970, 971, 975, 976, 977, 979, 982, 983`

### Validation — 5 patients

`957, 961, 965, 966, 974`

### Test — 6 patients

`954, 958, 972, 973, 980, 984`

No patient occurs in more than one split.

The frozen relation plan contains:

- train: 758 rows = 379 positive + 379 geometry-matched hard negative;
- validation: 268 rows = 134 + 134;
- test: 334 rows = 167 + 167;
- total: **1,360 relation examples from 28 independent patients**.

## 4. What is frozen before reading the new CT intensities

The hard relation construction remains the same as in the six-case matched-CCTA study.

Positive relation:

- true within-branch gap;
- gap length 4 mm;
- positive sampling step 2 mm.

Hard negative:

- nearby wrong branch;
- different polyline;
- different anatomical segment;
- distance 2.5–6.5 mm;
- anchor alignment >= 0.6;
- decoy alignment >= 0.6;
- axis consistency >= 0.4;
- one-to-one positive/negative matching uses geometry only.

Anatomical labels are used to construct controlled ground truth. They are **not model input**.

This is important because otherwise the image experiment would leak the answer into the predictor.

## 5. CT representation being tested first

We intentionally do **not** start with a larger CNN, Transformer, Mamba or 3-D architecture.

The strongest current six-case representation is radial 2.5-D local CCTA context.

For every candidate corridor:

1. place 17 positions between the two candidate endpoints;
2. sample the center HU profile;
3. construct an orthogonal cross-section frame;
4. sample 8 angular points on each of 1, 2 and 3 mm radii;
5. compute radial/background contrast summaries;
6. clip HU robustly to [-300, 1200];
7. combine with the already-frozen seven geometry features only in the combined ablation.

Three exact baselines:

```text
A. geometry only
B. radial 2.5-D CT only
C. geometry + radial 2.5-D CT
```

Classifier stays deliberately lightweight:

`StandardScaler + LogisticRegression(C=1, class_weight=balanced)`.

Reason: the experiment is about **patient-general image evidence**, not about whether a high-capacity learner can memorize two or three hearts.

## 6. CT / annotation integrity gate

Before a patient's HU values can enter the experiment, the raw original ImageCAS CT must match the frozen ImageCAS-X geometry.

For every case the runner checks:

- NIfTI shape;
- voxel spacing;
- full sform affine;
- expected ImageCAS-X mask provenance;
- raw CT SHA256.

Expected geometry is committed in:

`experiments/ccta_graph_lira_safe_repair/expanded_ct_28case/expected_ct_geometry.csv`

A mismatch causes a hard stop. The script must never silently resample a wrong patient into an apparently valid experiment.

Coordinate convention remains explicit:

`ImageCAS-X VTK LPS -> ImageCAS NIfTI RAS` by negating x and y before applying the inverse CT affine.

## 7. Threshold selection and test discipline

For every model:

- fit on train patients;
- select the operating threshold **only on validation patients**;
- threshold rule: maximize recall subject to `FPR <= 5%`;
- apply that threshold unchanged to all held-out test patients.

The previous held-out scans 980 and 984 have already been inspected in the six-case pilot. They must not be used for additional tuning.

The expansion adds four other held-out test patients so patient heterogeneity can be measured more honestly.

## 8. What remains frozen from canonical Graph-LIRA

The CT experiment does not reopen the full geometry pipeline.

Keep frozen:

- candidate generator;
- geometry candidate identity / compatibility models;
- four-class relation head baseline;
- global structural optimizer;
- relation confidence `tau = 0.85`;
- perturbation consistency `0.60`;
- max-min path construction downstream of accepted structure.

The only component that may be refit after the local 28-case experiment is the **image-conditioned relation-presence / calibration layer**.

## 9. Exact files required from Kaggle

Source dataset:

`https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas`

Only two raw split-archive pieces are needed for this runner:

1. `801-1000.z04`
2. `801-1000.change2zip`

The second file is the final split-ZIP member. Rename it locally to:

`801-1000.zip`

The runner deliberately refuses entries that do not begin on raw ZIP disk 3 (the z04 part).

You do **not** need all 89 GB.

## 10. Runner stored on this branch

Core source archive:

`scripts/research/ccta_graph_lira_safe_repair/extract_radial_features_z04.py.gz.b64`

Restore:

```bash
base64 -d scripts/research/ccta_graph_lira_safe_repair/extract_radial_features_z04.py.gz.b64 \
  | gzip -d > extract_radial_features_z04.py
```

Frozen plan generator already lives on the branch:

`scripts/research/ccta_graph_lira_safe_repair/build_expand_pair_plan.py.gz.b64`

The ready all-in-one local bundle used in this checkpoint is named:

`GraphLIRA_expand_radial_runner.zip`

Bundle SHA256:

`3e215e4e1bded4dbb824142869701a509cb806b882013fc80eccb1649b622d21`

Core extractor SHA256:

`c5c0d287bdf30bc4879c03398037afd6381b9b0d63a341b2079311d66e1ce1de`

Frozen pair-plan SHA256:

`4629bd082457bdb14a3c7653e6fc8b1592d648d91f087e14b522c38f0ccca523`

Expected CT geometry SHA256:

`309ba692c82cef14dcb80bb5863d49775b35840c83fb36847b865da8c4a83359`

## 11. Reproduction command

Expected local layout:

```text
kaggle_801_1000/
    801-1000.z04
    801-1000.zip
    graphlira_expand_radial/
        extract_radial_features_z04.py
        relation_pair_plan.csv
        expected_ct_geometry.csv
        geometry_matching_audit_plan.csv
        pair_plan_counts.csv
```

Run:

```bash
python graphlira_expand_radial/extract_radial_features_z04.py \
  --archive-dir . \
  --out-dir ~/GraphLIRA_CT_expanded_radial
```

The output archive is:

`~/GraphLIRA_CT_expanded_radial_results.zip`

It contains no raw CT.

## 12. Output contract

The result bundle must contain:

- `ct_alignment.csv`;
- `expanded_relation_features.csv`;
- `expanded_relation_summary.csv`;
- `expanded_relation_predictions.csv`;
- `protocol.json`;
- frozen relation plan;
- frozen expected CT geometry.

No raw NIfTI volume is written to the result ZIP.

## 13. Analysis immediately after the 28-case run

The experiment does not stop after the first summary table.

### Stage A — integrity

Verify all 28 CTs:

- patient ID;
- shape;
- spacing;
- affine;
- SHA256;
- split.

Any mismatch invalidates that patient before model evaluation.

### Stage B — local evidence

Compare:

- geometry;
- radial CT;
- geometry + radial CT.

Report:

- AUROC;
- AUPRC;
- validation-selected threshold;
- recall;
- FPR;
- precision;
- TP / FP / FN / TN.

### Stage C — patient heterogeneity

Report per patient, not just pooled rows.

The six-case study showed strong heterogeneity, so a pooled metric alone is not acceptable.

### Stage D — patient-cluster uncertainty

Bootstrap at patient level. Rows from the same patient are correlated and must not be treated as independent subjects.

### Stage E — Graph-LIRA insertion

Only after the local representation passes the cross-patient check:

- refit image-conditioned presence calibration on train;
- select/calibrate on validation;
- leave geometry identity ranking fixed;
- insert CT presence evidence into PAIR / JUNCTION / BOTH / NONE;
- run the same global Graph-LIRA optimizer.

### Stage F — frozen selective policy

Apply:

- `tau = 0.85`;
- consistency `>= 0.60`.

No test retuning.

Then measure:

- structural exact;
- false structural repair;
- incomplete-but-nonfalse;
- coverage;
- false among accepted;
- exact among accepted;
- repair-needed exact separately from correct NO-REPAIR.

### Stage G — hard anatomy

Specifically inspect:

- LAD;
- LCX;
- degree >= 4 junctions;
- wrong-branch cases near bifurcations.

This is where geometry-only residual errors were concentrated.

## 14. Decision tree after the result

### Outcome 1 — radial CT transfers and Graph-LIRA false rate stays controlled

Then radial CT becomes the new image baseline.

Next scientific step:

```text
radial CT baseline
vs
compact ANZA local encoder
```

ANZA must show incremental value over the simple radial signal under the same frozen patient split and false-repair budget.

After that, consider full cross-section tokens / sequence context.

### Outcome 2 — radial CT is locally good but Graph-LIRA calibration is still unstable

Do **not** jump to a bigger network.

First investigate:

- patient-wise HU normalization / contrast normalization;
- calibration by patient-independent validation;
- stronger negative diversity;
- confidence calibration;
- relation-specific calibration for pair vs junction;
- more matched patients.

The failure would be a calibration/generalization problem, not proof that more capacity is needed.

### Outcome 3 — radial CT no longer improves the expanded local relation problem

Then the six-case result was too small / cohort-specific.

Move to representations that preserve more cross-sectional structure:

1. candidate-aligned 3-D tube;
2. full cross-section CNN tokens;
3. sequence aggregation only after local image encoding;
4. ANZA as an explicit local encoder ablation.

### Outcome 4 — CT reduces false links but lowers repair recall

This is still scientifically useful.

The intended system is selective. A CT module can be valuable as a **veto / safety cue** if the risk-coverage tradeoff improves.

But this must be reported as safety/abstention, not as improved repair recall.

## 15. Where ANZA fits

ANZA is not being abandoned.

It is being moved to the scientifically correct place in the sequence.

We already know:

- local geometry is meaningful;
- simple radial CT carries signal;
- high-capacity 3-D/CNN variants were unstable with too few matched training patients.

Therefore ANZA should be tested only after the expanded cohort tells us whether the bottleneck is representation or calibration.

The clean ablation later is:

```text
same candidate pairs
same train/val/test patients
same Graph-LIRA
same tau
same consistency
same false budget

radial hand-crafted CT evidence
vs
compact conventional CNN encoder
vs
compact ANZA encoder
```

That lets us say what ANZA actually adds, instead of attributing an improvement to a simultaneous change in data, representation, and graph policy.

## 16. Where the research line is heading

The intended end system is:

```text
segmentation
 -> candidate endpoints / junctions
 -> geometry proposal
 -> CCTA local context
 -> relation existence / branch identity
 -> global Graph-LIRA compatibility
 -> calibrated uncertainty / abstention
 -> max-min path
 -> repaired vascular tree
```

The publication-level novelty is the **risk-controlled structural decision pipeline**, not merely a new convolution.

The strongest eventual comparison should answer:

> At a fixed false structural repair budget, how many real repair-needed scenes can each evidence source safely resolve?

That is stronger and more relevant than reporting only Dice or pair AUROC.

## 17. Current execution status in ChatGPT environment

The runner has actually been executed as far as the available data allow.

Verified here:

- Python source compiles;
- frozen plan loads;
- 1,360 rows confirmed;
- 28 unique patients confirmed;
- split = 17 train / 5 validation / 6 test;
- no patient split leakage;
- exact label balance confirmed: 379/379 train, 134/134 val, 167/167 test;
- runner starts with the intended 28 patient IDs.

The current ChatGPT container then stops at the first raw-data gate:

`FileNotFoundError: /mnt/data/801-1000.z04`

This is a genuine external data blocker, not a code/test failure. The runner bundle and annotations are available here, but the raw Kaggle `801-1000.z04` and final `801-1000.zip` bytes are not mounted in this environment.

No result metrics are fabricated past this point.

## 18. Exact continuation point

As soon as those two raw split-archive pieces are accessible in the execution environment:

1. run the existing runner unchanged;
2. freeze the 28-case local result;
3. add patient-cluster bootstrap;
4. refit only the CT relation-presence/calibration layer;
5. insert into frozen Graph-LIRA;
6. apply `tau=0.85` and consistency `0.60`;
7. analyze hard LAD/LCX / high-degree strata;
8. only then decide whether to run the ANZA encoder ablation.
