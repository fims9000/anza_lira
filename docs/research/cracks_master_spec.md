Да. Раз два официальных ZIP уже лежат в проекте, **GeoCrack больше не ждём**. Переключаем исследование на CRACKS и доводим его до конца.

Это даже научно удобнее: CRACKS содержит **400 сейсмических разрезов размером 255×701**, а разметки организованы по 35 аннотаторам: 26 новичков, 8 практиков и один эксперт-геофизик. Эксперт разметил 7636 разломов. Разметка имеет три состояния уверенности: confident no-fault, uncertain fault и certain fault. ([[arXiv](https://arxiv.org/abs/2408.11185?utm_source=chatgpt.com)][1]) Два скачанных тобой файла должны быть ровно `images.zip` размером около 94.2 MB и `Fault segmentations.zip` около 20.2 MB; официальные MD5 — `6557236191763af7bd8298ecb136d41e` и `01e1697e886da2079ff3c1967334a7ca`. ([[Zenodo](https://zenodo.org/records/13926822)][2])

Особенно важно: **мы не должны случайно превратить это в обычную “ещё одну сегментацию”**. Смысл новой работы:

**сейсмический разрез → ANZA-LIRA → маска разломов → внутренняя ориентация → скелет → граф → трассы разломов → количественные характеристики**, плюс проверка, связана ли внутренняя геометрическая уверенность модели с местами, где расходятся человеческие разметчики.

Ниже давай кодеру **целиком**. Это заменяет GeoCrack-часть предыдущего ТЗ. Все правила про skills, RTK, малые изменения, reviewer mode, state/evidence, отсутствие промежуточных commits и один финальный commit остаются в силе.

---

# FINAL CODER SPECIFICATION

# ANZA-LIRA × CRACKS

## Seismic Fault Segmentation, Geometry and Trace Extraction

## 0. STATUS AND PRIORITY

GeoCrack experiment is cancelled for the current submission because the official repository is unavailable.

The new primary dataset is:

```text
CRACKS:
Crowdsourcing Resources for Analysis and
Categorization of Key Subsurface Faults
```

Official dataset DOI:

```text
10.5281/zenodo.13926822
```

Two official ZIP files have already been downloaded manually by the user and placed somewhere in the project directory:

```text
images.zip
Fault segmentations.zip
```

DO NOT download them again.

DO NOT use Harvard Dataverse.

DO NOT delete the ZIPs.

DO NOT modify `main`.

DO NOT make intermediate Git commits.

The task is complete only when the real CRACKS experiment has finished and the final validator prints:

```text
CRACKS STUDY STATUS: COMPLETE
```

A synthetic/smoke result is NOT scientific completion.

---

# 1. GIT SAFETY

Before touching code:

```bash
git status
git branch --show-current
git log -5 --oneline
```

Preserve:

```text
main
feature/geocrack-trace-study
```

as untouched history/backup.

The active research branch must be:

```text
feature/cracks-final
```

### If current work is already uncommitted on `feature/geocrack-final`

Do not reset it.

Create external backup:

```bash
mkdir -p /home/lebedeffson/Code/_wip_backups/anza_lira
git diff > /home/lebedeffson/Code/_wip_backups/anza_lira/before_cracks_switch.patch
```

Then rename the current working branch if safe:

```bash
git branch -m feature/cracks-final
```

### If current implementation exists only in commit

```text
0e1ff6b0bc7bf8d541e6a5f98dba8c236ee0b89e
```

create the new working branch from `main` and bring the implementation into the working tree **without committing**:

```bash
git switch main
git switch -c feature/cracks-final
git cherry-pick -n 0e1ff6b0bc7bf8d541e6a5f98dba8c236ee0b89e
git reset
```

Do not lose current uncommitted changes if they exist.

No new commit until final completion.

---

# 2. DO NOT THROW AWAY THE EXISTING GEOCRACK WORK

Reuse the generic pieces already implemented:

```text
trace extraction
axial orientation mathematics
skeleton graph
GeoJSON
metrics
cluster/bootstrap infrastructure
heartbeat/resume
report generation
THESIS_NUMBERS
validator structure
agents
skills
RTK integration
scientific figures
```

Do NOT duplicate those mechanisms under new names.

Refactor them into dataset-neutral utilities only when necessary.

At the end, the final CRACKS pipeline must not depend on GeoCrack-specific assumptions.

Do not mass-refactor the whole repository before the CRACKS vertical slice works.

---

# 3. LOCATE THE TWO USER ZIP FILES

Search only inside the repository/project root and reasonable first-level subdirectories.

Expected exact names:

```text
images.zip
Fault segmentations.zip
```

Do not recursively scan the entire home directory.

Print only:

```text
path
size
MD5
```

Expected official MD5:

```text
images.zip
6557236191763af7bd8298ecb136d41e

Fault segmentations.zip
01e1697e886da2079ff3c1967334a7ca
```

If both hashes match:

```text
CRACKS ARCHIVES: VERIFIED
```

If they do not match:

```text
STOP DATA IMPORT
```

and report actual MD5.

Do not train on archives that fail official checksum validation.

---

# 4. DATA LOCATION

Raw ZIPs may remain where the user put them.

Do not duplicate 114 MB unnecessarily.

Extract into:

```text
data/cracks/
    images/
    annotations/
    manifests/
    splits/
```

Add dataset/generated paths to `.gitignore`.

Also specifically prevent accidental commit of root archives:

```gitignore
/images.zip
/Fault segmentations.zip
/data/cracks/
/results/cracks_study/
```

Do not ignore source code or compact final tables by an overbroad pattern.

---

# 5. VERIFY ARCHIVE STRUCTURE BEFORE WRITING A LOADER

Do not rely only on the paper description.

Inspect the actual verified ZIPs.

Generate:

```text
results/cracks_study/archive_inventory.json
```

containing:

```text
archive MD5
member count
directory names
file extensions
image identifiers
annotation directories
dimensions
channel counts
unique mask colors
missing pairs
duplicate names
corrupt files
```

Expected high-level structure from the official dataset:

```text
images:
    section_XXX.png

annotations:
    novice...
    practitioner...
    expert...
```

But actual archive content is source of truth.

The official repository states that `Fault segmentations.zip` contains 35 annotator directories and label files follow the corresponding section naming convention. ([[GitHub](https://github.com/olivesgatech/CRACKS)][3])

---

# 6. DO NOT ASSUME RGB VALUES OF THE ANNOTATION CLASSES

Official semantics are:

```text
certain no-fault
uncertain fault
certain fault
```

But do not hard-code RGB triples from memory.

Inspect actual expert PNG files and build a color inventory.

Create:

```text
results/cracks_study/expert_color_audit.json
```

For every unique observed RGB value:

```json
{
  "rgb": [0, 0, 0],
  "pixel_count": 0,
  "fraction": 0.0
}
```

Compare actual colors with official visualization/description.

Only after actual archive verification construct class mapping.

If colors cannot be mapped unambiguously:

```text
STOP BEFORE TRAINING
```

with a compact diagnostic.

Never guess label semantics.

---

# 7. PRIMARY TARGET DEFINITION

The primary experiment must use **expert labels only**.

Novice/practitioner labels must not be used to create the training ground truth in the primary experiment.

Primary three-class interpretation:

```text
certain fault       → positive
certain no-fault    → negative
uncertain fault     → ignore
```

This is deliberate.

We do not force uncertain expert pixels to become false negatives or unquestioned positives.

Training loss must support an ignore mask.

Pixel metrics must ignore uncertain expert pixels in the primary evaluation.

Call this target:

```text
expert_strict
```

---

# 8. REQUIRED SENSITIVITY TARGET

Without retraining, perform an additional evaluation:

```text
expert_inclusive
```

where:

```text
certain fault OR uncertain fault → positive
certain no-fault                 → negative
```

This is a sensitivity analysis.

Do NOT select model/hyperparameters using the inclusive test result.

Primary claim is based on `expert_strict`.

Inclusive evaluation shows how conclusions change when uncertain expert faults are counted as positive.

---

# 9. CHECK IF ANY PIXELS HAVE A FOURTH / UNLABELED STATE

The actual archive may contain:

```text
background
transparent
black/unassigned
```

or another state.

If a pixel is not unambiguously one of the three documented semantic classes:

```text
ignore
```

until proven otherwise.

Do not silently convert unknown colors to background.

Record their proportion.

If unknown pixels exceed 1% of expert masks, stop and investigate before training.

---

# 10. DATASET UNIT

One sample is one complete seismic section.

Expected nominal size:

```text
255 × 701
```

Do not resize the entire image to a square.

Do not distort fault geometry.

For models requiring divisibility by powers of two:

```text
255 × 701
→ padded to
256 × 704
```

Use non-destructive padding.

The original valid region mask must be preserved.

Metrics and trace extraction must be performed on the unpadded:

```text
255 × 701
```

domain.

Padding pixels are always ignored.

---

# 11. CRITICAL SPLIT RULE: DO NOT RANDOMLY SPLIT ADJACENT SECTIONS

The sections come from the same 3D seismic volume.

Adjacent section numbers may be highly correlated.

A naïve random 80/10/10 section split is forbidden.

Use a **blocked spatial split with guard regions**.

First determine the actual ordered expert-labeled section IDs.

If the verified data contains the expected ordered sequence sufficiently close to 1…400, use the following pre-declared protocol:

```text
TRAIN:
section 001–260

GUARD:
section 261–280

VALIDATION:
section 281–320

GUARD:
section 321–340

TEST:
section 341–400
```

Expected nominal:

```text
train = 260
validation = 40
test = 60
excluded guard = 40
```

Guard sections are NEVER used for:

```text
training
validation
threshold tuning
test metrics
```

They exist only to reduce direct adjacency between partitions.

---

# 12. IF SECTION IDs ARE MISSING

Do not silently shift boundaries.

Create:

```text
data/cracks/splits/split_manifest.json
```

containing actual IDs.

Assign sections by numeric coordinate intervals, not by row number.

Example:

```text
section_279 missing
```

does NOT cause `section_280` to become the 279th positional sample.

If fewer than:

```text
200 train
25 val
40 test
```

valid expert sections remain, stop and report.

---

# 13. SPLIT AUDIT

Create:

```text
scripts/check_cracks_split.py
```

Check:

```text
train ∩ val = 0
train ∩ test = 0
val ∩ test = 0

train ∩ guard = 0
val ∩ guard = 0
test ∩ guard = 0
```

Also verify:

```text
max(train ID) < min(val ID)
max(val ID) < min(test ID)
```

with the predeclared guard intervals.

After split creation:

```text
test_split.sha256
```

must be frozen.

Any later modification of test IDs must make the experiment fail.

---

# 14. TRAIN-ONLY NORMALIZATION

Determine channel structure from actual images.

Always expose network input as:

```text
[C, H, W]
```

If images are grayscale, do not arbitrarily apply ImageNet RGB normalization.

If architecture requires 3 channels, replicate grayscale only if necessary and record it.

Compute normalization statistics using train sections only.

Save:

```text
data/cracks/manifests/train_normalization.json
```

Validation/test must use frozen train statistics.

---

# 15. TRAINING PATCHES

Do not train exclusively on full 256×704 sections if that causes unnecessary memory cost.

Primary training unit:

```text
256 × 256 crop
```

from padded training sections.

Crop sampler:

```text
70% foreground-aware
30% uniformly random
```

Foreground-aware means the crop is centred near at least one **certain expert fault pixel**.

Do not use validation or test masks for crop policy fitting.

Each training section should contribute multiple stochastic crops across epochs.

Do not pre-save thousands of duplicated PNG crops.

Generate them on the fly.

---

# 16. AUGMENTATION

Allowed for training:

```text
horizontal flip
vertical flip
180-degree rotation
small intensity/contrast jitter
small additive noise if already supported
```

90-degree rotation is permitted only if physical interpretation of section axes is not used downstream during training.

Because the horizontal and vertical axes of seismic sections have different physical interpretation, default to:

```text
horizontal flip
small intensity transformations
```

and do NOT add 90-degree rotations automatically.

Do not use transformations that destroy fault geometry.

No random augmentation in validation or test.

---

# 17. FULL-SECTION EVALUATION

Validation/test inference must run on the complete:

```text
256 × 704 padded section
```

or via deterministic overlap-tile inference if GPU memory requires it.

If tiling is necessary:

```text
overlap >= 64 px
weighted blending
```

No hard seams.

After prediction:

```text
unpad → 255 × 701
```

before metrics.

This is essential because trace connectivity must be evaluated over an entire seismic section, not independent 256-pixel patches.

---

# 18. REUSE EXISTING LOSS

Do not invent a new loss just because the dataset changed.

Use the currently validated segmentation loss already used by the ANZA-LIRA experiment if it supports an ignore mask.

If it does not:

modify it minimally to support:

```text
valid_pixel_mask
```

All compared architectures use the identical loss.

No architecture-specific loss.

---

# 19. PRIMARY ARCHITECTURES

Keep the previously planned comparison.

Three seeds:

```text
41
42
43
```

Primary:

```text
baseline
az_thesis
```

Required full runs:

```text
baseline_seed41
baseline_seed42
baseline_seed43

az_thesis_seed41
az_thesis_seed42
az_thesis_seed43
```

Ablations at seed 42:

```text
az_no_fuzzy
az_no_aniso
attention_unet
```

Total required:

```text
9 runs
```

Do not add more models before these nine are complete.

---

# 20. MODEL FAIRNESS

Across models freeze:

```text
split
training crops policy
augmentations
loss
optimizer
learning rate
epochs
batch size
validation procedure
threshold grid
checkpoint criterion
test evaluator
```

Only architecture may differ.

Record a protocol hash.

All nine runs must contain the same protocol hash except for:

```text
model
seed
```

and explicit architecture-specific parameters.

---

# 21. THRESHOLD

Threshold selection uses validation only.

Example existing grid:

```text
0.30
0.35
...
0.80
```

Primary criterion:

```text
validation Dice on expert_strict
```

If the repository already freezes another criterion for this family of experiments, preserve it unless there is a documented scientific reason to change.

Once selected:

```text
threshold frozen
```

before opening test metrics.

---

# 22. DO NOT LOOK AT TEST DURING DEVELOPMENT

Implement a test lock.

Until:

```text
model selection frozen
AZ tuning frozen
threshold frozen
trace parameters frozen
```

the normal training command must not print test metrics.

Require explicit:

```text
--unlock-test
```

only in final evaluation stage.

Record:

```text
test_first_opened_at
config_hash
split_hash
commit/worktree hash
```

in provenance.

---

# 23. BOUNDED VALIDATION-ONLY TUNING

We want a good experiment, but not endless result chasing.

First run:

```text
baseline seed42
az_thesis seed42
```

on validation.

If both train normally and AZ is competitive, continue.

If AZ is clearly worse on validation, allow **one bounded AZ-specific tuning stage**.

Maximum:

```text
6 candidate configurations
```

using seed 42 only.

Do not tune generic training hyperparameters independently by model.

Candidate variables must be parameters that already exist in ANZA-LIRA and have direct mathematical meaning, for example:

```text
number of local modes
anisotropy strength/range
fuzzy gating strength
```

Use the actual parameter names from the existing code.

Do not invent parameter names.

Before running the search, save the entire candidate grid in:

```text
configs/cracks_az_tuning.yaml
```

After the first candidate is evaluated, the grid is immutable.

Selection criterion:

```text
primary:
validation Dice

tie-break:
validation clDice
```

Freeze winning AZ config.

Then run seeds 41/42/43.

Never tune on test.

---

# 24. STOP TUNING CONDITION

Do NOT keep tuning until ANZA-LIRA wins.

After the maximum six predeclared validation candidates:

```text
freeze best validation configuration
```

even if baseline remains better.

Negative results are acceptable.

Scientific completeness is based on protocol quality, not mandatory victory.

---

# 25. SANITY FAILURE VS NEGATIVE RESULT

If model produces something equivalent to a trivial predictor, for example:

```text
all background
all foreground
loss not decreasing
non-finite gradients
```

that is a software/optimization failure and must be debugged.

If model trains normally but scores lower than baseline:

```text
that is a scientific result
```

Do not “debug away” a legitimate negative result.

---

# 26. PIXEL METRICS

Primary `expert_strict` metrics:

```text
Dice
IoU
Precision
Recall
Specificity
Balanced Accuracy
```

Only valid, non-ignored pixels.

Secondary:

```text
expert_inclusive Dice
expert_inclusive IoU
expert_inclusive Precision
expert_inclusive Recall
```

Do not mix strict and inclusive numbers in one unlabeled table.

---

# 27. TOPOLOGY METRICS

Required on full sections:

```text
clDice
skeleton precision
skeleton recall
skeleton F1
symmetric skeleton distance
```

Skeleton metrics should use tolerance radii:

```text
1 px
2 px
3 px
```

Primary report:

```text
2 px
```

Other radii in sensitivity table.

---

# 28. ORIENTATION FROM ANZA-LIRA

Reuse the already implemented axial orientation method.

For modes (r):

[
C(p)=\sum_r \mu_r(p)\cos(2\theta_r(p)),
]

[
S(p)=\sum_r \mu_r(p)\sin(2\theta_r(p)).
]

Then:

[
\bar{\theta}(p) =

\frac{1}{2}
\operatorname{atan2}(S(p),C(p)).
]

Orientation coherence:

[
\rho(p) =

\frac{
\sqrt{C(p)^2+S(p)^2}
}{
\sum_r \mu_r(p)+\varepsilon
}.
]

Required invariant:

[
\theta \equiv \theta+\pi.
]

Unit test it.

---

# 29. ANISOTROPY STRENGTH

Reuse:

[
a_r(p)
======

\tanh
\left(
\left|
\log
\frac{\sigma_{u,r}(p)}
{\sigma_{s,r}(p)}
\right|
\right).
]

Aggregate:

[
A(p)
====

\rho(p)
\frac{
\sum_r\mu_r(p)a_r(p)
}{
\sum_r\mu_r(p)+\varepsilon
}.
]

Check:

```text
0 <= rho <= 1
0 <= A <= 1
```

and no NaN/Inf.

---

# 30. GROUND-TRUTH ORIENTATION

Derive local reference orientation from the expert fault skeleton.

For each matched GT skeleton point, estimate tangent direction by PCA over skeleton coordinates in a local geodesic or spatial neighbourhood.

Primary radius:

```text
5 px
```

Sensitivity:

```text
3 px
7 px
```

Axial angular error:

[
d_\pi(\theta_1,\theta_2) =

\frac12
\arccos
\left[
\cos 2(\theta_1-\theta_2)
\right].
]

Report:

```text
median
mean
90th percentile
```

degrees.

Do not compute orientation error on pixels where reference tangent is geometrically undefined.

---

# 31. MASK → FAULT TRACE GRAPH

On the full 255×701 prediction:

```text
probability map
→ threshold
→ skeleton
→ 8-connected graph
```

Skeleton vertex degree:

```text
degree 1 → endpoint
degree 2 → ordinary trace
degree >= 3 → junction
```

A raw trace segment is a path between:

```text
endpoint ↔ endpoint
endpoint ↔ junction
junction ↔ junction
```

---

# 32. DO NOT CLAIM RAW PNG CONTAINS INSTANCE IDS

Before writing object-level claims, inspect the labels.

If expert PNG provides only semantic/confidence colors and no fault-instance identity:

the extracted entities must be called:

```text
fault trace segments
candidate fault traces
trace graph branches
```

Do NOT claim:

```text
each extracted trace equals one uniquely identified geological fault
```

unless the archive actually provides instance-level ground truth and this is verified.

---

# 33. EDGE CONFIDENCE

Reuse geometry-aware trace confidence.

For neighbour edge (p,q):

[
\phi_{pq} =

\operatorname{atan2}
(y_q-y_p,x_q-x_p).
]

Axial geometric compatibility:

[
G(p,q)
======

\frac{
1+\cos 2(\bar\theta(p)-\phi_{pq})
}{2}.
]

A valid edge-confidence formulation is:

[
S(p,q)
======

\sqrt{P(p)P(q)}
\sqrt{\rho(p)\rho(q)}
\sqrt{A(p)A(q)}
G(p,q)G(q,p).
]

Require:

[
0\le S(p,q)\le1.
]

If the current implementation already has an equivalent verified formula, reuse it instead of duplicating.

---

# 34. JUNCTION PAIRING

At intersections, do not merge branches arbitrarily.

Estimate local branch tangent using first:

```text
5 skeleton pixels
```

from a junction.

Use axial orientation compatibility.

Trace joining parameters tuned using validation only.

Predeclare small grid, for example:

```text
max gap:
1, 2, 3 px

max axial angle:
10°, 20°, 30°

minimum branch length:
5, 10 px
```

Do not search a huge combinatorial space.

Primary validation criterion:

```text
0.7 * skeleton_F1
+
0.3 * endpoint_F1
```

Freeze before test.

---

# 35. FULL-SECTION TRACE EXPORT

For every test section:

```text
results/cracks_study/traces/section_XXX.geojson
```

Each feature:

```json
{
  "type": "Feature",
  "geometry": {
    "type": "LineString"
  },
  "properties": {
    "trace_id": 0,
    "section_id": 0,
    "pixel_length": 0.0,
    "chord_length": 0.0,
    "sinuosity": 0.0,
    "orientation_deg": 0.0,
    "orientation_coherence": 0.0,
    "mean_probability": 0.0,
    "mean_anisotropy": 0.0,
    "confidence": 0.0,
    "start_type": "",
    "end_type": ""
  }
}
```

Do not claim physical length in metres because pixel-to-physical scale must not be invented.

Use:

```text
pixel_length
```

unless the dataset gives explicit spacing and it is verified.

---

# 36. TRACE METRICS

Because semantic masks do not necessarily provide fault instance IDs, primary trace evaluation is geometry based.

Required:

```text
skeleton precision @2px
skeleton recall @2px
skeleton F1 @2px

endpoint F1 @3px
junction F1 @3px

symmetric skeleton distance
orientation error
total skeleton length error
connected-component fragmentation
```

Also:

```text
number of predicted trace branches
number of GT trace branches
```

but label this as graph segmentation statistics, not exact instance accuracy.

---

# 37. FRAGMENTATION METRIC

Introduce a simple interpretable structural metric.

For each GT connected skeleton component:

count how many predicted connected components intersect its 2-px dilation.

Define:

[
F_{\mathrm{frag}} =

\frac{1}{N}
\sum_{i=1}^{N}
\max(0,n_i-1).
]

Lower is better.

Also report:

```text
fraction of GT components represented by exactly one predicted component
```

This directly measures broken faults.

---

# 38. HUMAN DISAGREEMENT ANALYSIS — SECONDARY, CHEAP, REQUIRED

CRACKS has the major advantage that multiple annotators labelled the same seismic volume.

Use this without retraining.

Do not use crowd labels as primary training truth.

For each available test section and each pixel, among available non-expert annotations compute:

```text
N_valid
fraction fault_any
fraction fault_certain
3-class categorical entropy
```

where class semantics follow verified archive color mapping.

Do not assign arbitrary numeric confidence weights such as:

```text
certain=1
uncertain=0.5
```

for the primary disagreement measure.

Use categorical entropy.

---

# 39. HUMAN DISAGREEMENT MAP

For pixel class frequencies (p_c):

[
H(p)
====

-\frac{
\sum_c p_c(p)\log(p_c(p)+\varepsilon)
}{
\log K
},
]

where (K) is the number of actually represented semantic classes.

Thus:

```text
0 <= H <= 1
```

Only compute where at least:

```text
5 non-expert annotators
```

have a valid label.

Otherwise mark:

```text
insufficient_annotations
```

---

# 40. MODEL GEOMETRY VS HUMAN DISAGREEMENT

This is a secondary diagnostic and potentially one of the strongest thesis results.

On test sections evaluate whether high human disagreement is associated with:

```text
higher model error
lower orientation coherence rho
different anisotropy strength
lower predicted confidence
```

Do NOT use millions of pixels as independent samples for significance tests.

Aggregate first by section.

For each test section calculate:

```text
mean human entropy
prediction error rate
mean rho on fault neighbourhood
mean anisotropy
Dice
clDice
```

Then report section-level Spearman correlations.

Bootstrap by section.

Interpret association only.

Do not claim causality.

---

# 41. NOVICE/PRACTITIONER SUBGROUP ANALYSIS

If easy from directory names, compute additional descriptive maps:

```text
novice disagreement
practitioner disagreement
expert-vs-crowd disagreement
```

No retraining required.

This analysis is secondary.

It must not block the primary 9-run experiment if unexpected annotation sparsity makes it difficult.

But the basic combined non-expert entropy analysis IS required.

---

# 42. STATISTICAL UNIT

Sections are the independent evaluation unit for uncertainty summaries.

Do not treat:

```text
pixels
skeleton points
patches
```

as independent observations for bootstrap CI.

For each seed:

1. compute metrics per test section;
2. aggregate seeds within section where appropriate;
3. bootstrap test **sections**.

Primary bootstrap:

```text
2000 replicates
```

Report:

```text
AZ - baseline mean difference
95% bootstrap CI
```

for:

```text
Dice
IoU
clDice
skeleton F1
fragmentation
orientation error
```

For metrics where lower is better, make sign convention explicit.

---

# 43. CROSS-SEED STATISTICS

Do not inflate sample size by pretending:

```text
3 seeds × 60 sections = 180 independent sections
```

Primary uncertainty resamples sections.

Seeds represent training variability.

Report:

```text
mean across seeds
std across seeds
section-bootstrap CI of paired model difference
```

Paired comparison must use the same test section.

---

# 44. COMPUTATIONAL BUDGET

Dataset is small.

Do not create a giant hyperparameter campaign.

Default:

```text
30 epochs
```

with best validation checkpoint.

If learning clearly still improves at epoch 30, allow one documented extension:

```text
maximum 50 epochs
```

applied consistently to the primary architectures.

Do not silently give ANZA-LIRA more epochs than baseline.

Use early stopping only if applied identically.

---

# 45. FAILURE DETECTION

Automatically detect:

```text
loss NaN
gradient NaN
all-background prediction
all-foreground prediction
zero positive validation prediction
checkpoint corruption
test split mutation
protocol hash mismatch
```

Such runs are:

```text
FAILED
```

not valid negative scientific results.

Repair root cause and rerun.

---

# 46. REQUIRED UNIT TESTS

Keep all old generic tests.

Add CRACKS-specific tests:

```text
test_cracks_archive_hash.py
test_cracks_inventory.py
test_cracks_label_mapping.py
test_cracks_split.py
test_cracks_ignore_mask.py
test_cracks_padding.py
test_cracks_full_section_inference.py
test_human_disagreement.py
```

Required exact behaviours include:

```text
theta == theta + pi
perfect prediction → Dice 1
perfect prediction → clDice approx 1
perfect skeleton → skeleton F1 1
perfect orientation → angular error approx 0
uncertain expert pixels do not affect strict loss
test IDs cannot be altered after freeze
```

---

# 47. SMOKE TEST BEFORE REAL TRAINING MATRIX

Use the real CRACKS files, but a tiny subset of **training IDs only**.

Do not touch actual test IDs in smoke.

Example:

```text
8 train sections
4 temporary pseudo-validation sections
```

These must all come from the TRAIN interval.

Do not use actual val/test for development smoke.

Run:

```text
2 epochs baseline
2 epochs AZ
```

Full vertical slice:

```text
load
→ train
→ checkpoint
→ reload
→ inference
→ unpad
→ metrics
→ geometry
→ skeleton
→ graph
→ GeoJSON
→ table
→ figure
→ report
```

Only after PASS run real split.

---

# 48. REAL EXECUTION ORDER

Required order:

```text
1. archive verification
2. archive inventory
3. label semantic audit
4. split creation
5. split audit
6. normalization
7. unit tests
8. training-only smoke
9. baseline seed42
10. AZ seed42
11. bounded AZ validation tuning if needed
12. freeze AZ
13. freeze threshold/tracing parameters
14. baseline seeds 41/43
15. AZ seeds 41/43
16. ablations
17. unlock test
18. all final test inference
19. trace analysis
20. human disagreement analysis
21. bootstrap
22. figures
23. final report
24. thesis evidence
25. final validator
26. scientific reviewer pass
27. single Git commit
```

Do not reorder by convenience if it exposes test information early.

---

# 49. ONE-COMMAND RUN

Primary Linux entry point:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/cracks_study.py full
```

Thin shell wrapper:

```text
scripts/run_cracks_full_study.sh
```

PowerShell wrapper optional, but Linux is primary.

Business logic belongs in Python.

The orchestrator must be resumable.

Already completed valid phases must be:

```text
SKIP
```

not repeated.

---

# 50. HEARTBEAT

Long training must not flood the agent context.

Each active run writes:

```text
heartbeat.json
```

with only:

```json
{
  "model": "az_thesis",
  "seed": 42,
  "epoch": 12,
  "max_epoch": 30,
  "val_dice": 0.0,
  "best_val_dice": 0.0,
  "best_epoch": 0,
  "status": "RUNNING"
}
```

No batch-level output to agent context.

Full logs go to disk.

---

# 51. USE RTK / COMPACT COMMAND OUTPUT

Do not print:

```text
full pytest trace when tests pass
full training logs
full recursive tree
full git diff
all JSON results
```

Use compact summaries.

Only inspect targeted failures.

After every stage:

```text
targeted test
→ targeted diff
→ reviewer gate
→ state update
```

---

# 52. REQUIRED RESULT TABLE

Generate automatically:

```text
results/cracks_study/tables/model_comparison.csv
```

Columns:

```text
model
seed

strict_dice
strict_iou
strict_precision
strict_recall

inclusive_dice
inclusive_iou

cldice

skeleton_precision_2px
skeleton_recall_2px
skeleton_f1_2px

endpoint_f1_3px
junction_f1_3px

fragmentation
orientation_error_median_deg
orientation_error_mean_deg

parameter_count
inference_ms
```

No manually typed scientific values.

---

# 53. ABLATION TABLE

Generate:

```text
ablation_comparison.csv
```

Rows:

```text
baseline
attention_unet
az_no_fuzzy
az_no_aniso
az_full
```

Seed 42.

This table answers:

```text
Does directional geometry matter?
Does fuzzy agreement matter?
Does their joint use matter?
```

No stronger causal conclusion than the controlled architecture comparison supports.

---

# 54. FIGURE 1 — TASK AND QUALITATIVE RESULT

Automatically select the **median AZ-baseline Dice-delta** test section.

Panels:

```text
A. seismic section
B. expert strict ground truth
C. baseline prediction
D. ANZA-LIRA prediction
```

White/neutral publication style.

No AI decorative graphics.

---

# 55. FIGURE 2 — ERROR DIFFERENCE

Same median example.

Show:

```text
GT fault
both correct
AZ recovered
baseline-only correct
AZ new false positive
AZ new false negative
```

Clear compact external legend.

Do not use neon palette.

---

# 56. FIGURE 3 — GEOMETRY

Panels:

```text
A. seismic section + fault prediction
B. ANZA-LIRA local orientation axes
C. orientation coherence rho
D. anisotropy strength
E. extracted fault trace graph
```

Do not plot an orientation glyph at every pixel.

Sample axes sparsely.

---

# 57. FIGURE 4 — QUANTITATIVE COMPARISON

Point + interval plot.

Metrics:

```text
Dice
clDice
skeleton F1
fragmentation
orientation error
```

Models:

```text
baseline
full ANZA-LIRA
```

Separate ablation figure/table if space permits.

No bar chart by default.

---

# 58. FIGURE 5 — HUMAN DISAGREEMENT

If data coverage is sufficient:

Panels:

```text
expert label
human disagreement entropy
ANZA-LIRA error
orientation coherence
```

Plus compact section-level plot:

```text
human entropy
vs
model error / rho
```

This figure is secondary but potentially valuable for the thesis.

---

# 59. EXAMPLE SELECTION MUST BE AUTOMATIC

Create:

```text
median case
best AZ delta
worst AZ delta
```

Primary article figure uses:

```text
median case
```

Best/worst are supplementary diagnostics.

Do not cherry-pick article example manually.

---

# 60. OUTPUT TREE

Final:

```text
results/cracks_study/
    provenance.json
    archive_inventory.json
    expert_color_audit.json
    split_report.json
    environment.txt
    protocol.json
    test_unlock.json

    runs/
        baseline_seed41/
        baseline_seed42/
        baseline_seed43/
        az_thesis_seed41/
        az_thesis_seed42/
        az_thesis_seed43/
        az_no_fuzzy_seed42/
        az_no_aniso_seed42/
        attention_unet_seed42/

    tables/
        model_comparison.csv
        model_mean_std.csv
        ablation_comparison.csv
        bootstrap_comparison.csv
        strict_vs_inclusive.csv
        human_disagreement.csv
        trace_metrics.csv

    traces/
        section_*.geojson

    figures/
        fig1_segmentation.*
        fig2_error_difference.*
        fig3_geometry.*
        fig4_metrics.*
        fig5_disagreement.*
        best_case.*
        worst_case.*

    THESIS_NUMBERS.json
    THESIS_EVIDENCE.md
    FINAL_REPORT.md
```

---

# 61. THESIS_NUMBERS.JSON

All future thesis numbers come from this file.

Minimum:

```json
{
  "dataset": {},
  "archive_validation": {},
  "label_semantics": {},
  "split": {},
  "training": {},
  "baseline": {},
  "anza_lira": {},
  "delta": {},
  "ablations": {},
  "strict_vs_inclusive": {},
  "trace_metrics": {},
  "orientation": {},
  "human_disagreement": {},
  "bootstrap": {},
  "runtime": {},
  "limitations": []
}
```

No scientific number may be manually copied into this JSON.

It is generated from experiment artifacts.

---

# 62. THESIS_EVIDENCE.MD

This file is NOT the final paper.

It is a fact sheet for scientific writing.

For every prospective claim:

```text
CLAIM
EVIDENCE
SOURCE FILE
METRIC
UNCERTAINTY
LIMITATION
ALLOWED WORDING
FORBIDDEN OVERCLAIM
```

Example:

```text
CLAIM:
ANZA-LIRA reduced fragmentation.

EVIDENCE:
mean paired delta = ...
95% CI = ...

ALLOWED:
The proposed model produced fewer fragmented
fault traces under the fixed test protocol.

FORBIDDEN:
ANZA-LIRA reconstructs geological faults correctly.
```

---

# 63. FINAL_REPORT.MD

Required sections:

```text
1. Research question
2. CRACKS data
3. Annotation semantics
4. Why expert labels are primary
5. Strict and inclusive targets
6. Spatial blocked split
7. Leakage/adjacency control
8. Model architecture
9. ANZA-LIRA mathematics
10. Training protocol
11. Pixel segmentation results
12. Topology results
13. Trace extraction
14. Orientation analysis
15. Ablations
16. Human disagreement analysis
17. Statistical uncertainty
18. Best/median/worst cases
19. Failures
20. Limitations
21. What can be claimed
22. What cannot be claimed
23. Reproduction commands
```

---

# 64. IMPORTANT LIMITATIONS TO RECORD

At minimum:

```text
one seismic volume
section-level rather than independent-field validation
spatial dependence may remain despite guard blocks
expert annotation still contains uncertainty
raw semantic masks may not encode unique fault instance IDs
trace extraction is derived from segmentation/skeleton geometry
pixel length is not physical fault length
results do not establish geological interpretation outside CRACKS/F3
```

Do not hide these.

---

# 65. SCIENTIFIC STORY MUST REMAIN DISTINCT FROM PRIOR WORK

Do not turn the thesis into:

```text
a repeat of the Kolomna operator-properties paper
```

and do not repeat:

```text
medical vessel segmentation from Tashkent
```

The new scientific story is:

```text
ANZA-LIRA in seismic fault interpretation
+
preservation of fault-trace topology
+
model-native orientation geometry
+
trace graph extraction
+
relation to human annotation disagreement
```

The Kolomna work may be cited as the mathematical origin of the operator.

The Tashkent work may be cited only if scientifically necessary.

Do not copy paragraphs.

---

# 66. NO ERGODICITY CLAIM

Do NOT introduce:

```text
ergodic ANZA-LIRA
Anosov system
ergodic convolution
measure-preserving fault mapping
```

into this experiment.

The current study is about:

```text
directional local aggregation
fault segmentation
trace geometry
```

The ergodicity question remains a separate future mathematical project.

---

# 67. PRE-FINAL SCIENTIFIC REVIEWER MODE

After all results are generated:

switch from IMPLEMENTER MODE to:

```text
SCIENTIFIC REVIEWER MODE
```

No code changes initially.

Ask:

```text
Was test hidden until freeze?
Are all model conditions matched?
Did uncertain expert pixels leak into negatives?
Was section adjacency handled?
Do traces correspond to what we claim?
Are bootstrap units sections?
Were negative metrics omitted anywhere?
Do figures match the actual tables?
Does THESIS_NUMBERS match source CSV?
Is any claim stronger than evidence?
```

Write:

```text
results/cracks_study/SCIENTIFIC_AUDIT.md
```

Only then repair genuine issues.

---

# 68. FINAL VALIDATOR

Create/adapt:

```bash
python scripts/validate_cracks_study.py --phase final
```

It must fail unless ALL are true:

```text
official ZIP MD5 PASS
archives readable
expert mapping verified
split PASS
test hash frozen
unit tests PASS
smoke PASS
6 primary seed runs PASS
3 ablation runs PASS
strict metrics finite
inclusive metrics finite
topology metrics finite
orientation metrics finite
trace GeoJSON valid
human disagreement analysis complete
bootstrap complete
figures complete
THESIS_NUMBERS complete
THESIS_EVIDENCE complete
FINAL_REPORT complete
SCIENTIFIC_AUDIT PASS
no TODO/FIXME in research path
no NaN/Inf
no fake/synthetic value in real tables
```

Final output exactly:

```text
CRACKS ARCHIVES: VERIFIED
CRACKS LABEL SEMANTICS: VERIFIED
CRACKS SPLIT: VERIFIED
CRACKS TEST LOCK: VERIFIED
CRACKS RUN MATRIX: COMPLETE
CRACKS STATISTICS: COMPLETE
CRACKS SCIENTIFIC AUDIT: PASS

CRACKS STUDY STATUS: COMPLETE
```

---

# 69. STOP CONDITIONS

Do NOT stop for normal engineering failures:

```text
path issue
tensor shape
OOM
one bad visualization
checkpoint resume bug
missing dependency
mask conversion bug
test failure
```

Fix root cause and continue.

If the same defect survives two repair attempts:

```text
STOP PATCHING
→ reproduce minimally
→ diagnose root cause
→ verify diagnosis
→ make one repair
```

A negative scientific result is NOT a blocker.

---

# 70. AGENT CONTEXT RULE

Do not keep this entire specification loaded in working context.

Store it at:

```text
docs/research/cracks_master_spec.md
```

Existing:

```text
AGENTS.md
.cursor/rules
.agents/skills
TASK_STATE.json
EVIDENCE.json
RTK
```

remain the execution mechanism.

Create/adapt only the necessary CRACKS skills:

```text
cracks-data
cracks-experiment
trace-extraction
human-disagreement
statistical-validation
thesis-evidence
final-scientific-audit
```

First inspect existing skills and extend them instead of duplicating equivalents.

Per phase:

```text
AGENTS.md
→ TASK_STATE
→ one relevant skill
→ targeted files only
```

---

# 71. NO INTERMEDIATE COMMITS

During all work:

```text
NO git commit
NO push
```

After every major verified phase:

```bash
git diff > /home/lebedeffson/Code/_wip_backups/anza_lira/cracks_latest.patch
```

Also update agent state/evidence.

Only after:

```text
CRACKS STUDY STATUS: COMPLETE
```

run:

```bash
pytest
git diff --check
git status
git diff --stat
```

Review full diff relative to `main`.

Remove only clearly dead experimental junk from the final branch.

Do not delete reusable verified infrastructure.

---

# 72. EXACTLY ONE FINAL COMMIT

Only when every final gate passes:

```bash
git add <reviewed files>
git commit -m "Add CRACKS seismic fault segmentation and trace study"
```

That must be the **only new final CRACKS implementation commit**.

Do not push unless explicitly instructed by the user.

Report:

```text
branch
commit hash
tests
run matrix
final validator status
primary scientific results
limitations
exact files for thesis writing
```

---

## И ещё одна важная вещь для кодера

Перед тем как он вообще начнёт обучение, пусть **сначала посмотрит реальные expert masks из ZIP и выдаст короткий DATA AUDIT**. Потому что именно семантику RGB-классов я намеренно не разрешаю ему угадывать: официально известно значение трёх классов, но конкретную кодировку в скачанном PNG нужно подтвердить непосредственно по данным. Официальный репозиторий подтверждает три цветовых состояния и структуру `expert / practitioner / novice`. ([[GitHub](https://github.com/olivesgatech/CRACKS)][3])

И split я специально поменял по сравнению с GeoCrack. Здесь 400 изображений — **не 400 независимых фотографий**, а последовательные сечения одного Netherlands North Sea subsurface volume. ([[arXiv](https://arxiv.org/abs/2408.11185?utm_source=chatgpt.com)][1]) Поэтому случайный shuffle дал бы очень слабую научную проверку. Наш блокированный test с двумя 20-срезовыми буферами намного честнее, хотя и он не превращает test в независимый геологический район — это мы прямо укажем как ограничение.

А secondary-анализ с 34 неэкспертными разметчиками может дать работе действительно хороший дополнительный смысл: модель не просто «чуть лучше Dice», а мы сможем проверить, **становится ли её внутренняя ориентационная геометрия менее уверенной там, где сами люди расходятся в интерпретации разлома**. CRACKS как раз специально создавался с уровнями expertise и annotation confidence. ([[Alregib](https://alregib.ece.gatech.edu/software-and-datasets/cracks-crowdsourcing-resources-for-analysis-and-categorization-of-key-subsurface-faults/?utm_source=chatgpt.com)][4])

**Вот этот вариант я бы уже запускал.** Датасет маленький, официальный, хэши известны, сейсмическая постановка самостоятельная, и почти вся уже написанная нами инфраструктура реально переиспользуется.

[1]: https://arxiv.org/abs/2408.11185?utm_source=chatgpt.com "CRACKS: Crowdsourcing Resources for Analysis and Categorization of Key Subsurface faults"
[2]: https://zenodo.org/records/13926822 "CRACKS: Crowdsourcing Resources for Analysis and Categorization of Key Subsurface faults | Zenodo"
[3]: https://github.com/olivesgatech/CRACKS "GitHub - olivesgatech/CRACKS · GitHub"
[4]: https://alregib.ece.gatech.edu/software-and-datasets/cracks-crowdsourcing-resources-for-analysis-and-categorization-of-key-subsurface-faults/?utm_source=chatgpt.com "CRACKS: Crowdsourcing Resources for Analysis and Categorization of Key Subsurface faults – Ghassan AlRegib"
