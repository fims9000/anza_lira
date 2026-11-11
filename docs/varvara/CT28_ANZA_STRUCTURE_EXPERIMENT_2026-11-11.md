# CT28 CCTA3D: ANZA structural architecture audit and Varvara handoff

**Research branch:** `experiment/ccta3d-topology-anisotropy-2026-11-11` (draft PR #24).
**Status:** experimental development, no new trained ImageCAS CT28 outcome yet.
**Never merge as "ANZA beats CNN" from numerical or phantom tests alone.**
The frozen K10/15, five development-validation subjects and six unopened held-out
CT28 subjects are unchanged. No medical CT volume is committed to the repository.

## 1. Confirmed audit findings and why we changed the model

1. **Merged junction endpoints:** among the 72 true JUNCTION candidates in
   frozen CT28 validation, **62 have a closest pair of endpoints below 3 mm**.
   The old `_peak_centers` decodes one *summed* Gaussian endpoint heatmap
   and suppresses every local maximum inside a **3-voxel** neighborhood.
   It may therefore substitute a wrong third end. In contrast, none of
   the 170 true PAIR candidates have a closest pair below 3 mm.
2. **Independent geometry example:** on three 3D Gaussian tips 1.9–2.5 mm
   apart, the original summed-heatmap peak search gives a best-matched
   worst-tip position error near 1.98 mm. Independent individual-channel
   Gaussian moments yield error below 0.001 mm on this **ideal noise-free**
   example; this is not a medical localization measurement.
3. **Moving frame:** original canonical frame can flip/rotate sharply under
   0.8 mm endpoint perturbations. The opt-in fixed LPS frame avoids this
   discontinuity. Endpoint channel order is still permutation invariant
   at the model level.
4. **Phantom label integrity:** prior synthetic y=0 JUNCTION cases sometimes
   contained a fully connected 26-neighbor ground-truth vessel. The generator
   now rejects and resamples such contradictory samples, and fails if the
   oracle topology disagrees with its target. Previous scores from the
   defective generator cannot be compared as if they were the corrected data.
5. **Synthetic shortcuts:** a local independently reproduced close-JUNCTION
   generator test (96 train, 96 validation examples) showed that **mean HU
   alone can reach ~0.71 AUROC** without tracing any vessel. The exact model
   workflow therefore includes fixed train-oriented scalar baselines,
   voxel-scrambled HU predictions and *identical per-patch HU histogram*
   (rank normalization) stress tests. Rank normalization is synthetic-only,
   not proposed CCTA preprocessing.
6. **Old structural three-seed benchmark:** corrected synthetic JUNCTION
   mean AUROC ANZA-Structure ~0.852 vs CNN-Structure ~0.790 over three
   seeds/12 epochs. PAIR ~0.999 vs ~1.000. Removing ANZA's forced axial
   preference yielded almost identical scores. **This does not prove
   ANZA-specific structural or clinical superiority.**
7. **Connectedness is not identity:** two candidate ends can belong to the
   *same* anatomical vascular tree but represent an incorrect local
   continuation. Fuzzy 26-neighbor connectedness and binary clDice alone
   cannot resolve this. The original ImageCAS-X files may carry multiple
   coronary segment labels: audit actual available labels on TRAIN before
   proposing segment-identity supervision.

### Same-tree false continuation: new adversarial oracle

The first binary-connectivity phantoms were too easy: positive vessels were
connected and negatives often disconnected. This is NOT the full coronary
branch-identity task.

A second diagnostic `audit_connected_branch_oracle.py` now creates one
unchanged, fully 26-connected 3D Y-shaped vessel tree per pair. It compares
an intended straight-trunk continuation with a wrong shallow branch while
**both candidate endpoints are in the same physical component**.
The branch angle is swept over 15, 20, 25, 35, 50, 65, and 90 degrees.
This is an explicit *branch-identity* label, not a connected-component label.

An independent CPU reproduction of the geometry path readout on ideal masks
showed a structural failure: with 15°, 20°, 25°, and 35° wrong branches,
its false-candidate support was respectively around 0.862, 0.988, 1.000
and 0.997, despite the incorrect continuation. A fixed 0.85 candidate
threshold would accept all four wrong shallow branches. This was a
standalone reproduced calculation of the readout equations, **not** a
new medical evaluation or a complete pytest run. The matching GitHub
workflow remains queued for independent verification.

This demonstrates why neither perfect dense vessel segmentation nor a
differentiable path bottleneck can, alone, establish coronary branch
identity. Do not tune the same-tree synthetic angle grid until the results
look favorable; retain these hard negatives as a falsification fixture.
Train-only 3D tangent-orientation/branch-label consistency is a candidate
future architectural extension but MUST be validated against the original
CT28 train-only segment/tangent audit before implementation or claimed
clinical gain.

## 2. Architecture after review

```text
verified original 3D ImageCAS CCTA + frozen CT28 candidate endpoints
    -> 24x24x24 mm CT patch in fixed LPS coordinates
       old control: [HU, summed Gaussian tips] (2 channels)
       new: [HU, tip A, tip B, tip C] (4 channels; PAIR C=0)
    -> shared-weight, permutation-invariant endpoint 3D stem
    -> matched compact CNN / 3-mode AZConv3D ANZA (optional axial-bias ablation)
    -> dense vessel head supervised ONLY on 17 training patients
       balanced BCE + Dice + differentiable clDice
    -> structural bottleneck: PAIR five centerline hypotheses;
       JUNCTION three arms to a shared hub chosen from 27 positions
    -> candidate logits and patient-grouped out-of-fold scores
    -> matched 15-perturbation relation HGB + original thresholds/consistency
    -> path reconstruction + full mask, Dice/clDice and false-bridge accounting
```

The **primary fair comparison** is `conv3d_sep_structure` against
`anza3d_sep_structure`. A third `anza3d_noaxial_sep_structure`
has the same additional segmentation supervision and structure bottleneck
but does not force negative hyperbolicity. Do not compare the new ANZA
with an old CNN lacking matching dense supervision and endpoint channels.

The original `conv3d`, `anza3d` and summed-heatmap structure variants
remain available as ablations; no historical checkpoint is overwritten.

### Security / target provenance

- The four model inputs contain CT and candidate locations only.
- Original vessel reference masks are a **separate** train-only target array
  in ignored local NPZ cache, never concatenated with the network input.
- Train masks must have the exact original verified CT shape and affine.
- `GroupKFold` isolates patients, and perturbed relation HGB train scores
  come from **fold models excluding the scored patient**.
- Five development-validation masks are post-hoc metrics only; held-out
  patients are not opened. All generated files are SHA-bound and create-only.

### Latest stricter three-seed close-tip challenge (verified CI)

GitHub Actions run 43104818037 trained seven matched baselines
for 12 epochs across three fixed seeds, on true JUNCTION tips closer than
3 mm. This is an **unseen synthetic generator**, still not ImageCAS.

| Protocol | CNN separate-tip AUROC | ANZA separate-tip AUROC | ANZA no-axial AUROC |
|---|---:|---:|---:|
| Raw synthetic HU | **1.000** | 0.954 | 0.953 |
| Rank-matched identical HU histograms | **0.953** | 0.902 | 0.899 |

HU-voxel-shuffled control falls to roughly chance, so the models do respond
to spatial evidence. However, at **candidate score >=0.85 none of the
tested models accepted a true or false junction**. This means a strong
candidate ranking alone does not yield useful accepted repairs.

These results supersede earlier easier 0.852-vs-0.790 ANZA-favored
synthetic conclusions. On the better close-tip stress test the CNN wins,
and axial hyperbolicity has no isolated demonstrated benefit.

**Do not change tau=0.85 merely to claim a win.** We added a
strictly **train-only, patient-crossfit and monotone probability calibration
diagnostic**. It preserves ranking (positive slope) and never uses a
development-validation label to fit its scale. Numerical calibration on
a separate synthetic calibration set is now a tested ablation. The
existing 84-feature relation HGB, original K10 and production inference
are NOT automatically changed by this research diagnostic.

After producing the two local patient-OOF model packs, inspect calibration
for both types and both compared architectures:

\`\`\`bash
VARIANT=anza3d_sep_structure
ROOT=varvara/anza_lira_local_results/ct28_sep_fixedlps_seed_20261102

python scripts/research/ccta_graph_lira_safe_repair/audit_ct28_neural_calibration.py \
  --kind PAIR --variant "$VARIANT" \
  --oof-dir "$ROOT/oof_pair_$VARIANT" \
  --out "$ROOT/calibration_pair_$VARIANT"

python scripts/research/ccta_graph_lira_safe_repair/audit_ct28_neural_calibration.py \
  --kind JUNCTION --variant "$VARIANT" \
  --oof-dir "$ROOT/oof_junction_$VARIANT" \
  --out "$ROOT/calibration_junction_$VARIANT"
\`\`\`

Repeat for \`conv3d_sep_structure\` and the other fixed seeds, each in a
new immutable output directory. Outputs include 17-train-patient
leave-one-patient-out calibrated scores, five-val-patient prediction-only
scores, train-only Brier measurements, SHA provenance and an immutable
JSON calibrator. The five-val ground-truth labels are not used in fitting
the calibrator. These **sidecars are not active clinical HGB inputs**;
promotion requires a separately trained relation head using correctly
crossfitted calibrated train features, followed by full 5-patient
mask evaluation at frozen thresholds.

### Nonlinear endpoint identity: why four channels were not yet enough

During the next architecture review we found a precise algebraic limitation
in the previous shared-weight tip stem:

\`\`\`text
   sum_i Conv3d(tip_i) == Conv3d(sum_i tip_i)   (bias=False)
\`\`\`

Thus the previous separate-tip input still collapsed into the old summed
heatmap *before* the CNN or ANZA local operator. Only the downstream
structural bridge used the three individual tip positions. At sufficiently
close true junction ends, losing this multiset information can weaken
branch-identity discrimination.

A new opt-in, **matched for CNN and ANZA** \`DeepSets\` stem computes
\`mean_i GELU(Conv3d(tip_i))\` and a learnable second-moment term
\`tanh(gain) * mean_i GELU(Conv3d(tip_i))^2\` (with a zero-padded PAIR third
tip). These representations are permutation invariant but can distinguish
distinct three-tip configurations with the SAME summed heatmap. The
difference from the previous 4-channel stem is one trainable scalar
gain per model; historical encoders remain unchanged.

New paired variants:
\`conv3d_deepsets_sep_structure\`,
\`anza3d_deepsets_sep_structure\`, and
\`anza3d_noaxial_deepsets_sep_structure\`.
A metamorphic test compares two different positive three-tip distributions
with identical summed maps: the old stem must produce equal features,
the new stem must produce different ones, and permuting the tips must
produce identical features. This is an **expressiveness** test, not proof
of superior clinical or phantom accuracy. New synthetic raw/rank
three-seed runs and public cardiac CTA numerical smoke must finish before
scientific conclusions.

To train the matched nonlinear-head variants on the authorized ImageCAS
workstation use the same cache, seed, fold and hardware constraints:

\`\`\`bash
DEVICE=mps
SEED=20261102
CACHE=varvara/anza_lira_local_results/ct28_sep_fixedlps_trainmask_cache
ROOT=varvara/anza_lira_local_results/ct28_deepsets_fixedlps_seed_\${SEED}

python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_study.py \
  --stage all --structure-models --separate-endpoints --deepsets \
  --include-noaxial-control --frame-mode fixed_lps \
  --cache-dir "$CACHE" --root "$ROOT" \
  --seed "$SEED" --device "$DEVICE" \
  --epochs 8 --folds 5 --batch-size 2 \
  --relation-training augmented_15 \
  --pair-identity-source neural_pair_junction --patient-id 953
\`\`\`

For the five-volume matched primary comparison add
\`--deepsets\` to the same \`ccta_3d_val_an_compare.py\` command.
Use a unique \`--output-root\` and keep \`neural_pair_junction\`
identical for both new architectures. Repeat with three predeclared
seeds, and compare against the original nonlinear-off 4-channel
ANZA/CNN arm *without* changing the validation source or hard gate.

## 3. Critical additional correction: learned junction identity

In the original K10/15 decision path, the final PAIR candidate is chosen by
`geometry_plus_ct_score` but the final JUNCTION candidate by
`geometry_score`. The earlier neural runtime replaced `ct_score` for
the relation classifier and optionally PAIR identity, **but never JUNCTION
identity**. Thus even a perfect local ANZA JUNCTION ranking could not change
the chosen physical bifurcation. This is a direct architectural limitation.

The research branch now offers `--pair-identity-source neural_pair_junction`:
the learned JUNCTION probability is also placed in the experiment-only
`geometry_plus_ct_score` identity stream for JUNCTION. The original
`geometry_score` remains untouched and the 84-feature relation HGB still
receives the same geometrical evidence plus experimental CT evidence. The
historical `frozen_fusion` and `neural_pair` policies remain available.

The new policy affects physical candidate identity, 15-repeat consistency
and the logged selection utility. **This is not a guaranteed improvement**:
a high AUROC neural JUNCTION model can still make unsafe false connections.
Compare both ANN types under identical policies and the 0.85 relation gate.
The three-seed summary refuses mixing `neural_pair` and
`neural_pair_junction` reports.

## 3. Local prerequisites for Varvara

Switch to the experimental branch:

```bash
git fetch origin
git switch experiment/ccta3d-topology-anisotropy-2026-11-11
source .venv-ccta/bin/activate

python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_preflight.py \
  --require-validation --min-free-gb 10
```

The following must be accessible on that computer:

- Verified ImageCAS 801–1000 archive
  (`varvara/anza_lira_local_data/imagecas_801_1000/`);
- original 17 train and 5 development-val reference segmentations
  (`varvara/anza_lira_local_data/Varvara_CT28_junction_source_pack/segmentations/`);
- selected K10/15 relation model, SHA and frozen path v2;
- frozen `scene_plan_train_val.csv`, `scene_endpoints_train_val.csv`,
  `scene_candidates_train_val.csv` and CT alignment metadata;
- Python / Torch suitable for the actual accelerator (CUDA or MPS).

If the preflight fails, fix the input/provenance issue; do not bypass SHA
checks, fabricate an ImageCAS file, or open six held-out test patients.
**Never upload CCTA voxels, masks, personal data or checkpoints to public CI.**

### Train-only original branch-label audit

```bash
python scripts/research/ccta_graph_lira_safe_repair/audit_ct28_anatomical_segments.py \
  --out varvara/anza_lira_local_results/ct28_train_anatomical_segments.json
```

Review the fraction of unassigned background endpoints, number of distinct
segment labels, and whether true/false candidate assignments can be
distinguished. Equal segment label is not, by itself, proof of a valid link.
This command only opens the 17 permitted train scans.

## 4. First complete 3D CT28 training pass

Select `DEVICE=mps` on the original Mac workstation if MPS works;
`DEVICE=cuda` on a supported NVIDIA workstation. Keep identical device,
budget, split and model selection for all compared variants.

Example ONE seed. The cache may be reused between seeds only if its SHA,
tip encoding, physical frame, and target protocol match exactly. Use a
**new root for every seed**.

```bash
DEVICE=mps
SEED=20261102
CACHE=varvara/anza_lira_local_results/ct28_sep_fixedlps_trainmask_cache
ROOT=varvara/anza_lira_local_results/ct28_sep_fixedlps_seed_${SEED}

python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_study.py \
  --stage all --structure-models --separate-endpoints \
  --include-noaxial-control --frame-mode fixed_lps \
  --cache-dir "$CACHE" --root "$ROOT" \
  --seed "$SEED" --device "$DEVICE" \
  --epochs 8 --folds 5 --batch-size 2 \
  --relation-training augmented_15 --pair-identity-source neural_pair_junction \
  --patient-id 953
```

Repeat with `SEED=20261103` and `20261104` and unique roots. If a
stage fails, diagnose the actual exception, keep its failure artifacts, then
resume only the missing stage using `--stage patches|oof|relations|pilot|volume`
when the existing immutable receipts allow. Do not silently reuse old
two-channel/2-tip models or rename old checkpoint files.

**Before multi-seed training, verify one short smoke run on the local
accelerator**, especially `grid_sample`, `einsum`, the 27-hub readout
and the ANZA backward path. A previous CPU smoke on a public 3D CTA does
not prove MPS/CUDA compatibility or trained accuracy.

## 5. Five-patient frozen development-validation

First establish the original unchanged frozen validation manifest and K10
pilot masks, per the existing
`docs/varvara/CCTA_3D_LOCAL_IMAGECAS_RUN_2026-11-03.md`.
Use exactly the same mask, path and hard gates for CNN and ANZA.

Primary ANZA vs CNN, for each seed/root:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_val_an_compare.py \
  --run --structure-models --separate-endpoints \
  --freeze varvara/anza_lira_local_results/ccta_3d_validation/frozen_dev_val_20261103.json \
  --study-root "$ROOT" \
  --output-root "varvara/anza_lira_local_results/ct28_sep_val_${SEED}" \
  --device "$DEVICE" --relation-training augmented_15 \
  --pair-identity-source neural_pair_junction
```

Ablation (CNN-Structure against ANZA without forced axial preference);
requires a **different output root**:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_val_an_compare.py \
  --run --structure-models --separate-endpoints --noaxial-control \
  --freeze varvara/anza_lira_local_results/ccta_3d_validation/frozen_dev_val_20261103.json \
  --study-root "$ROOT" \
  --output-root "varvara/anza_lira_local_results/ct28_sep_noaxial_val_${SEED}" \
  --device "$DEVICE" --relation-training augmented_15 \
  --pair-identity-source neural_pair_junction
```

Finally, aggregate three immutable primary reports (not the ablation outputs):

```bash
python scripts/research/ccta_graph_lira_safe_repair/summarize_ct28_structure_seeds.py \
  --report 20261102:varvara/anza_lira_local_results/ct28_sep_val_20261102/matched_val_comparison.json \
  --report 20261103:varvara/anza_lira_local_results/ct28_sep_val_20261103/matched_val_comparison.json \
  --report 20261104:varvara/anza_lira_local_results/ct28_sep_val_20261104/matched_val_comparison.json \
  --out varvara/anza_lira_local_results/ct28_sep_three_seed_summary.json
```

The aggregator refuses missing patients, mixed model pairs, altered K10
frozen baselines and incompatible relations. It flags clDice wins that
also increase erroneous interbranch bridges instead of promoting
high recall with unacceptable damage.

**Primary new topology experiment** uses neural PAIR+JUNCTION identity because
otherwise the JUNCTION model cannot choose a different physical branch.
Also run `neural_pair` as a separately frozen historical identity-policy
control with a distinct output root if computational resources allow. Do not
mix the two policies in one multi-seed report.

## 6. Exact measurements required in Varvara's report

For every patient and seed, report:

1. **Input validity:** original CT affine and SHA, HU coverage, endpoint
   physical margins, fraction of partially outside CT patches, and missing
   individual tip channels.
2. **Local candidate quality:** PAIR/JUNCTION top-1, AUROC/AUPRC, true and
   false candidate ranks, calibration and negative-control score changes.
3. **Relation decision:** `NONE/PAIR/JUNCTION/BOTH`, exact/false/incomplete,
   score before `tau=0.85`, the consistency acceptance, and each refusal
   reason. Do not silently relax the safety gate.
4. **Full 3D mask:** actual true gaps repaired, accepted wrong interbranch
   connections, added voxels not supported by GT, Dice, clDice, inference
   seconds and GPU memory, on the same frozen broken input.
5. **Ablations:** old summed endpoints vs new separate tips; canonical vs
   fixed LPS if computationally feasible; CNN vs ANZA with the SAME dense
   supervision; ANZA forced-axial vs no-axial; HU-shuffled and heatmap-only
   controls on training/development data.
6. **Failure panels:** at least three visually verified 3D false connections
   or missed restorations, with original CT slices, candidate tip locations,
   predicted vessel probabilities, potential hub, path geometry and explicit
   rejection/acceptance reason.

Synthetic unit tests check whether the architecture can represent connectivity
and find close endpoints. They do NOT prove the HGB consistency gate works
on real CT or that restoring a one-voxel line meaningfully recovers vessel
lumen. The current mask pipeline often uses `repair_radius_mm=0`;
separate centerline quality from volumetric lumen reconstruction.


## Additional counterexample: matched same-tree PAIR rank, and negative ablations

The new `benchmark_same_tree_identity.py` freezes two candidate labels
per independently generated tree with **the same CT, dense vessel GT, and
endpoint separation**. The pair class encodes synthetic "continue through
trunk" vs "turn onto branch"; *both are within a single connected tree*.
This is a directional stress fixture, NOT universal medical branch truth.
The hard-negative report includes paired wins/losses, mean score margin,
AUROC/AUPRC, shuffled-HU control, and 0.85 accepted false branches.

On a preliminary tiny 16-tree training / 8-tree validation / 3-epoch /
one-seed CPU pilot, the **full CT BCE** runs achieved:
- CNN sep: 4/8 correct paired ranks, AUROC 0.484;
- ANZA sep: 7/8 correct paired ranks, AUROC 0.875;
- ANZA signed-neighbor-minus-center variant: 7/8, AUROC 0.875.

Without CT, the corresponding results were CNN 6/8, ANZA 3/8,
ANZA contrastive 3/8. The ANZA win therefore depended on 3D intensity
structure in this small pilot, but cannot be generalized from eight trees.
**No architecture accepted any candidate at raw tau=0.85.**

An additional equally budgeted within-tree ranking loss on that pilot:
- full CT: CNN 6/8, ANZA 7/8, contrast ANZA 7/8;
- endpoints only: CNN 6/8, ANZA 4/8, contrast ANZA 4/8.
Thus paired ranking improved some cases but did NOT prove ANZA-specific
improvement or useful high-confidence accepted repairs. The signed
transport ablation provided no demonstrated gain over original ANZA.

The ranking loss is now separately opt-in in patient-grouped CT28 OOF:
`--ranking-weight 0.5` (0 keeps historical objective). This samples
positive/negative candidates from the **same training scene** inside
the patient fold; no validation/test scene may enter. It retains the
original class BCE and train-only vessel losses. Run as an **additional
study in distinct output roots**, not as a replacement for the primary
unchanged `--ranking-weight 0` three-seed baseline. The study driver
refuses downstream use if persisted OOF ranking protocol differs.

**No claim of clinical gain** until 17 train / 5 dev ImageCAS full masks
are evaluated with the same true/false bridge safety criteria.

## 7. Test evidence and release criteria

- The preceding structural branch passed 72 regression tests and the
  public unannotated CTA-cardio numeric smoke before the new 4-channel
  representation was introduced.
- New tests cover separate-tip HU invariance, subvoxel endpoint moments,
  permutation symmetry, train-only dense labels, grouped OOF checkpoint
  loading and 3-seed summary safety.
- The latest synthetic stress workflows additionally require accurate
  26-connected GT labels, scalar one-feature controls, shuffled HU, equal
  HU histograms and 0.85-threshold accepted true/false decisions.
- **Latest code requires successful new GitHub Actions checks.** Do not claim
  it fully verified while runs are still queued or failing.

A valid promotion claim requires ANZA to beat matched CNN in the selected
patient-level structural metric **without increasing unsafe false bridges**;
it must not rely on one best seed, easier synthetic data, or cherry-picked
thresholds. Five development patients are descriptive, not independent
clinical confirmation. Only open the six frozen held-out patients after
the final architecture and operating threshold have been fixed.

**External blocker:** original licensed ImageCAS CT28 volumes and local
trained checkpoint artifacts are not mounted in this execution environment.
Varvara must run the actual 17-train/5-dev experiments on the authorized
machine. Do not infer clinical results from public CTA numeric smoke or
synthetic AUROC.

## Completed available-data training (2026-11-12; SYNTHETIC ONLY)

The canonical train run is [GitHub Actions #43118173646](https://github.com/fims9000/anza_lira/actions/runs/43118173646), built from SHA `b5d6208caa85130086d954da77072933942b0835`.
All **three** experiment jobs completed successfully, with 21 independently reload-verified model state dicts; SHA-256 hashes and predictions are archived as workflow artifacts.

Data: synthetic paired same-26-connected-tree 24³ Y-vessels, identical CT and dense mask within true/false candidate pairs, equal endpoint distance. 64 train trees, 24 validation trees, *48 independent synthetic test trees*, 12 epochs; full image runs use 3 seeds (17,41,73); endpoint-only negative control uses seed17. Models have **6 internal channels**, not the actual CT28/clinical configuration. The synthetic ground-truth convention selects continuation through the trunk over turning onto the branch, which is NOT a universal clinical definition.

| Test metric (mean 3 seeds; 48 trees) | CNN | ANZA | ANZA signed contrast |
| --- | ---: | ---: | ---: |
| Full synthetic HU, ordinary BCE AUROC | **0.787** | 0.709 | 0.750 |
| Full HU, BCE correct same-tree PAIR | **43.3/48** | 36.7/48 | 38.0/48 |
| Full HU, BCE + paired ranking AUROC | **0.847** | 0.756 | 0.805 |
| Full HU, BCE + paired ranking correct PAIR | **46.0/48** | 41.3/48 | 41.3/48 |
| Raw accepted at frozen 0.85 threshold | **0** | **0** | **0** |

Paired tree bootstrap vs CNN, ordinary BCE: ANZA correct-rate difference -0.139, 95% CI [-0.208,-0.069]; signed-contrast -0.111, CI [-0.191,-0.035]. With train-only paired ranking: ANZA -0.097, CI [-0.160,-0.042]; signed contrast -0.097, CI [-0.153,-0.049]. The independent endpoint-only control gave almost random global AUROC (~0.51), confirming that the full-image gains on this fixture are not reproduced without HU.

**Conclusions:** the larger, three-seed synthetic test **falsifies any claim that ANZA currently outperforms parameter-matched CNN** on this specific same-tree branch-identity task. Ranking improves performance but benefits CNN too. Signed transport did not reverse the ordering. No model accepted a correct or false connection at 0.85, so *no repaired-vessel benefit has been demonstrated*. Frozen CT28 inference, patient split, relation HGB and thresholds were not altered by this training. These synthetic six-channel weights must not be substituted for clinically trained CT28 weights, and neither real ImageCAS validation nor held-out clinical test was accessed.

