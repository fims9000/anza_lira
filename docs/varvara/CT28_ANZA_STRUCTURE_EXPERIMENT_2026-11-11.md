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
  --relation-training augmented_15 --pair-identity-source neural_pair \
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
  --pair-identity-source neural_pair
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
  --pair-identity-source neural_pair
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
