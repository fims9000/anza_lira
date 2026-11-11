# CT28 3D: fixed-LPS patch-frame experiment

Status 2026-11-11: development-only experiment. This is NOT evidence that ANZA
outperforms CNN or improves real coronary segmentations. Frozen K10/15, thresholds,
paths and held-out test subjects must remain unchanged.

## Hypothesis

Historical "canonical" patches sort candidate endpoints by LPS coordinates and
construct an oriented frame using the first two endpoints. Small perturbations
can reorder the endpoints or switch the reference axis and strongly rotate the
whole CT/heatmap patch. Both ANZA and CNN may be affected.

The opt-in "fixed_lps" frame uses the centroid but a constant LPS basis. This
eliminates this particular orientation discontinuity, but sacrifices rotation
normalization and can lose endpoint FOV coverage. The two modes require
independent model training, not checkpoint reuse.

## Exact CT-free audit

From repo root:

    python scripts/research/ccta_graph_lira_safe_repair/audit_ccta3d_frame_stability.py \
      --split val --seed 20260920 --repetitions 15 \
      --output evidence/ccta3d_fixed_frame/val_audit.json

The audit reads committed scene/endpoint/candidate CSVs and uses the actual
CT28 endpoint perturbation generator. It records per-candidate frame rotation
and physical endpoint coverage for 24^3 patches, but reads NO medical CT.
It does not measure network accuracy.

## Train-only / development-validation work with local ImageCAS CT

Run the read-only preflight on the workstation with the verified ImageCAS CT
archive, reference masks, K10/15 pack and existing frozen path v2:

    python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_preflight.py \
      --require-validation --min-free-gb 10

Run BOTH modes "canonical" and "fixed_lps" and seeds 20261102, 20261103,
20261104. Never reuse the previous training checkpoint across frame modes.
Use separate patch-cache and experiment-root directories for each frame mode
and seed. Example for ONE run:

    FRAME=fixed_lps
    SEED=20261102
    CACHE=varvara/anza_lira_local_results/ccta3d_frame_fixed_lps_patches
    ROOT=varvara/anza_lira_local_results/ccta3d_frame_fixed_lps_seed_20261102

    python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_study.py \
      --stage all --patient-id 953 --frame-mode "$FRAME" \
      --cache-dir "$CACHE" --root "$ROOT" \
      --seed "$SEED" --device cuda --epochs 8 --folds 5 --batch-size 2 \
      --relation-training augmented_15 --pair-identity-source neural_pair

For a Mac MPS workstation substitute "--device mps". Use the same device,
training budget, samples, model selection and evaluation for both variants.
Both PAIR and JUNCTION encoders and their patient-OOF relation HGB must be
retrained. The six held-out test patients remain inaccessible.

For development full-mask evaluation, apply the unchanged validation freeze
to both models with "ccta_3d_val_an_compare.py --run
--relation-training augmented_15 --pair-identity-source neural_pair"
and supply the matching "--study-root", a NEW "--output-root" and the same
"--freeze" manifest. The control "frozen_fusion" is separate; it does not
measure neural PAIR selection.

## Main outcomes

- PAIR and JUNCTION top-1, AUROC, AUPRC on the same candidates, per patient
  and seed.
- Exact, incorrect, incomplete and abstained scene decisions; relation
  confusion, tau and consistency rejection reasons.
- Actual full-mask recovered gaps, false inter-branch bridges, added voxels,
  Dice, clDice and wall-clock time.
- Physical endpoint coverage and CT-volume support for both patch modes.
- Counterfactual controls: HU-only, heatmap-only, HU shuffled between
  candidates within train patients, ANZA anisotropy-off.
- Patient-level contrasts and seed variation. These five development patients
  were previously used for selection; there is no independent significance
  claim from them.

Do NOT lower tau based on the five development subjects or inspect held-out CT.
ANZA should be promoted only after matched CNN and K10/15 comparisons show a
real improvement in an explicitly chosen structural objective WITHOUT
unacceptable false component bridges. Otherwise retain the stronger baseline.
