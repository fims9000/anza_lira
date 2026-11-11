# CT28 ANZA structure-aware experiment, 2026-11-11

**Research status: experimental. ANZA superiority on ImageCAS/CT28 is NOT
established. Do not replace the frozen K10/15 or touch six held-out test subjects.**

## Motivation and exact change

The original \`Candidate3dEncoder\` uses one binary relation BCE per
PAIR/JUNCTION candidate. Its pooled head has no explicit dense topology target.
The original AZConv3d metric
\`G=exp(2h) uu^T + exp(-h)(I-uu^T)\` has unrestricted signed \`h\`:
\`h>0\` penalizes axial transport MORE than transverse transport.

The new opt-in variants \`anza3d_structure\` and \`conv3d_structure\` receive
the SAME two input channels (real normalized CCTA HU and the existing
endpoint Gaussian heatmap) and the SAME extra train-only 3D vessel masks.

- ANZA-Structure constrains \`h<0\`, making transport cheaper along the
  learned direction than transverse to it. The original ANZA remains unchanged.
- BOTH structure variants produce a train-supervised dense vessel probability.
  Auxiliary loss: 0.35 * (balanced BCE + Dice + differentiable clDice).
- BOTH produce an explicit differentiable path support score. PAIR evaluates
  a straight and four modestly curved candidate paths; JUNCTION requires
  three arms to reach an approximate common hub. The link BCE receives
  0.15 weight. The classification logit contains a positive-weight continuity
  score. This still does NOT guarantee branch identity or correct topology.
- The CNN-Structure variant has exactly the same dense head, route readout,
  label access and optimizer budget. The fair comparison is
  \`anza3d_structure\` vs \`conv3d_structure\`, not against an unaugmented CNN.
- Dense labels are taken ONLY from 17 CT28 train subjects. Mask affines/shapes
  must match the verified original CT grid and reference SHA is recorded.
  The five development-validation and six held-out test labels NEVER enter
  training; test cohort is not accessed.

## Verified nonclinical evidence

- Structural physics, broken-path/three-arm behavior, gradient, sparse clDice
  and patient-OOF protection numerical tests are in
  \`tests/test_ccta3d_structure_bridge.py\` and
  \`tests/test_ccta3d_structure_oof.py\`.
- The public 3D Slicer CTA-cardio image is used strictly for real-HU/coordinate/
  numeric forward/backward smoke: \`smoke_ccta3d_public_realct.py\`.
  It contains no gold coronary candidate identities. Its randomly initialized
  predictions are NOT accuracy evidence.
- Synthetic branch-preserving/broken tube results are in
  \`benchmark_structural_tubes.py\`; this is a narrow, deliberately generated
  development diagnostic. Results are seed-sensitive. At 8 epochs PAIR
  \`conv3d_structure\` reached AUROC 1.000/0.997, and
  \`anza3d_structure\` 1.000/0.979 on two seeds. Both topology variants
  remain weak on synthetic JUNCTION. This does NOT justify promoting ANZA.
- A separate oracle run feeds perfect vessel target masks into the structural
  readout to determine whether a geometry/readout limitation exists
  independently of ANN optimization.
- On real frozen validation geometry, the peak-recovery check measures whether
  the heatmap reliably represents the intended PAIR/JUNCTION endpoints.
  No CT or held-out data is opened in this geometry audit.

## Reproducible CT28 experiment on original local ImageCAS data

The raw ImageCAS volumes and original selected frozen model weights are not
in GitHub. Run on the authorized local research workstation only, not Actions.
Do not commit CT, raw masks, patches or trained patient models publicly.

    python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_preflight.py \
      --require-validation --min-free-gb 10

For each seed 20261102, 20261103, 20261104, create a NEW experiment root.
The patch cache may be shared across seeds ONLY when the exact image, frame,
dense supervision and SHA provenance agree. Example:

    FRAME=fixed_lps
    SEED=20261102
    CACHE=varvara/anza_lira_local_results/ct28_structure_fixedlps_trainmask_cache
    ROOT=varvara/anza_lira_local_results/ct28_structure_fixedlps_seed20261102

    python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_study.py \
      --stage all \
      --structure-models --frame-mode "$FRAME" \
      --cache-dir "$CACHE" --root "$ROOT" \
      --seed "$SEED" --device cuda --epochs 8 --folds 5 \
      --batch-size 2 --relation-training augmented_15 \
      --pair-identity-source neural_pair --patient-id 953

For an MPS workstation, replace \`--device cuda\` with \`--device mps\`.
The exact device, epochs, train/val split, focal selection and thresholds
must match between ANZA-Structure and CNN-Structure.

Full five-patient development-validation comparison (using the same frozen
development freeze and NEW output root):

    python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_val_an_compare.py \
      --run --structure-models \
      --freeze varvara/anza_lira_local_results/ccta_3d_validation/frozen_dev_val_20261103.json \
      --study-root "$ROOT" \
      --output-root varvara/anza_lira_local_results/ct28_structure_fixedlps_seed20261102_val \
      --device cuda --relation-training augmented_15 \
      --pair-identity-source neural_pair

Do not silently switch to \`frozen_fusion\` for primary evaluation: its PAIR
identity comes from old geometry+CT and does not demonstrate the new encoder.

**Acceptance, NOT predetermined:** before opening held-out, freeze a topology
metric: exact repairs/false accepted component bridges, junction recovery, gap
recall, clDice, Dice, abstention and runtime at identical operating thresholds.
A 5-patient descriptive validation is not independent clinical evidence.
If ANZA fails to outperform matched CNN, document the failure and retain CNN.
