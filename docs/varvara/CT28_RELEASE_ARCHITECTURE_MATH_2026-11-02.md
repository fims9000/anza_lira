# CT28 Graph-LIRA: release architecture, mathematics, and next decisive experiment

Status: **release-candidate engineering implementation / development-only scientific evidence** (2026-11-02). This document does not claim that held-out testing, new neural training, or full-mask CCTA evaluation has been performed.

## 1. What is actually frozen

Source branch: `research/coronary-connectivity-repair`, K10/15 from the commit `55e52efa5acf273e491e153ecb70a75ce6fdaff0`.
- Patients: train 17, validation 5, test 6. The held-out Graph-LIRA test is closed.
- Local HGB: PAIR/JUNCTION geometry trained with 5 position-only copies; CT on clean; PAIR identity geometry+CT HGB with 10 joint copies; JUNCTION identity geometry HGB.
- 84-input CT-conditioned relation HGB, trained with 15 joint perturbation copies and patient-level OOF local scores.
- Candidate identity, compatibility, `tau=0.85`, 15 evaluation perturbations and exact-set consistency `>=0.60` stay fixed.
- Path v2: maximize minimum CT support, then allow edges >=95% of optimum and choose shorter routes, with endpoint clearance.

This PR changes neither weights nor local/structural predictions. The safety gate now verifies that the proposed relation type is **exactly** the selected type and that the selected candidates satisfy its compatibility constraints.

## 2. Formal scope: relation versus trajectory versus mask

Let `X` be CCTA; `M` a *predicted, broken* vessel mask; `E(M)` its observed endpoint set; and `C_P(E)`, `C_J(E)` the PAIR and degree-3 JUNCTION candidate sets. This is fundamentally a **selective structural decision**, not ordinary pair classification.

A structural repair `r` is a set of PAIR and/or JUNCTION candidates. Define the feasible set

\[
\mathcal F(E)=\{\varnothing\}\cup C_P(E)\cup C_J(E)\cup
\{(p,j):p\in C_P(E),j\in C_J(E),\;\mathrm{ends}(p)\cap\mathrm{ends}(j)=\varnothing\}.
\]

The selected repair type `t(r)` lies in NONE, PAIR, JUNCTION, BOTH. The relation head computes `p_theta(t|z)`; candidate scores `s_phi(c|X,E)` are **ranking utilities**, not necessarily calibrated probabilities.

The existing K10/15 uses a relation-class maximum confidence gate followed by identity selection and exact-set agreement under perturbation. It can abstain even when its clean relation argmax is correct. The frozen acceptance decision is

\[
A(X,E)=\mathbf 1[t(r)\ne\mathrm{NONE}]
\mathbf 1[t(r)=\widehat t]
\mathbf 1[r\in\mathcal F(E)]
\mathbf 1[\mathrm{consistency}(r)\ge0.60]
\]

where `\hat t` already passed `p_theta(\hat t|z)\ge0.85`. An unrepresentable BOTH must abstain; do **not** silently downgrade it to PAIR or JUNCTION.

### Path stage, unchanged

For CT-supported edge weights `a_e`, the v1 bottleneck of a path `\gamma` is

\[
b^*=\max_\gamma\min_{e\in\gamma}a_e.
\]

Train-selected v2 searches among edges with `a_e\ge0.95b^*` and minimizes total physical path length, subject to endpoint/JUNCTION constraints. This creates a voxel-center route but does **not** itself change any 3D segmentation mask.

### New conservative mask stage (experimental implementation)

Let `A` be the image affine (voxel index to RAS mm). For a route set `\Gamma` and train-fixed physical radius `r_{\rm mm}`, the proposed added voxels are

\[
\Delta M(\Gamma)=
\left(\bigcup_{u\in\Gamma}
\{v:\|A_{3\times3}(v-u)\|_2\le r_{\rm mm}\}\right)\setminus M .
\]

The repair is an **atomic transaction**: reject all additions, with original `M` preserved, if
1. The path is not 26-connected, leaves the volume, or lacks original-mask endpoint anchors.
2. The selected endpoints do not belong to distinct original connected components.
3. The new mask joins any originally unselected component, or TWO BOTH repairs touch.
4. The added-voxel budget is exceeded, or CT support is absent where required.
5. The final number of components differs from `n_before - sum_relations(k_rel-1)`, where `k` is 2 for PAIR and 3 for JUNCTION.

This component constraint is **not a complete anatomical guarantee**: a route can be wrong while still satisfying it. A centerline/clinical audit remains necessary.

Implementation: `ct28_mask_reconstruction.py`; standalone local CLI: `ct28_mask_reconstruction_cli.py`. Its unit tests use synthetic volume data only. A real experimental mask must be derived from a **predicted segmentation**, not the ground-truth ImageCAS-X mask.

## 3. One mathematically focused experiment for the remaining JUNCTION/BOTH bottleneck

Current facts: HGB K10/15 gets 155/206 exact pre-gate and 52/0 accepted after gate. JUNCTION and BOTH contribute only 2/36 and 2/36 accepted. Existing diagnostics attribute losses to unstable relation confidence, relation type and candidate identity. A prior MLP CE+JS experiment got 54/0 on the *same development cohort*, but does not justify replacing K10/15.

**Pre-register ONE exploratory experiment, without changing the frozen K10/15:** replace separate hard type/identity regularity with a *structured perturbation-consistency objective*. The differentiable feasible-set distribution and JS loss are now implemented as a **train-only research primitive** in `ct28_structured_objective.py`, with autograd and finite-difference gradient tests. **No model has been trained or validated with this objective yet.**

For each feasible type `t`, let `\mathcal F_t(X)` be feasible candidate sets with that type; for `r\in \mathcal F_t` define

\[
q_\phi(r\mid t,X)=
\frac{\exp[U_\phi(r,X)/T]}
{\sum_{r'\in\mathcal F_t(X)}\exp[U_\phi(r',X)/T]},
\quad U_\phi(r,X)=\sum_{c\in r}s_\phi(c,X).
\]

Set `q_\phi(\varnothing|\mathrm{NONE},X)=1`. This normalizes **within type**, so types with more candidates do not automatically acquire larger total prior probability. Candidate utilities require separate calibration/learning; uncalibrated HGB probabilities should not be asserted to define reliable likelihoods.

A joint structural distribution on feasible repairs can then be written as

\[
Q_{\theta,\phi}(r\mid X)=p_\theta(t(r)\mid X)\,q_\phi(r\mid t(r),X).
\]

For a clean training scene `X` and a controlled perturbation `T_kX` (with explicit candidate identity correspondence), use

\[
\mathcal L =
\mathcal L_{\rm CE}(t^*,p_\theta)
+\alpha[-\log q_\phi(r^*\mid t^*,X)]
+\lambda K^{-1}\sum_{k=1}^{K}
  \mathrm{JS}[Q(X)\,\|\,Q(T_kX)].
\]

For NONE only the type loss is required; identity loss is omitted. If candidate pools are regenerated, missing candidates must be treated as **candidate recall failures** rather than silently assuming fixed identity correspondence. No clean unseen reference scores may be used at inference.

**Ablation contract:** compare frozen HGB K10/15 against one pre-registered structured model with identical train/validation split, patient-level OOF for train-local evidence, same candidate generator, same perturbations and `tau=0.85`/consistency `0.60`. Fix `T,\alpha,\lambda` by *train-only patient-grouped cross-validation* before touching validation. Report all types, per-patient changes, false accepted repairs, and rejections. One exploratory pass is not evidence of generalization, and it must not re-open the K10/15 selection process.

**Alternative control:** same-capacity CT head with patient-safe, label-independent CT-sham features. This checks whether additional feature capacity, rather than image information, explains the gain. Existing count-only accuracy (~73.6% on covered validation) remains a major shortcut warning.

## 4. Statistical evidence, not headline optimization

Selected K10/15 on five *reused development patients*:
- Geometry-only: 157/206 exact, 20/368 false before gate; 16/0 exact/false accepted after gate.
- CT-conditioned: 155/206 exact, 11/368 false before gate; 52/0 exact/false accepted after gate.
- Validation accepted exact by patient: `957:11`, `961:15`, `965:7`, `966:13`, `974:6`.
- PAIR accepted 48/134, JUNCTION 2/36, BOTH 2/36.

Primary clinical-facing question: at a stated false-bridge risk, how often is the **correct anatomical connection** restored in new, independent subjects? Never infer a population guarantee from 0/52 false accepted here.

Use `ct28_selected_statistics.py` to regenerate validation tables, per-patient counts and **exploratory patient-cluster paired bootstrap** directly from `selected_validation/graph_lira_predictions_val.csv`. Do not report the historical root-level `graph_lira_patient_bootstrap_val.csv` as K10/15.

End-to-end mask metrics after future eligible inference should be computed with `ct28_mask_metrics.py`, strictly post hoc:

\[
\mathrm{Dice}=\frac{2|M\cap G|}{|M|+|G|},\quad
T_{\mathrm{prec}}=\frac{|\mathrm{skel}(M)\cap G|}{|\mathrm{skel}(M)|},
\quad T_{\mathrm{sens}}=\frac{|\mathrm{skel}(G)\cap M|}{|\mathrm{skel}(G)|},
\]
\[
\mathrm{clDice}=\frac{2T_{\mathrm{prec}}T_{\mathrm{sens}}}
{T_{\mathrm{prec}}+T_{\mathrm{sens}}}.
\]

Report before/after paired Dice and clDice by patient, reference-supported vs unsupported added voxels, branch-specific false connections, number of rejected proposed bridges, absolute added volume (mm3), and runtime. Both predicted and reference masks must be registered in one voxel grid. Full-mask Dice/clDice cannot be computed from the structural scene CSV alone.

Reference for clDice: Shit et al., CVPR 2021, https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html .

## 5. Release boundaries, test firewall and next actions for Varvara

1. Review and merge the **engineering** safety/test PR into her active `research/coronary-connectivity-repair` only after tests pass. It is not a new validated model.
2. Apply the mask stage to **one train-only broken segmentation** first. Investigate failure reasons (unanchored endpoints / third-component contacts) before deciding any train-only radii or voxel budgets. Run synthetic topology tests and record the resulting mask, component labels, added voxels and post-hoc Dice/clDice.
3. Apply the *frozen* mask protocol on the existing **five validation patients** without another parameter sweep. Include validation failure cases and explicit no-repair baselines.
4. Define the evaluation scope in advance. Only then consider the **one-shot held-out Graph-LIRA test** on 954, 958, 972, 973, 980, 984. Earlier PAIR-only work used these patients; do not advertise them as entirely unseen across the entire research program.
5. Next, expand to patient-disjoint real segmentation failures on the ImageCAS-X cohort (800 scans are described publicly, but dataset and annotation version must be verified locally). Compare *no repair*, geometry shortcut, geometry selective, CT selective, and an image-guided unconditional path on identical predicted masks.
6. **Only after the architecture is stable:** compare radial handcrafted CT, compact 3D CNN, and ANZA-style anisotropic encoder with matched splits, complexity budget and downstream head. The earlier seismic ANZA B3-vs-B2 experiment did not establish incremental value; it cannot be assumed to improve 3D CCTA.

Important: held-out test subjects remain inaccessible in this PR. No `torch` training, data fetch, CT processing or new model selection is performed.

## 6. Commands on the clone (development mode)

```bash
python -m pytest -q tests/test_ct28_safety_gate.py tests/test_ct28_scene_shortcuts.py tests/test_ct28_candidate_permutation.py tests/test_ct28_mask_reconstruction.py tests/test_ct28_mask_reconstruction_cli.py tests/test_ct28_mask_metrics.py tests/test_ct28_mask_metrics_cli.py tests/test_ct28_selected_statistics.py tests/test_ct28_structured_objective.py
python scripts/research/ccta_graph_lira_safe_repair/audit_ct28_scene_shortcuts.py
python scripts/research/ccta_graph_lira_safe_repair/ct28_selected_statistics.py --out-md /tmp/ct28_selected_results.md --out-json /tmp/ct28_selected_results.json
```

To exercise the experimental mask stage, use a train/validation-only `paths_val.jsonl` (not available in Git) plus a separately generated **broken predicted mask** aligned to the CT affine:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ct28_mask_reconstruction_cli.py \
  --routes /path/to/paths_val.jsonl --scene-id '<selected-scene-id>' \
  --broken-mask /path/to/broken_prediction.npy --affine /path/to/voxel_to_ras.npy \
  --radius-mm '<train-chosen-radius>' --max-new-voxels '<train-chosen-budget>' \
  --output-mask /path/to/repaired.npy --output-report /path/to/reconstruction.json
```

Neither arbitrary clinical thresholds nor an unseen dataset outcome are supplied by this document. The trained models and medical volumes reside in the ignored local Varvara folder and must not be committed.

For a separately produced repaired mask (evaluation only, never a solver input):

```bash
python scripts/research/ccta_graph_lira_safe_repair/ct28_mask_metrics_cli.py \
  --patient 957 --before /path/to/broken_prediction.npy \
  --after /path/to/repaired.npy --reference /path/to/reference.npy \
  --affine /path/to/voxel_to_ras.npy --output /path/to/posthoc_metrics.json
```

The clinical-dataset objective remains open; only the mathematical objective implementation and synthetic mask-interface tests are verified here.
