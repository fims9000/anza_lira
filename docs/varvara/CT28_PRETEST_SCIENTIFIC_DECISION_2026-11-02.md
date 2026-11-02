# CT28 Graph-LIRA: pre-test scientific decision record (2026-11-02)

**Scope:** read-only validation audit plus a fail-closed safety fix on a separate branch. The selected K10/15 weights, its `tau=0.85`, `consistency=0.60`, local scores, and its saved predictions are unchanged. No held-out Graph-LIRA test is authorized or executed here.

## Result we can report now

Selected K10/15, five validation subjects, 368 controlled scenes, of which 206 require repair:

| Metric | Geometry-only | CT-conditioned |
| --- | ---: | ---: |
| Structurally exact before consistency gate | 157/206 | 155/206 |
| False structural repairs before gate | 20/368 | 11/368 |
| Exact accepted after gate | 16 | 52 |
| False accepted after gate | 0 | 0 |
| Coverage, all scenes | 16/368 (4.35%) | 52/368 (14.13%) |
| Correct repair fraction, repair-needed scenes | 16/206 (7.77%) | 52/206 (25.24%) |

CT-conditioned accepted exact per patient (geometry-only in parentheses): 957 11 (3), 961 15 (5), 965 7 (4), 966 13 (2), 974 6 (2). Subgroup acceptance: PAIR 48/134, JUNCTION 2/36, BOTH 2/36.

**Interpretation:** CT-conditioned Graph-LIRA has better selected **coverage** on this development protocol, not better pre-gate structural exact count. The zero observed false repairs on five repeatedly used validation patients cannot establish a safe false-repair rate in a new population.

## Audited hazards

1. **Scene construction shortcut.** A count-only lookup trained on 1,036 train scenes predicts the relation class on 270/367 count-covered validation scenes (73.57%, 367/368 covered). Direct count removal in the relation head was already tested elsewhere; the risk persists through correlated score-landscape features. All controlled positives have fixed endpoint positions by family: PAIR E00+E01, JUNCTION E00+E01+E02, and BOTH PAIR E00+E01 / JUNCTION E02+E03+E04. The code ranks candidate IDs by geometry/CT scores rather than using IDs as model features; deterministic true positions are a **benchmark weakness**, not proven model exploitation.
2. **Fixed candidate pool under perturbations.** Existing perturbation evaluation rescores original candidates rather than regenerating candidates. Its consistency means conditional decision stability, not end-to-end detector stability.
3. **Potential partial BOTH acceptance.** A relation head can propose BOTH while no endpoint-disjoint PAIR/JUNCTION pair can be selected. The previous gate used only `selected_relation != NONE` and consistency; it did not require compatibility or selected type to equal proposed type. This branch adds a fail-closed condition. Both saved selected models have zero proposed-vs-selected mismatch over all 368 scenes each, so current 16/52 acceptance counts are unchanged by the fix.
4. **Insufficient evidence of full-mask improvement.** Path v2 builds routes for all 52 accepted validation scenes, but does not yet rasterize a corrected full 3D vessel segmentation or report paired Dice/clDice/connected-component outcomes.
5. **Historical output confusion.** Root-level `graph_lira_per_patient_val.csv` and `graph_lira_patient_bootstrap_val.csv` belong to an earlier model; reporting current K10/15 requires recomputation from `selected_validation/graph_lira_predictions_val.csv`.

## Fastest credible route to an independent result

### Gate A — freeze the controlled evaluation

- Complete this PR's fail-closed safety change and focused regression tests.
- Add an **ID-renaming metamorphic test** (relabel endpoint IDs and the corresponding candidate relations without changing geometry/CT) and confirm structural decisions match under inverse mapping. This should be a test of full inference, not an assertion about the scene plan.
- Run one **CT-information sham**: same 84-dimensional head/augmentation/OOF, but train-patient-wise shuffled CT score stream, so extra capacity does not masquerade as CT evidence. Maintain patient isolation. Include a sham with consistent train/validation transformation; do not independently shuffle validation labels or leak the target.
- Declare the residual controlled-scene shortcut and the frozen-pool limitation if robustness redesign is deferred.
- Freeze the exact annotation version, input SHA256, code commit, noise protocol, models, chosen path v2, metric denominator, failure handling, and stopping conditions. **Do not update validation selection after inspecting held-out outcomes.**

### Gate B — one-shot held-out test, controlled scope only

Held-out IDs: 954, 958, 972, 973, 980, 984. Apply the frozen generator, local models, relation head, identity rule, compatibility, gates, and path rule once. Save per-subject scene IDs, classification and structural outcomes, paths, path failures, time, and grouped metrics. Use patient-level intervals and make clear only six independent patients are represented. Since the earlier PAIR-only research already reported held-out outcomes on these CT subjects, treat this as *new pipeline held-out*, not a fully untouched external cohort across the program.

### Gate C — publishable full-volume experiment

Create naturally imperfect segmentation masks using a documented segmentation model on **patient-disjoint** CT studies. From predicted connected components derive endpoints and candidates. Compare no repair, geometry bridge, non-selective image-guided route, geometry selective Graph-LIRA, and CT-conditioned selective Graph-LIRA on identical predictions. For every accepted relation, rasterize a conservative 3D lumen corridor from the chosen CT route and record added voxel volume, new false bridges, and component connections. Evaluate Dice, clDice, bifurcation connectedness, branch-level false bridging, abstention, and wall-clock time per patient. Include separately controlled synthetic gaps and natural segmentation errors.

**Only after Gate C** is the study ready to claim *improved full 3D coronary segmentation topology*. ANZA-versus-compact-CNN encoder ablation can follow on the same frozen downstream task, rather than delaying the structural baseline again.

## Claim boundaries

Acceptable today: *On five development subjects with controlled CCTA-derived gap scenes, CT-conditioned evidence increased accepted exact repairs from 16 to 52 at zero observed false accepted repairs, with a large residual abstention rate in JUNCTION/BOTH.*

Not justified today: *Clinical safety, superior real 3D segmentation, natural-gap generalization, or validated benefit of the ANZA 3D encoder.*

## Reproduction commands

```bash
python -m pytest -q tests/test_ct28_safety_gate.py tests/test_ct28_scene_shortcuts.py
python scripts/research/ccta_graph_lira_safe_repair/audit_ct28_scene_shortcuts.py
```

The shortcut diagnostic is read-only and requires only Python's standard library and the committed `scene_plan_train_val.csv`. The safety tests require pytest; the saved-output test needs only the committed selected validation CSV. Running end-to-end Graph-LIRA requires the ignored local medical CT archive and model pack, which are not part of this review.
