# S71 frozen retrospective protocol — correct historical S65–S67 synthetic labels

2026-10-26. Frozen BEFORE execution of the S71 historical replays. Research branch: `research/bubble-chamber-graph-lira`. No native H50Z physical track ground truth is available.

## Scientific question

S69 established a serious orientation-dependent *evaluation* bug in S64; S70 replays the S64 original predictions and fixes their synthetic correctness counts. Do the published S65, S66, and S67 negative/abstention findings survive when the correct S69 evaluator scores the exact same image-to-graph predictions? This is a **retrospective erratum**, not fresh held-out evidence, and must not be used for tuning a new method.

## Locked inputs and partitions

- Exact source files from the repository snapshot of 26 October 2026, prior to S71, and exact S65–S67 historical seeds and source-only geometry fit, frozen in the existing source. S65 CAL seeds 20266501/20266511 (32 scenes/photo), S65 TEST 20266503/04/13/14 (24 per seed). S66 CAL 20266601/20266611 (32/photo), S66 TEST 20266603/04/13/14 (24 per seed). S67 CAL 20266701/20266711 (16/photo), S67 TEST 20266703/04/13/14 (24 per seed).
- Source film-070 visualMetro 145 audited complete frames, fixed whole-frame train/CAL/TEST 79/30/36. The same two H50Z photographs (39 private crops) are photo backgrounds only and belong to ONE physical acquisition run. Source film-070 geometry is NOT H50Z native ground truth. All recorded correct/incorrect links concern artificially injected tracks.
- The original S56 geometry scorer, 5%-FPR source CAL threshold and solver, S39 detector, S65 pixel-only endpoint filter, S66 pixel-only extension, S67 crossing stitch, original protocol/cutoff and seeds are **frozen**. No parameter search or changes to inference allowed. Only `audit_synthetic_scene` is shadowed by an evaluation-only dual audit, and its historic return is fed back to the old benchmark unchanged.
- Dual evaluator: original `bubble_chamber.s64_pixel_to_path.audit_synthetic_scene` and orientation-invariant evaluator `bubble_chamber.s69_orientation_invariant_audit.audit_orientation_invariant_synthetic`. On the same finished inference, require *identical* chosen edge count, candidate count, node identity association, unmatched count, original scoring terms, global solver failures, synthetic decoy association. Only correctness, wrong-endpoint count, noncorrect count, any-error-scene, full synthetic paths may change; original positive cannot be invalidated by correction.

## Historical verification and corrected CAL selections

- Re-run CAL and TEST for all three historical studies. Reproduce old published CAL and TEST metrics exactly, including per-seed totals and geometry-only cutoff; any mismatch is a hard failure. Store old reports untouched and add new S71 report + public-safe aggregate JSON only.
- After CAL is re-evaluated, compute a **counterfactual corrected-CAL decision** using each study's ORIGINAL frozen selection rule; do not re-fit the model or inspect TEST before this choice. Report both historical CAL choice and corrected-CAL choice. Any difference is retrospective reclassification and NOT an independent confirmation of superiority.
- For S65, original TEST evaluates the historical CAL-selected filter; if corrected CAL changes the chosen filter, a separate retrospectively evaluated counterfactual branch may be reported as exploratory (never merge its totals with the historical TEST). S66 and S67 evaluate all fixed arms on historical TEST, making corrected-CAL counterfactual comparisons available from the same predictions.
- Produce paired within-synthetic-scene 95% bootstrap difference intervals with fixed `np.random.default_rng` seeds, 1200 replicates, for corrected vs original and each algorithm vs S39. Do not treat crops or photographed tracks as independent camera runs. Enumerate per-source directions and joint totals for each method, as well as number of wrong-endport corrections and completed synthetic paths.

## Privacy and scientific limits

No native H50Z GT; no physical automatic join permission. Never publish BMP, PNG, 7z, native measured coordinates, per-scene source identities, predicted polylines, extracted real-photo pixels or training weights. Public Git: source-only Python, protocol, aggregate JSON, SHA-256 source locks and experiment IDs; raw photos and per-scene injected GT remain private locally. Freeze only algorithm/code hashes (never reveal private file hashes that enable file identity linking).

## Prespecified decision interpretation

Previous conclusions may change solely because the evaluator was wrong. Do not present corrected retrospective counts as new model training or external validation. Do not promote a method unless independently frozen CAL policy and later new TEST with a different physical acquisition run support the benefit; at S71 level, default remains original S39. The next work (S72) should use corrected evaluation from the beginning, with brand-new synthetic seeds and, when available, independent real-image runs.
