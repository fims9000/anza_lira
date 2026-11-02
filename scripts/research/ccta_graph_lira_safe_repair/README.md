# CCTA / Graph-LIRA research scripts

This directory contains two different classes of code. Do not treat every historical script as current production/canonical code.

## A. Current collaborator-facing scripts

These are the scripts to use now:

- `verify_varvara_collab_pack.py` — integrity check for the complete CT28 PAIR handoff;
- `restore_ct28_pair_features.py` — restores the committed lossless PAIR feature table;
- `train_ct28_pair_from_features.py` — retrains/saves geometry, radial CT, and geometry+radial CT PAIR models;
- `restore_ct28_pair_models.py` — restores convenience regenerated model snapshots;
- `train_geometry_hgb_from_pair_plan.py` — reproduces the strong HGB geometry-only PAIR baseline.

For new JUNCTION+CT work, data access is documented in:

`docs/varvara/CT28_DATA_ACCESS.md`

## CT28 scene-level code

The selected working baseline is **K10/15**, recorded in `artifacts/varvara/ct28_graph_lira/selected_validation/selected_result.json`. A plain run of `train_ct28_relation_head.py` uses the earlier Logistic local baseline; it does not recreate K10/15's augmentation chain.

| Responsibility | Scripts |
|---|---|
| Shared local/relation models and structural inference | `train_ct28_relation_head.py`, `run_ct28_graph_lira.py`, `ct28_feature_precision.py` |
| Augmentation and copy-count experiments | `train_ct28_position_robust_*.py`, `train_ct28_*augmentation*.py`, `run_ct28_*copy_sweep.py`, `combine_ct28_copy_sweeps.py`, `evaluate_ct28_copy_sweep.py` |
| Selected-model consistency, failure and feature audits | `diagnose_ct28_selected_type.py`, `summarize_ct28_selected_failures.py`, `audit_ct28_junction_geometry_stability.py` |
| Accepted-repair path construction and anatomy evaluation | `ct28_path_stage.py`, `run_ct28_path_stage.py`, `audit_ct28_path_anatomy.py`, `audit_ct28_train_path_anatomy.py` |

Non-promoted relation/ranking, MLP/JS, and candidate path-evidence scripts are stored in `artifacts/varvara/ct28_graph_lira/ablations/rejected/rejected_experiment_scripts.tar.gz`. Restore them to their original paths using the command in that directory's README before reproducing those ablations. They are not required by the selected baseline.

Augmentation scripts are successive experiments with different reference packs, not interchangeable launchers for the selected model. Some historical input directories must first be restored from the archive described in `artifacts/varvara/ct28_graph_lira/ablations/rejected/README.md`. Keep helper module names stable: scripts import each other and saved model pipelines reference `ct28_feature_precision`.

## B. Archived/exploratory research scripts

Many historical scripts are stored as `.py.gz.b64`.

They are intentionally preserved because they correspond to earlier experiments and negative results. They are **research archive**, not the recommended entry point for a collaborator.

Restore one when a specific historical experiment must be audited:

```bash
base64 -d < script.py.gz.b64 | gzip -d > script.py
```

Then verify its protocol/SHA from the corresponding research checkpoint before treating a rerun as the same experiment.

Examples of historical code:

- six-patient CT pilots;
- old add/veto / hysteresis integration;
- mask-context experiments;
- geometry stress tests;
- old cross-patient and representation comparisons.

Do not start the current task by running these files in sequence.

## Missing historical runner

The old exact `run_graph_lira_large_scale.py` and its local `scenes_full.pkl/models.joblib/relation_type_model.joblib` were not persisted as canonical Git artifacts.

That is documented explicitly in:

`docs/varvara/ANSWER_TO_QUESTIONS_2026-09-26.md`

The old scientific results remain in `results/.../2026-09-20/` and the large-scale checkpoint documents.

## Current execution entry point

Read:

`docs/varvara/CURRENT_TASK.md`

then current results:

`artifacts/varvara/ct28_graph_lira/README.md`
