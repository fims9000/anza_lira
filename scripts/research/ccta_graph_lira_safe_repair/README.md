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

`docs/varvara/FINAL_HANDOFF_2026-09-27.md`

then:

`docs/varvara/CURRENT_TASK.md`
