# Карта артефактов и текущая задача

Актуализировано: 2026-09-27  
Каноническая ветка: `research/coronary-connectivity-repair`

## Current source of truth

Начинать с:

1. `docs/varvara/FINAL_HANDOFF_2026-09-27.md`
2. `docs/varvara/CURRENT_TASK.md`
3. `docs/varvara/ANSWER_TO_QUESTIONS_2026-09-26.md`

## CT28 PAIR — complete collaboration pack

Canonical directory:

`artifacts/varvara/ct28_pair/`

There are:

- `relation_pair_plan.csv` — exact 1,360-row frozen pair plan;
- `expanded_relation_predictions.csv`;
- `expanded_relation_summary.csv`;
- `protocol.json`;
- `ct_alignment.csv`;
- `expected_ct_geometry.csv`;
- `headline_test.csv`;
- `paired_bootstrap_ci.csv`;
- `payload/expanded_relation_features.csv.xz.b64`;
- `models_payload/geometry.joblib.gz.b64`;
- `models_payload/radial_hu_summary_v1.joblib.gz.b64`;
- `models_payload/geometry_plus_radial_v1.joblib.gz.b64`.

Restore features:

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_features.py
```

Retrain PAIR models:

```bash
python scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_from_features.py \
  artifacts/varvara/ct28_pair/expanded_relation_features.csv \
  --out-dir ct28_pair_models
```

Restore convenience PAIR snapshots:

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_models.py
```

Strong geometry HGB:

```bash
python scripts/research/ccta_graph_lira_safe_repair/train_geometry_hgb_from_pair_plan.py
```

Integrity check:

```bash
python scripts/research/ccta_graph_lira_safe_repair/verify_varvara_collab_pack.py
```

## Model-name map

### `geometry_hgb`

Strong binary PAIR geometry-only HGB baseline.

Not the old scene-level four-class Graph-LIRA HGB.

### `geometry`

Lightweight logistic geometry-only model inside the CT28 three-way local ablation.

### `radial_hu_summary_v1`

Lightweight logistic binary PAIR model using radial 2.5-D CCTA summaries.

### `geometry_plus_radial_v1`

Lightweight logistic binary PAIR model using geometry + radial CCTA.

This is the model whose held-out CT28 result is:

- AUROC 0.9847;
- recall 82.63%;
- FPR 1.80%;
- precision 97.87%.

### old scene-level relation head

Historical HGB predicting:

`NONE / PAIR / JUNCTION / BOTH`

from out-of-fold geometry score distributions.

Its exact old training runner/model pickle was not persisted.

## Historical files that do not exist

Do not search for:

- `run_graph_lira_large_scale.py`;
- `graph_lira_large_scale/scenes_full.pkl`;
- `graph_lira_selective/models.joblib`;
- `graph_lira_relation_type/relation_type_model.joblib`.

Those exact old local artifacts were not committed.

This is documented, not hidden.

## Old CT pilot

`ct_scene_graph_lira_hybrid.py.gz.b64` is a six-patient pilot.

Old `PAIR_IMG/JUNC_IMG` are pilot image-presence models.

They are useful historical evidence, not canonical current models.

Naive CT add/veto integration increased held-out false structural repair, so it is explicitly **not** the implementation target.

## Raw CT

Raw CT is intentionally outside Git.

- not needed for frozen PAIR reproduction;
- required for new JUNCTION+CT feature extraction.

See:

`docs/varvara/CT28_DATA_ACCESS.md`

## Exact next task

```text
JUNCTION geometry
vs
JUNCTION CT
vs
JUNCTION geometry + CT
        ↓
PAIR_CT + JUNCTION_CT
        ↓
CT-conditioned NONE / PAIR / JUNCTION / BOTH
        ↓
frozen Graph-LIRA
        ↓
repair / abstain
```

Keep frozen:

- patient split;
- candidate-generation protocol;
- held-out test discipline;
- initial Graph-LIRA safety policy;
- `tau=0.85`;
- perturbation consistency `0.60`.

Do not move to CNN / ANZA / Transformer until this integration question is answered.
