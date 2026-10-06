# Варвара — начать отсюда

Каноническая рабочая ветка:

`research/coronary-connectivity-repair`

С 2026-10-06 именно она снова является текущей точкой сборки всей coronary /
CCTA линии. JUNCTION-работа из `experiment/ct28-junction` уже перенесена
сюда.

После обновления репозитория сначала открыть:

`docs/varvara/CURRENT_TASK.md`

Это единственный текущий task/handoff документ.

## Что уже находится в canonical branch

### PAIR

Frozen CT28 PAIR pack:

- 28 patients;
- 17 train / 5 validation / 6 held-out test;
- reproducible features / predictions / models;
- `geometry_plus_radial_v1` held-out:
  - AUROC 0.9847;
  - recall 82.63%;
  - FPR 1.80%;
  - precision 97.87%.

Integrity check:

```bash
python scripts/research/ccta_graph_lira_safe_repair/verify_varvara_collab_pack.py
```

### JUNCTION

В canonical branch уже находятся:

- `build_junction_plan.py`;
- `extract_junction_ct_features.py`;
- `train_junction_baselines.py`;
- `train_junction_lira_fusion.py`;
- текущие baseline metrics.

JUNCTION held-out test пока не открываем при выборе архитектуры/thresholds.

## Что больше не является текущей инструкцией

Не использовать как current task:

- `docs/varvara/FINAL_HANDOFF_2026-09-27.md`;
- `docs/varvara/JUNCTION_GENERATOR_REVIEW_2026-10-03.md`;
- `docs/varvara/JUNCTION_WORK_BRANCH.md`.

Они оставлены только как история решений.

Не продолжать работу в:

- `experiment/ct28-junction`;
- `experiment/junction-ct28`;
- `research/varvara-ccta-graph-lira`.

Текущие решения и ближайшие действия — только в
`docs/varvara/CURRENT_TASK.md`.
