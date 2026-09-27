# Ответ на вопросы по артефактам и текущей задаче

Дата: 2026-09-26

Этот файл можно использовать как короткий handoff после вопросов о моделях, старых Graph-LIRA артефактах и CT28.

## 1. geometry_hgb

Да, это `HistGradientBoosting` над геометрическими признаками PAIR-кандидата.

Код:
`scripts/research/ccta_graph_lira_safe_repair/run_expanded_geometry_baselines.py.gz.b64`

Отдельного canonical `geometry_hgb.joblib` в Git нет. Модель воспроизводится из frozen split и кода.

Важно: это не тот же объект, что scene-level HGB relation head в canonical Graph-LIRA.

## 2. radial_hu_summary_v1 / geometry_plus_radial_v1

Это локальные бинарные PAIR-модели CT28.

Обе обучаются как:

`StandardScaler + LogisticRegression(C=1, class_weight=balanced)`

- `radial_hu_summary_v1`: только radial CCTA features;
- `geometry_plus_radial_v1`: geometry + radial CCTA features.

Именно `geometry_plus_radial_v1` дал held-out CT28:

- AUROC 0.9847;
- recall 82.63%;
- FPR 1.80%;
- precision 97.87%.

Это **не** full Graph-LIRA result.

## 3. Старые .pkl / .joblib

В canonical branch нет:

- `run_graph_lira_large_scale.py`;
- `graph_lira_large_scale/scenes_full.pkl`;
- `graph_lira_selective/models.joblib`;
- `graph_lira_relation_type/relation_type_model.joblib`.

Это не скрытые файлы: они действительно не были сохранены как canonical artifacts.

Результаты тех прогонов сохранены в CSV/JSON/docs.

## 4. Что было до CT

Canonical geometry pipeline:

```text
PAIR candidate geometry scores
+
JUNCTION candidate geometry scores
        ↓
scene-level HGB relation head
        ↓
NONE / PAIR / JUNCTION / BOTH
        ↓
global Graph-LIRA
        ↓
confidence / perturbation consistency
        ↓
repair / abstain
```

Relation head использует out-of-fold train score distributions.

Canonical selective values:

- tau = 0.85;
- consistency = 0.60.

## 5. ct_scene_graph_lira_hybrid.py

Это старый six-patient pilot.

В нём CT использовался скорее как auxiliary presence/add-veto signal поверх canonical geometry relation logic.

Он не является готовой 28-patient architecture.

На held-out test CT add-only увеличил false structural repair, поэтому старый integration rule повторять не надо.

## 6. PAIR_IMG / JUNC_IMG

Это exploratory six-case image-presence models.

Идея разделить image evidence на PAIR и JUNCTION остаётся полезной, но старые веса не являются canonical.

Текущий CT28 сильный результат пока доказан для PAIR.

Полноценный 28-patient JUNCTION+CT signal — следующий незакрытый блок.

## 7. Что делать сейчас

### A. JUNCTION+CT

На тех же 28 patients и том же split:

- geometry junction;
- CT junction;
- geometry + CT junction.

### B. CT-conditioned relation head

После появления PAIR_CT и JUNCTION_CT:

```text
PAIR geometry + PAIR CT
+
JUNCTION geometry + JUNCTION CT
        ↓
NONE / PAIR / JUNCTION / BOTH
```

### C. frozen Graph-LIRA

Не менять graph optimizer и не tune test.

Сначала проверить переносимость при frozen safety policy.

## 8. Где начинать

Прочитать:

1. `docs/varvara/CURRENT_TASK.md`
2. `docs/varvara/NEGATIVE_RESULTS_THAT_MATTER.md`
3. `docs/varvara/REPRODUCE_CT28_PAIR_BASELINE.md`
4. `docs/varvara/ARTIFACT_MAP_AND_CURRENT_TASK_2026-09-26.md`

Компактные артефакты:

`artifacts/varvara/ct28_pair/`

Полный CT28 PAIR collaboration pack теперь закрыт в Git:

- `artifacts/varvara/ct28_pair/relation_pair_plan.csv`;
- `artifacts/varvara/ct28_pair/expanded_relation_predictions.csv`;
- `artifacts/varvara/ct28_pair/expanded_relation_summary.csv`;
- `artifacts/varvara/ct28_pair/protocol.json`;
- полный feature table хранится losslessly в `artifacts/varvara/ct28_pair/payload/expanded_relation_features.csv.xz.b64`.

Восстановить `expanded_relation_features.csv`:

`python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_features.py`

Если нужны именно сохранённые PAIR model snapshots, они тоже добавлены в Git в `artifacts/varvara/ct28_pair/models_payload/` и восстанавливаются командой:

`python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_models.py`

Для переносимой воспроизводимости предпочтительнее retrain:

`scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_from_features.py`

Повторный train уже проверен: scores совпадают с сохранёнными row-level predictions до floating-point precision, thresholded predictions совпадают полностью.

## 9. Что не нужно сейчас делать

- не восстанавливать старые local pickle-файлы;
- не тюнить geometry ещё раз;
- не повторять six-case add-only CT integration;
- не уходить сразу в большой CNN / Transformer / Mamba;
- не считать CT28 PAIR result полноценным JUNCTION/Graph-LIRA result.


## 10. Обновление 2026-09-27 — данных для PAIR baseline больше ждать не нужно

Пробел с `expanded_relation_features.csv` и row-level predictions закрыт. Артефакты честно перегенерированы из frozen protocol и сохранены в collaboration pack.

Поэтому текущий старт теперь такой:

1. `git pull` канонической ветки;
2. при желании воспроизвести PAIR baseline из committed feature payload;
3. не тратить время на старые отсутствующие pickle-файлы;
4. переходить к `JUNCTION+CT` из `docs/varvara/CURRENT_TASK.md`.
