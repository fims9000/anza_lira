# HISTORICAL HANDOFF — superseded

This document is preserved for provenance of the 2026-09-27 state. It is **not** the current task. For current work use `docs/varvara/CURRENT_TASK.md` on `research/coronary-connectivity-repair`.

# Варвара — финальный handoff текущего этапа

Дата: 2026-09-27  
Каноническая ветка: `research/coronary-connectivity-repair`

Это единственная текущая точка входа после вопросов по моделям, старым Graph-LIRA artifacts и CT28 данным.

## 1. Сначала проверить, что pack целый

После `git pull` из корня репозитория:

```bash
python scripts/research/ccta_graph_lira_safe_repair/verify_varvara_collab_pack.py
```

Ожидаемое начало вывода:

`PASS: Varvara CT28 collaborator pack is internally consistent.`

Этот check проверяет текущие handoff files, hashes compact payloads, frozen 1,360-row pair plan и headline metrics.

## 2. Что теперь закрыто

CT28 PAIR collaboration pack complete.

В Git есть:

- exact frozen `relation_pair_plan.csv`;
- row-level `expanded_relation_predictions.csv`;
- `expanded_relation_summary.csv`;
- `protocol.json`;
- CT alignment / expected geometry / bootstrap;
- lossless compressed full `expanded_relation_features.csv`;
- restore/retrain scripts;
- regenerated snapshots трёх lightweight PAIR моделей;
- отдельный reproducible `geometry_hgb` training script.

Canonical pack:

`artifacts/varvara/ct28_pair/`

## 3. Быстрая проверка PAIR baseline

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_features.py
```

```bash
python scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_from_features.py \
  artifacts/varvara/ct28_pair/expanded_relation_features.csv \
  --out-dir ct28_pair_models
```

Expected held-out `geometry_plus_radial_v1`:

- AUROC 0.9846893040;
- recall 82.6347%;
- FPR 1.7964%;
- precision 97.8723%;
- TP / FP / FN / TN = 138 / 3 / 29 / 164.

Convenience snapshots:

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_models.py
```

Strong binary PAIR geometry HGB:

```bash
python scripts/research/ccta_graph_lira_safe_repair/train_geometry_hgb_from_pair_plan.py
```

## 4. Не путать четыре объекта

- `geometry_hgb` — strong binary PAIR geometry HGB baseline;
- `geometry` in CT28 runner — lightweight logistic geometry-only baseline;
- `geometry_plus_radial_v1` — binary PAIR geometry+CCTA model;
- old scene-level HGB — historical four-class `NONE/PAIR/JUNCTION/BOTH` head.

CT28 result above относится **только к local binary PAIR relation**.

Это ещё не full CT-conditioned Graph-LIRA.

## 5. Ответ по старым Graph-LIRA files

Canonical Git copies отсутствуют:

- `run_graph_lira_large_scale.py`;
- `graph_lira_large_scale/scenes_full.pkl`;
- `graph_lira_selective/models.joblib`;
- `graph_lira_relation_type/relation_type_model.joblib`.

Это historical reproducibility gap старого geometry-only execution, не ошибка clone.

Не надо их искать.

Что известно про старую architecture/results сохранено в large-scale checkpoint docs/results. Exact old local binaries не восстанавливаем из агрегированных цифр.

Полные ответы на исходные вопросы:

`docs/varvara/ANSWER_TO_QUESTIONS_2026-09-26.md`

## 6. Старый CT hybrid

`ct_scene_graph_lira_hybrid.py.gz.b64` — six-patient pilot.

Он использовал CT как auxiliary add/veto/presence evidence поверх geometry logic.

Direct insertion на held-out patients увеличивал false structural repair.

Поэтому не использовать его как новую финальную 28-patient architecture.

Old `PAIR_IMG/JUNC_IMG` weights тоже exploratory и не canonical.

## 6.1. Важный статус кода Graph-LIRA

Для текущего JUNCTION+CT этапа отсутствие старого `run_graph_lira_large_scale.py` не блокирует работу.

Но позже мы **не будем** делать вид, что старый Graph-LIRA runner аккуратно лежит где-то готовый к импорту. Перед full integration будет создан новый canonical runner/module с явными входами и сохранением protocol/results. Старые archived scripts используются как историческая реализация/референс, а не как скрытая зависимость.

То есть сейчас задача — JUNCTION+CT. Восстанавливать старые pickle/scenes перед этим не требуется.

## 7. Текущий незакрытый научный блок

**28-patient JUNCTION+CT evidence.**

На тех же 28 пациентах и том же frozen split:

1. JUNCTION geometry;
2. JUNCTION CT;
3. JUNCTION geometry + CT.

Raw CT для **этой новой feature extraction** нужен.

Exact source/cohort/archive/alignment:

`docs/varvara/CT28_DATA_ACCESS.md`

Для старого PAIR reproduction raw CT не нужен.

## 8. После JUNCTION+CT

```text
PAIR geometry + PAIR CT
+
JUNCTION geometry + JUNCTION CT
        ↓
CT-conditioned scene-level relation head
        ↓
NONE / PAIR / JUNCTION / BOTH
        ↓
frozen Graph-LIRA
        ↓
tau = 0.85
consistency = 0.60
        ↓
repair / abstain
```

Первый full-graph comparison — без held-out test retuning.

## 9. Что не делать

Не надо:

- искать старые отсутствующие pkl/joblib;
- заново тюнить frozen PAIR baseline;
- менять 17/5/6 patient split;
- использовать anatomical branch names как inference features;
- повторять old CT add-only/hysteresis rule как final;
- начинать с большого CNN / Transformer / Mamba;
- заявлять ANZA superiority до clean ablation.

## 10. Минимум, который принести после первого JUNCTION+CT прохода

- frozen junction dataset/plan;
- representation description;
- geometry / CT / geometry+CT metrics on val and held-out test;
- row-level predictions;
- per-patient metrics;
- patient-cluster uncertainty;
- failure-case list;
- exact reproduction commands.

## 11. Если нужен контекст

В таком порядке:

1. `docs/varvara/CURRENT_TASK.md`
2. `docs/varvara/ANSWER_TO_QUESTIONS_2026-09-26.md`
3. `docs/varvara/NEGATIVE_RESULTS_THAT_MATTER.md`
4. `docs/varvara/REPRODUCE_CT28_PAIR_BASELINE.md`
5. `docs/varvara/CT28_DATA_ACCESS.md`

Главная цель остаётся не «максимальный AUROC», а больше правильных automatic repairs при контролируемом false structural repair risk и возможности abstain.