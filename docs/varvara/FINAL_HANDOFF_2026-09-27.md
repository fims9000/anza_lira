# Варвара — финальный handoff текущего этапа

Дата: 2026-09-27  
Каноническая ветка: `research/coronary-connectivity-repair`

Этот файл — единственная точка входа после вопросов от 26 сентября. Старые заметки нужны только при конкретной необходимости.

## 1. Что теперь закрыто

Пробел с CT28 PAIR collaboration artifacts закрыт.

В Git есть:

- exact frozen `relation_pair_plan.csv` — 1,360 rows;
- row-level `expanded_relation_predictions.csv`;
- `expanded_relation_summary.csv`;
- `protocol.json`;
- CT alignment / expected geometry / pair counts / bootstrap summaries;
- lossless compressed payload полного `expanded_relation_features.csv`;
- скрипт восстановления feature table;
- скрипт обучения/сохранения трёх CT28 PAIR моделей;
- отдельный скрипт воспроизведения сильного `geometry_hgb`.

Поэтому ждать raw CT или старые `.joblib/.pkl` перед началом задачи больше не нужно.

## 2. Быстрая проверка PAIR baseline

После `git pull` из корня репозитория:

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_features.py
```

Затем:

```bash
python scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_from_features.py \
  artifacts/varvara/ct28_pair/expanded_relation_features.csv \
  --out-dir ct28_pair_models
```

Ожидаемый held-out результат `geometry_plus_radial_v1`:

- AUROC 0.9846893040;
- recall 82.6347%;
- FPR 1.7964%;
- precision 97.8723%;
- TP / FP / FN / TN = 138 / 3 / 29 / 164.

Повторное обучение на восстановленном feature table уже проверено: row-level scores совпадают с сохранёнными predictions до floating-point precision, thresholded predictions совпадают полностью.

Если нужны именно сохранённые PAIR `.joblib`, а не retrain, в Git теперь есть их compact payloads. Восстановить:

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_models.py
```

Это convenience snapshots; для научной воспроизводимости предпочтителен retrain из committed feature table.

Сильный geometry HGB воспроизводится отдельно:

```bash
python scripts/research/ccta_graph_lira_safe_repair/train_geometry_hgb_from_pair_plan.py
```

Ожидаемый held-out geometry HGB:

- AUROC 0.968482;
- recall 37.7246%;
- FPR 1.1976%;
- precision 96.9231%.

## 3. Что означает CT28 результат

Это **локальный бинарный PAIR relation experiment**.

Он доказывает, что для geometry-matched PAIR candidates CCTA context даёт дополнительный patient-general signal.

Он **не** доказывает улучшение полного Graph-LIRA и не покрывает JUNCTION.

Не смешивать:

- `geometry_hgb` — сильный binary PAIR geometry baseline;
- `geometry` в CT28 runner — logistic geometry-only baseline;
- `geometry_plus_radial_v1` — local PAIR geometry+CT model;
- canonical scene-level HGB — четырёхклассовый relation head NONE / PAIR / JUNCTION / BOTH.

## 4. Текущая новая задача

Главный незакрытый блок:

**28-patient JUNCTION + CT evidence.**

На тех же 28 пациентах и том же patient split нужно построить сравнение:

1. JUNCTION geometry;
2. JUNCTION CT;
3. JUNCTION geometry + CT.

После этого:

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

Первый Graph-LIRA test делаем без retuning на held-out patients.

## 5. Что не надо делать

Не надо:

- искать старые `scenes_full.pkl`, `models.joblib`, `relation_type_model.joblib` — их canonical copies не существовало;
- брать старый six-patient `ct_scene_graph_lira_hybrid.py` как финальную архитектуру;
- просто добавлять CT-positive relations поверх geometry head — этот pilot увеличивал false structural repair;
- заново оптимизировать PAIR baseline;
- начинать с большого CNN / Transformer / Mamba;
- использовать anatomical branch names как inference features;
- менять patient split;
- подбирать thresholds по test.

## 6. Что читать, если нужен контекст

В таком порядке:

1. этот файл;
2. `docs/varvara/CURRENT_TASK.md`;
3. `docs/varvara/NEGATIVE_RESULTS_THAT_MATTER.md`;
4. `docs/varvara/ANSWER_TO_QUESTIONS_2026-09-26.md`;
5. `docs/varvara/REPRODUCE_CT28_PAIR_BASELINE.md`.

Широкий research context и будущая статья остаются в `ARTICLE_DIRECTION.md` / полном roadmap, но для текущего кода они не нужны.

## 7. Что принести после первого JUNCTION+CT прохода

Минимальный результат:

- frozen junction dataset/plan с patient split;
- описание CT junction representation;
- geometry / CT / geometry+CT metrics на val и held-out test;
- row-level predictions;
- per-patient metrics;
- patient-cluster uncertainty;
- список failure cases;
- точные команды воспроизведения.

После этого вместе решаем, готов ли JUNCTION_CT к объединению с PAIR_CT в четырёхклассовый relation head.

## 8. Главный критерий

Цель не «максимальный AUROC любой ценой».

Итоговая задача — увеличить число правильных автоматических repairs при сохранении низкого false structural repair risk и возможности abstain на неоднозначных сценах.
