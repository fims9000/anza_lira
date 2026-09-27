# Карта репозитория для Варвары

Каноническая ветка:

`research/coronary-connectivity-repair`

Не нужно читать репозиторий сверху вниз.

## 1. Единственная точка входа сейчас

Сначала:

`docs/varvara/FINAL_HANDOFF_2026-09-27.md`

После него — только если нужен конкретный уровень деталей:

- `docs/varvara/CURRENT_TASK.md` — текущая задача JUNCTION+CT;
- `docs/varvara/ANSWER_TO_QUESTIONS_2026-09-26.md` — ответы по HGB, pair models, старым .pkl/.joblib, PAIR_IMG/JUNC_IMG;
- `docs/varvara/NEGATIVE_RESULTS_THAT_MATTER.md` — отрицательные результаты, которые реально ограничивают архитектуру;
- `docs/varvara/REPRODUCE_CT28_PAIR_BASELINE.md` — воспроизведение CT28 PAIR baseline.

Широкий научный контекст:
- `VARVARA_START_HERE.md`;
- `docs/varvara/ARTICLE_DIRECTION.md`;
- `docs/varvara/RESULTS_TO_USE.md`;
- `docs/varvara/ROADMAP.md`.

## 2. CT28 PAIR collaborator pack — уже complete

Каталог:

`artifacts/varvara/ct28_pair/`

В нём есть:

- `relation_pair_plan.csv` — exact frozen 1,360-row plan;
- `expanded_relation_predictions.csv` — row-level predictions;
- `expanded_relation_summary.csv`;
- `protocol.json`;
- `ct_alignment.csv`;
- `expected_ct_geometry.csv`;
- `pair_plan_counts.csv`;
- `headline_test.csv`;
- `paired_bootstrap_ci.csv`;
- `payload/expanded_relation_features.csv.xz.b64` — lossless compressed full feature table;
- `models_payload/` — compact regenerated PAIR model snapshots.

Восстановить feature table:

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_features.py
```

Retrain/save PAIR models:

```bash
python scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_from_features.py \
  artifacts/varvara/ct28_pair/expanded_relation_features.csv \
  --out-dir ct28_pair_models
```

Восстановить convenience model snapshots:

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_models.py
```

Сильный HGB geometry baseline:

```bash
python scripts/research/ccta_graph_lira_safe_repair/train_geometry_hgb_from_pair_plan.py
```

Raw multi-GB CCTA для уже frozen PAIR baseline не нужен. Он понадобится только при изменении image representation / повторном feature extraction.

## 3. Что было старым Graph-LIRA и не надо искать

Canonical old copies отсутствуют:

- `run_graph_lira_large_scale.py`;
- `graph_lira_large_scale/scenes_full.pkl`;
- `graph_lira_selective/models.joblib`;
- `graph_lira_relation_type/relation_type_model.joblib`.

Это исторический reproducibility gap, а не ошибка клонирования.

Сохранённые результаты старого geometry Graph-LIRA:

- `docs/research/ccta_graph_lira_safe_repair/LARGE_SCALE_800_CASE.md`;
- `docs/research/ccta_graph_lira_safe_repair/LARGE_SCALE_800_CASE_REPORT.md`;
- `results/ccta_graph_lira_safe_repair/2026-09-20/`.

## 4. Текущий научный блок

Уже есть patient-general CT28 PAIR evidence.

Сейчас не хватает:

**28-patient JUNCTION+CT evidence.**

После него:

`PAIR_CT + JUNCTION_CT -> CT-conditioned NONE/PAIR/JUNCTION/BOTH -> frozen Graph-LIRA`.

До завершения этого шага не надо уходить в большой CNN / Transformer / Mamba и не надо объявлять ANZA частью финального метода.

## 5. Полезные technical paths

- `scripts/research/ccta_graph_lira_safe_repair/` — reproducibility/training helpers;
- `experiments/ccta_graph_lira_safe_repair/expanded_ct_28case/` — frozen cohort/protocol;
- `results/ccta_graph_lira_safe_repair/2026-09-21/` — CT28 result tables;
- `docs/research/ccta_graph_lira_safe_repair/REAL_CT28_RESULTS_AND_PROMOTION.md` — подробный CT28 checkpoint;
- `docs/research/ccta_graph_lira_safe_repair/RELATED_WORK_AND_DIFFERENTIATION_2026.md` — related work.

Главное правило: если старый документ противоречит `FINAL_HANDOFF_2026-09-27.md`, текущим считается FINAL_HANDOFF.
