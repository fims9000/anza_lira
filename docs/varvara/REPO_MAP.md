# Карта репозитория для Варвары

Не нужно читать репозиторий сверху вниз.

## Главная ветка для работы

`research/varvara-ccta-graph-lira`

Она основана на полной технической ветке `research/ccta-graph-lira-safe-repair`, поэтому код, результаты и reproducibility внутри сохранены.

## С чего начинать

`VARVARA_START_HERE.md`

## Текущая статья / направление

`docs/varvara/ARTICLE_DIRECTION.md`

## Самые важные результаты

`docs/varvara/RESULTS_TO_USE.md`

## Что делать дальше

`docs/varvara/ROADMAP.md`

## Подробный технический checkpoint CT28

`docs/research/ccta_graph_lira_safe_repair/REAL_CT28_RESULTS_AND_PROMOTION.md`

## Общий research checkpoint

`docs/research/ccta_graph_lira_safe_repair/CHECKPOINT.md`

## Related work

`docs/research/ccta_graph_lira_safe_repair/RELATED_WORK_AND_DIFFERENTIATION_2026.md`

## Frozen baseline matrix

`experiments/ccta_graph_lira_safe_repair/BASELINE_MATRIX.md`

## 28-patient experiment

`experiments/ccta_graph_lira_safe_repair/expanded_ct_28case/`

Там лежат frozen cohort metadata и expected geometry.

## Основные компактные результаты

`results/ccta_graph_lira_safe_repair/2026-09-21/real_ct28_headline_test.csv`

`results/ccta_graph_lira_safe_repair/2026-09-21/real_ct28_patient_cluster_bootstrap_ci.csv`

`results/ccta_graph_lira_safe_repair/2026-09-21/real_ct28_paired_patient_cluster_bootstrap_ci.csv`

`results/ccta_graph_lira_safe_repair/2026-09-21/real_ct28_multibudget_risk_coverage.csv`

`results/ccta_graph_lira_safe_repair/2026-09-21/real_ct28_anatomy_subgroups.csv`

`results/ccta_graph_lira_safe_repair/2026-09-21/real_ct28_per_patient.csv`

`results/ccta_graph_lira_safe_repair/2026-09-21/real_ct28_alignment.csv`

## Воспроизводимость CT feature extraction

`scripts/research/ccta_graph_lira_safe_repair/extract_radial_features_z04.py.gz.b64`

## Анализ expanded CT

`scripts/research/ccta_graph_lira_safe_repair/analyze_expanded_ct_results.py.gz.b64`

## Geometry baselines

`scripts/research/ccta_graph_lira_safe_repair/run_expanded_geometry_baselines.py.gz.b64`

`scripts/research/ccta_graph_lira_safe_repair/bootstrap_geometry_baselines.py.gz.b64`

## Старые exploratory результаты

Они оставлены в технических каталогах для истории и проверки решений.

Для основной работы их не надо читать, пока не возникнет конкретный вопрос:

- почему отказались от конкретного feature;
- какой baseline уже пробовали;
- почему не используем конкретный threshold/representation.

В статье в первую очередь использовать curated results из `docs/varvara/RESULTS_TO_USE.md`.
