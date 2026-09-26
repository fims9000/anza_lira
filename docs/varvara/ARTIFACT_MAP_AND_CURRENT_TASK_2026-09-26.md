# Карта артефактов и текущая задача — 2026-09-26

Этот файл отвечает на вопросы по конкретным моделям/файлам и уточняет текущую архитектуру. Он важнее старых локальных путей из исследовательских заметок.

## 1. Что реально есть в Git

### Geometry HGB

Это отдельный сильный бинарный geometry-only baseline для 28-patient relation task.

Код обучения:
`scripts/research/ccta_graph_lira_safe_repair/run_expanded_geometry_baselines.py.gz.b64`

Результаты:
`results/ccta_graph_lira_safe_repair/2026-09-21/expanded_geometry_baselines_summary.csv`

На held-out test:
- AUROC 0.9685
- recall 37.72%
- FPR 1.20%
- precision 96.92%

Сериализованный `.joblib` модели в Git **не сохранён**. Модель воспроизводится из скрипта и frozen split.

### radial_hu_summary_v1 / geometry_plus_radial_v1

Это модели локальной pair-relation задачи на 28 matched CCTA пациентах.

Они не являются отдельными сохранёнными neural checkpoints.

Frozen runner:
`scripts/research/ccta_graph_lira_safe_repair/extract_radial_features_z04.py.gz.b64`

Протокол:
`experiments/ccta_graph_lira_safe_repair/expanded_ct_28case/README.md`

В этом эксперименте для:
- geometry only
- radial 2.5-D CT
- geometry + radial 2.5-D CT

используется одинаковый лёгкий классификатор:

`StandardScaler + LogisticRegression(C=1, class_weight=balanced)`

Порог выбирается только на validation при FPR <= 5%.

Именно `geometry_plus_radial_v1` дал CT28 held-out test:
- AUROC 0.9847
- recall 82.63%
- FPR 1.80%
- precision 97.87%

Важно: это **локальная бинарная pair-relation задача**, а не end-to-end Graph-LIRA и не четырёхклассовый relation head.

Сериализованных `.joblib` этих моделей в Git сейчас нет.

## 2. Что в Git отсутствует

Следующие старые локальные артефакты не находятся в канонической ветке:

- `run_graph_lira_large_scale.py`
- `graph_lira_large_scale/scenes_full.pkl`
- `graph_lira_selective/models.joblib`
- `graph_lira_relation_type/relation_type_model.joblib`

Их результаты сохранены в CSV/JSON и документации, но сами старые локальные binaries/source под этими именами не были перенесены в Git.

Не надо тратить время на их поиск: это реальный reproducibility gap старой исследовательской сессии.

Текущие зафиксированные результаты 800-case geometry Graph-LIRA находятся в:
- `docs/research/ccta_graph_lira_safe_repair/LARGE_SCALE_800_CASE.md`
- `docs/research/ccta_graph_lira_safe_repair/LARGE_SCALE_800_CASE_REPORT.md`
- `results/ccta_graph_lira_safe_repair/2026-09-20/`

## 3. Что было в canonical geometry Graph-LIRA

Нужно различать три уровня.

### A. pair_model

Локально ранжирует PAIR candidates по геометрическим признакам.

### B. junction_model

Локально ранжирует JUNCTION candidates по геометрическим признакам.

### C. relation-type head

После локальных score distributions используется отдельный scene-level head, который решает тип структурного действия:

`NONE / PAIR / JUNCTION / BOTH`

Для canonical 800-case benchmark этот relation-type head — **HistGradientBoosting (HGB)**.

Он обучался по out-of-fold train-score distributions, а confidence threshold был выбран на validation:

`tau = 0.85`

Это **не тот же объект**, что `geometry_hgb` из 28-patient binary pair experiment.

После relation-type decision работает global Graph-LIRA compatibility/optimization, затем selective gate:

`perturbation consistency >= 0.60`

## 4. Что делает ct_scene_graph_lira_hybrid.py

Файл:
`scripts/research/ccta_graph_lira_safe_repair/ct_scene_graph_lira_hybrid.py.gz.b64`

Это **старый six-patient pilot**, а не финальный 28-case implementation.

В нём canonical geometry остаётся ответственным за candidate identity/compatibility.

CT используется как дополнительный **relation-presence signal**.

То есть логика не была простой:
`pair geometry + geometry+CT + junction geometry -> новый общий head`.

Ближе к фактической схеме:

```text
pair_model / junction_model
        ↓
canonical geometry relation scores
        ↓
frozen relation-type head: NONE / PAIR / JUNCTION / BOTH
        ↓
auxiliary CT presence evidence
        ↓
add / veto (hysteresis pilot)
        ↓
Graph-LIRA compatibility
```

В six-case pilot CT add-only/hysteresis оказался небезопасным на held-out patients: false structural repair вырос. Поэтому этот integration rule **не надо переносить как готовое решение**.

## 5. PAIR_IMG и JUNC_IMG

Они относятся к six-case pilot и были отдельными image-conditioned presence signals для PAIR и JUNCTION.

Их смысл был правильный: CT должен давать evidence не только для PAIR, но и для JUNCTION.

Но эти модели были обучены на слишком маленьком matched-CT cohort и не стали canonical models.

Поэтому:
- использовать их старые веса как финальные — нет;
- использовать саму идею PAIR_CT / JUNCTION_CT — да.

## 6. Важное уточнение по текущему CT28 результату

CT28 experiment, который дал:

`AUROC 0.9847 / recall 82.63% / FPR 1.80% / precision 97.87%`

проверяет **pair-relation evidence** на geometry-matched positive/negative pairs.

Он ещё **не даёт полноценный 28-patient JUNCTION+CT signal**.

То есть вопрос про отсутствие junction+CT абсолютно правильный.

Это текущий недостающий блок перед честным full CT-conditioned Graph-LIRA.

## 7. Raw CT и expanded_relation_features.csv

Raw 28 CCTA volumes намеренно не хранятся в Git.

Источник и exact cohort описаны в:
`docs/research/ccta_graph_lira_safe_repair/EXPANDED_CT_28CASE_EXECUTION_AND_ROADMAP.md`

Используются 28 ImageCAS IDs из блока 953–984 с официальным split:
- 17 train
- 5 validation
- 6 test

Raw source: ImageCAS Kaggle, archive block `801-1000.z04`.

`expanded_relation_features.csv` создаётся runner'ом `extract_radial_features_z04.py`.

Сам CSV в текущей канонической ветке **не сохранён**. В Git сохранены compact metrics/provenance/alignment, но не full feature table.

Для полного collaborator reproduction его нужно либо:
1. заново получить из frozen runner + raw CT, либо
2. положить generated feature table в Git/релизный bundle, поскольку raw CT внутри него нет.

## 8. Текущая исследовательская задача

Не надо сейчас искать старые `.joblib` и пытаться буквально воскресить six-case hybrid.

Следующая чистая задача состоит из двух частей.

### Part A — 28-patient JUNCTION+CT evidence

На тех же 28 matched CCTA пациентах построить image evidence для junction candidates.

Не менять:
- patient split
- candidate generator
- geometry candidate models
- relation head protocol
- test thresholds

Нужно сравнить:
- geometry junction score
- CT junction evidence
- geometry + CT junction evidence

и проверить patient transfer.

### Part B — новый CT-conditioned four-class relation head

После того как есть:
- PAIR geometry + CT evidence
- JUNCTION geometry + CT evidence

строится единый scene-level input для:

`NONE / PAIR / JUNCTION / BOTH`

Сравнение должно быть:

```text
canonical geometry relation head
vs
CT-conditioned relation head
```

при frozen:
- candidate generation
- graph optimizer
- tau = 0.85 (или новый confidence threshold только если protocol заранее определён и выбирается исключительно на validation)
- perturbation consistency = 0.60
- held-out test без retuning

Первичный safe вариант — сначала сохранить canonical tau/policy и проверить, переносится ли CT gain вообще.

## 9. Что считать успехом

Не достаточно повысить AUROC.

На full Graph-LIRA нужно показать при фиксированном low-false режиме:
- repair-needed exact выше geometry-only;
- false structural repair не растёт существенно;
- coverage / abstention явно указан;
- результат устойчив по пациентам;
- отдельно проверены LAD / LCX / high-degree junctions.

После этого имеет смысл переходить к:
`radial CT vs compact CNN vs compact ANZA`.
