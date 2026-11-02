# CT28: consistency и причины abstain

Код невыбранных экспериментов перенесён в [архив](../ablations/rejected/README.md); команды ниже используют исходные пути после его восстановления.
Все числа здесь — validation по пяти пациентам; held-out test не открывался. Фиксированы `tau=0.85`, порог exact-set consistency `0.60` и 15 **оценочных** повторов с `σ=0.8 мм` для позиции и `σ=20°` для касательных. Train-копии, описанные в [истории версий](../ablations/README.md), создавались отдельно и не являются этими 15 проверками.

Ранние подробные CSV/JSON этого отчёта упакованы локально в `varvara/anza_lira_local_results/ct28_rejected_archive/consistency_diagnostics.tar.gz`; файл `../ablations/rejected/archive_manifest.json` содержит SHA256 каждого из 29 исходных файлов. Для восстановления исходных путей из корня репозитория: `tar -xzf varvara/anza_lira_local_results/ct28_rejected_archive/consistency_diagnostics.tar.gz -C .`. Диагностика **выбранного K10/15** по-прежнему находится в локальной `varvara/anza_lira_local_results/ct28_copy_count_ablations/selected_*`.

## Определения

`Raw type consistency` — доля повторов с тем же argmax среди `NONE/PAIR/JUNCTION/BOTH` **до** `tau`; `tau type` — тот же тип сохранился и прошёл `tau`; `identity given type` — при сохранённом типе выбраны те же candidate IDs; `exact-set agreement` — совпали тип и весь набор candidate IDs. Финальный gate требует, чтобы исходный полный набор IDs был модой повторов и exact-set consistency достигла 0.60. Даже высокая отдельная confidence не гарантирует прохождения gate. `Proposed` — исходная сцена, для которой до consistency предложен repair; `accepted` — прошедший gate repair. Для настоящей `NONE` оставить сцену без repair — правильное решение.

## Ранний clean HGB и Logistic: почему потребовалась работа над consistency

Это **ранняя** пара моделей с fusion identity для обоих типов; она не равна выбранному K10/15. Relation head в обеих — HGB, различается семейство local models. В таблице средние по предложившим repair сценам:

| Local models | Proposed | Accepted | Base confidence | Perturbed confidence | Доля confidence ≥ tau | Raw type | Tau type | Identity при том же type | Exact-set agreement | Combined consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic | 164 | 14 | 97.93% | 85.93% | 66.54% | 64.19% | 47.68% | 50.12% | 24.27% | 15.77% |
| HGB | 190 | 0 | 98.61% | 89.23% | 73.58% | 39.86% | 29.33% | 61.63% | 17.19% | 3.09% |

**Сравнение:** опора — ранний clean Logistic local при той же HGB relation head и fusion identity. HGB local выше по perturbed confidence, но по принятому exact repair лучше Logistic (14 против 0). **Ни одна строка не стала текущим K10/15**; позднее менялись local augmentation, identity и relation training.

| Local models | Исходный тип | Proposed сцен | Tau-type consistency | Identity при том же type | Exact-set agreement | Прошли gate |
|---|---|---:|---:|---:|---:|---:|
| Logistic | PAIR | 118 | 64.69% | 50.64% | 33.16% | 14 |
| Logistic | JUNCTION | 20 | 8.00% | 44.87% | 2.67% | 0 |
| Logistic | BOTH | 26 | 1.03% | 50.00% | 0.51% | 0 |
| HGB | PAIR | 125 | 40.48% | 61.03% | 24.27% | 0 |
| HGB | JUNCTION | 31 | 12.47% | 59.04% | 4.52% | 0 |
| HGB | BOTH | 34 | 3.73% | 71.88% | 2.75% | 0 |

**Сравнение:** та же ранняя пара Logistic/HGB; PAIR/JUNCTION/BOTH — разные исходные типы, их численности не ранжируются между собой. По прохождению gate Logistic лучше только для PAIR (14 против 0), для JUNCTION/BOTH обе модели дают 0. **Таблица диагностическая**, текущий вариант её заменил.

После `tau` большинство ранних JUNCTION/BOTH превращались в `NONE`: Logistic JUNCTION 92.0% и BOTH 85.4%; HGB JUNCTION 84.1% и BOTH 74.9%. У HGB 73.58% повторов всё же имели confidence ≥0.85, но нередко уверенно выбирали **другой тип**. Endpoint-disjoint compatibility не объясняла отказ: когда повтор сохранял BOTH, совместимая комбинация находилась. До gate CT-conditioned Logistic давала 50.97% exact среди repair-needed и 15.49% false среди всех сцен; после gate 14 exact и 0 false. Clean HGB давала 85.44% exact, 3.53% false до gate, затем 0 accepted.

Для 206 repair-needed сцен каждой ранней модели первичная причина отказа назначалась по порядку `accepted → correct_none → low confidence → no repair proposed → compatibility failure → type unstable → PAIR/JUNCTION identity unstable → BOTH combination unstable → exact-set mode failure`. Это диагностическая классификация, а не изменение решения:

| Первичный исход | Logistic | HGB |
|---|---:|---:|
| Принято | 14 | 0 |
| Нестабилен relation type | 76 | 162 |
| Нестабилен PAIR identity | 69 | 21 |
| Нестабилен JUNCTION identity | 0 | 1 |
| Confidence ниже `tau` | 38 | 13 |
| Repair не предложен | 9 | 9 |

**Сравнение:** опора — ранний Logistic; по числу принятых repair он выше HGB (14 против 0), но остальные строки — причины отказа, где «лучшей» строки нет. **Оба распределения исторические**, текущий разбор приведён ниже.

Из 162 истинных `NONE` Logistic оставила 157 без repair, HGB — 156; остальные ошибочные предложения отклонены gate. Ранние машинные таблицы (`clean_hgb/`, `clean_logistic/`, `consistency_components_val.csv`, `abstention_reasons_val.csv`, `consistency_diagnostics.json`) находятся в указанном выше локальном архиве, не в текущей папке.

## Выбранный K10/15: разбор всех 368 validation-сцен

PAIR identity использует fusion score, JUNCTION — geometry score. На clean: **155/206 structural exact и 11/368 false до gate; после gate 52 exact, 0 false, 0 incomplete**, в том числе 48 `pair_only`, 2 `junction_only`, 2 `mixed`. Сохранённые 5 520 решений joint-noise повторов воспроизведены с максимальным отклонением вероятности `1.11e-16`. Для 206 repair-needed сцен первичные исходы:

| Исход | PAIR | JUNCTION | BOTH | Всего |
|---|---:|---:|---:|---:|
| Принято точно | 48 | 2 | 2 | 52 |
| Нестабилен relation type | 38 | 16 | 20 | 74 |
| Confidence ниже `tau` | 20 | 12 | 11 | 43 |
| Нестабилен PAIR identity | 27 | 0 | 0 | 27 |
| Нестабилен JUNCTION identity | 0 | 6 | 0 | 6 |
| Нестабильна комбинация BOTH | 0 | 0 | 3 | 3 |
| Repair не предложен | 1 | 0 | 0 | 1 |

**Основа:** только выбранный K10/15; PAIR/JUNCTION/BOTH — классы, а не конкурирующие модели. Наибольший вклад в 52 принятых — PAIR (48); это **текущая диагностика**, заменять модель по этой таблице нечем.

Из 162 `NONE`-сцен 157 остались без repair, ещё пять ошибочных предложений отклонены gate. `varvara/anza_lira_local_results/ct28_copy_count_ablations/selected_diagnostics/abstention_reasons_val.csv` содержит полный scene-level разбор; там же `failure_cases_val.csv` и `perturbation_decisions_val.csv`. В `selected_type_diagnostics/` лежат четыре вероятности, все 84 scene features, local scores, оценки центров и trace для joint/position/tangent шумов.

Названия accuracy в разных JSON относятся к разным стадиям: `raw` в relation metrics — argmax до `tau` (**88.86%** для K10/15); `tau_relation_type_accuracy` в structural metrics — после `tau`, до consistency (**85.60%**); `relation_type_accuracy` там же — после abstain, когда отклонённый repair считается `NONE` (**58.15%**). Последнее число не является качеством исходного relation head.

Среди 36 `junction_only` clean raw top class — JUNCTION в 31, а после `tau` в 23; для 36 `mixed` raw BOTH в 30, после `tau` в 24. Таблица ниже относится **только** к 23 и 24 исходно post-`tau` правильным сценам; 15 повторов на сцену не являются новыми независимыми пациентами:

| Joint-noise повторы | Raw type верен | Тот же type проходит `tau` | Сохранился exact candidate set | Тип тот же, confidence < `tau` |
|---|---:|---:|---:|---:|
| `junction_only`: 345 | 293 | 155 | 80 | 138 |
| `mixed`: 360 | 269 | 91 | 78 | 178 |

**Основа:** K10/15 с динамическими scores при joint-noise; `junction_only` и `mixed` имеют разные знаменатели и не являются альтернативными моделями. Таблица показывает отказы, **не выбирает замену** текущему варианту.

Для JUNCTION взаимно исключающие причины на 345 повторах: 52 смены raw type, 138 падений confidence при том же типе, 75 смен candidate set после прохождения `tau`, 80 стабильных. Для mixed: 91, 178, 13, 78. Значит, исчезновение JUNCTION/BOTH — не всегда ошибка argmax; часто **тот же тип** опускается ниже неизменного `tau`.

Position noise разрушает больше, чем tangent: у тех же исходно верных сцен post-`tau` JUNCTION в 166/345 position-only против 293/345 tangent-only; BOTH в 106/360 против 271/360. Диагностическая фиксация clean JUNCTION geometry score stream при joint noise повышала JUNCTION post-`tau` с 155 до 244/345; фиксация JUNCTION CT — до 178. Для mixed фиксация JUNCTION geometry давала 128/360, PAIR geometry — 134/360 вместо 91. **Ни одна фиксация не является допустимым inference**: clean scores неизвестны при реальном ошибочном endpoint set.

Среди 65 competitive JUNCTION-сцен (`junction_only` и `mixed`) истинный candidate выше лучшего ложного в 691/975 joint, 721/975 position-only, 861/975 tangent-only повторах; семь остальных сцен не имеют ложного JUNCTION candidate в pool. Следовательно, потеря coverage происходит и на relation type/confidence, и на candidate identity.

Адресные ablations на **той же** K10/15 схеме: удаление 22 JUNCTION geometry-derived relation inputs (18 summary и 4 cross terms) оставило 62 признака и дало 117/206 exact, 7/368 false до gate, 22/0 после, 0 JUNCTION/mixed. Замена local JUNCTION geometry на joint-noise trained модель с patient-OOF и переобучением relation дала 162/206 exact, 9/368 false до gate, 49/0 после, 0 JUNCTION и 1 mixed. Высокий pre-gate счёт не перешёл в безопасное принятие JUNCTION. Исходный K10/15 остаётся контрольной моделью; `tau`, consistency и test не менялись. Протоколы и trace — в `varvara/anza_lira_local_results/ct28_copy_count_ablations/{selected_type_diagnostics,no_junction_geometry_selected,joint_junction_geometry_selected}/`.

## Scene shortcut: полные 84 против 76 без прямых counts

Удалены только восемь прямых candidate-count входов (`pair_candidate_count`, `junction_candidate_count`, `pair_count_repeat`, `junction_count_repeat` в двух score streams); local models и scores не менялись, relation HGB переобучена на тех же train rows и весах. Полный 84-feature head предварительно воспроизведён с погрешностью `1.11e-16`.

| Validation | Полные 84 | Без прямых counts, 76 |
|---|---:|---:|
| Raw relation accuracy | 88.86% | 88.86% |
| Raw macro-F1 | 0.8384 | 0.8445 |
| Accuracy после `tau` | 85.60% | 84.24% |
| Structural exact до gate, из 206 | 155 | 150 |
| False repair до gate, из 368 | 11 | 10 |
| Exact / false после gate | **52 / 0** | 49 / 0 |
| Принято `junction_only` / `mixed` | **2 / 2** | 0 / 1 |

**Сравнение:** опора — полный 84-feature K10/15; вариант 76 признаков выше только по raw macro-F1 и имеет меньше pre-gate false (10 против 11), но после gate уступает (49/0 против 52/0) и теряет оба принятых `junction_only`. **84 признака оставлены** в текущем варианте; shortcut-риск этим не снят.

У варианта без counts число `junction_only` с low-confidence отказом выросло с 12 до 18. Ранний count-only classifier на scene pack имел около 73.6% validation accuracy, поэтому риск подсказки от generator остаётся. Удаление прямых counts не убирает косвенный размер pool из распределений scores; полный head оставлен как **validation-претендент**, а не доказательство отсутствия shortcut. Evidence — `varvara/anza_lira_local_results/ct28_copy_count_ablations/no_cardinality_selected/`.

## Проверка восстановления отклонённых repair, 2026-11-02

Контроль — K10/15. Все 368 clean validation-сцен, их 15 joint-повторов, identity-совместимость и пороги прежние. Вероятности исходной relation head и все её структурные решения воспроизведены точно. Test закрыт.

Из 155 исходно точных repair gate отклоняет 103. Для каждого повтора причина определена последовательно: смена raw типа → падение confidence при том же типе → смена identity. Преобладающая причина на уровне сцены (при равенстве — тот же порядок):

| Отклонённые исходно точные сцены | Confidence | Raw type | Identity | Всего |
|---|---:|---:|---:|---:|
| `pair_only` | 38 | 6 | 17 | 61 |
| `junction_only` | 14 | 1 | 5 | 20 |
| `mixed` | 19 | 3 | 0 | 22 |

В их 1 545 повторах: 581 падение confidence, 277 смен типа, 265 смен identity, 422 совпадения исходного решения. Среди 297 low-confidence повторов JUNCTION/mixed сумма `P(JUNCTION)+P(BOTH)` достигает 0.85 только в 109. Разделение вероятности между классами объясняет часть отказов.

Проверена одна factorized relation head: два бинарных HGB оценивают наличие PAIR и JUNCTION, четыре вероятности собираются как произведения Bernoulli-вероятностей. Входы — те же 84 признака, train clean+15 joint с весами 0.5/0.5; local OOF scores не менялись. Это новая абляция с предположением независимости двух relations.

Отдельно проверен линейный pairwise ranker: PAIR — 45 geometry+CT признаков и 10 joint train-копий, JUNCTION — 9 geometry и 5 position-копий. LogisticRegression (`C=1`, без intercept) учится на разностях true/false кандидатов одной сцены; imputer/scaler обучены только на train. На inference sigmoid от utility используется только для identity. Scores ранжировщика не подаются в relation head, её вероятности заморожены.

| Вариант относительно K10/15 | Exact / false до gate | Exact / false после | Точные JUNCTION / mixed |
|---|---:|---:|---:|
| **K10/15, контроль** | **155 / 11** | **52 / 0** | **2 / 2** |
| Factorized relation HGB | 141 / 8 | 48 / 0 | 0 / 0 |
| Pairwise ranker только PAIR | 117 / 49 | 38 / 7 | 2 / 1 |
| Pairwise ranker только JUNCTION | 149 / 17 | 50 / 0 | 2 / 0 |
| Оба pairwise ranker | 109 / 57 | 37 / 7 | 2 / 0 |

На competitive candidates линейный ranker ухудшил top-1: PAIR clean `167/170 → 101/170`, joint `1793/2550 → 1482/2550`; JUNCTION clean `63/65 → 56/65`, joint `691/975 → 647/975`. Это отрицательный результат **линейного** ранжировщика, не проверка всех ranking-моделей. Ни одна абляция не заменяет K10/15; 80 точных принятых не достигнуто.

Проверены те же пять пациентов и точные страты `(scene_family, PAIR count, JUNCTION count)`. Factorized head улучшила число точных лишь у пациента 966 (`13→14`), ухудшив у остальных; PAIR ranker дал ложные repair у трёх пациентов. Machine evidence — `varvara/anza_lira_local_results/ct28_recovery_v1/`: `blocked_103_scenes.csv`, `blocked_by_patient_type.csv`, `metrics.json`, `per_patient_val.csv`, `count_strata_val.csv`; подкаталог `pairwise_identity/` содержит аналогичные ranking-результаты. Код: `evaluate_ct28_recovery.py` и `train_ct28_candidate_ranker.py` в `scripts/research/ccta_graph_lira_safe_repair/`.

## Усреднение relation confidence: фиксированный ансамбль

Опора — K10/15. Проверен **один** ансамбль из пяти relation HGB: seed `0…4`, `max_features=0.8`, остальные параметры прежние. Каждая модель обучена на тех же 17 train-пациентах и 84 признаках с clean+15 joint-копиями, веса 0.5/0.5; на inference усредняются четыре вероятности. Local scores, identity, candidate pool, `tau=0.85`, consistency `0.60` и 15 validation-повторов не менялись. Размер ансамбля и доля признаков по validation не подбирались; индивидуальные модели по её результатам не отбирались.

| Clean validation, 368 сцен | K10/15 | Ансамбль пяти HGB |
|---|---:|---:|
| Exact / false до gate | 155 / 11 | 153 / 11 |
| Exact / false после gate | **52 / 0** | 50 / 0 |
| Точные `junction_only` / `mixed` после gate | **2 / 2** | 2 / 1 |
| Raw relation accuracy | 88.86% | 88.86% |
| Relation log loss | 0.2799 | 0.2774 |

Ансамбль вернул 1 из прежних 103 отклонённых точных сцен, но потерял 3 из 52 принятых. На тех же 103 сценах средняя вероятность правильного типа в joint-повторах: JUNCTION `0.6992→0.6999`, mixed `0.5901→0.5916`, PAIR `0.7612→0.7604`. Доля правильных post-`tau` повторов не выросла: JUNCTION `40.33→40.00%`, mixed `22.12→22.12%`, PAIR `53.88→53.01%`. Небольшое улучшение log loss не превратилось в устойчивый repair.

**Решение:** ансамбль отклонён, K10/15 сохранён. Результат не поддерживает это сглаживание как исправление confidence; он не исключает другие способы обучения или калибровки. По пациентам exact после gate: `11/15/7/13/6 → 10/13/7/14/6` для `957/961/965/966/974`; ложных 0 в обоих вариантах. Сравнение в точных count-стратах сохранено.

Код — `scripts/research/ccta_graph_lira_safe_repair/train_ct28_confidence_ensemble.py`. Модель, SHA входов, вероятности и метрики — в игнорируемом Git `varvara/anza_lira_local_results/ct28_confidence_ensemble_v1/`: `metrics.json`, `predictions_val.csv`, `confidence_repeats_val.csv.gz`, `confidence_on_blocked103_val.csv`, `per_patient_val.csv`, `count_strata_val.csv`. Исходные вероятности и решения K10/15 воспроизведены точно; сохранённый ансамбль повторно проверен после загрузки. Held-out test не открывался.

## Явный штраф за изменение вероятностей: MLP CE против CE+JS

Один контролируемый эксперимент: две одинаковые relation MLP `84→64 tanh→4`, одинаковая инициализация seed 0 и train-нормализация. Первая получает class-balanced cross-entropy; вторая дополнительно минимизирует среднее расхождение Jensen–Shannon между вероятностями clean train-сцены и каждой из её 15 joint-копий (`λ=1`). CE-веса clean/копий — 0.5/0.5. Штраф применяется только при обучении; на inference чистая reference-сцена не требуется. Local scores, identity и gates прежние, test закрыт.

| Clean validation, 368 сцен | Exact / false до gate | Exact / false после | Точные JUNCTION / mixed |
|---|---:|---:|---:|
| **HGB K10/15 — основной контроль** | **155 / 11** | **52 / 0** | **2 / 2** |
| MLP CE — контроль архитектуры | 152 / 17 | 49 / 0 | 1 / 1 |
| **MLP CE+JS — отдельный JUNCTION-претендент** | 141 / 12 | **54 / 0** | **4 / 1** |

В этом запуске штраф улучшил ту же MLP: `49→54` точных после gate и `1→4` JUNCTION. Среднее JS на train снизилось `0.0889→0.0611`, на validation `junction_only` — `0.1418→0.1151`; у HGB оно ниже (`0.0873`). Поэтому общей устойчивости лучше HGB пока нет. В исходном виде верный JUNCTION проходит tau в `17/36` сценах против `23/36` HGB; после gate остаётся более устойчивое подмножество.

Относительно HGB новый вариант принял 7 новых точных сцен и потерял 5 прежних. Итог по пациентам `957/961/965/966/974`: `11/15/7/13/6 → 13/13/7/16/5`, ложных 0. **K10/15 не заменён:** CE+JS повышает общий exact и JUNCTION, но теряет один mixed и ухудшает pre-gate exact. Это один exploratory seed; path stage для новых 54 принятых сцен ещё не оценивался.

Оптимизация NumPy/SciPy проверена конечными разностями (максимальная ошибка градиента `3.84×10⁻¹¹`). Обе модели исчерпали одинаковый бюджет L-BFGS 1 000 итераций; сходимость до заданного допуска не достигнута. Бюджет увеличен с 200 по train-loss до чтения MLP validation-метрик; чекпойнты и λ по validation не выбирались. Это ограниченный эксперимент, не оптимум архитектуры.

Код — `scripts/research/ccta_graph_lira_safe_repair/train_ct28_probability_consistency.py` (`--check-only` проверяет градиент). Полные модели, параметры, SHA, вероятности, confidence, per-patient и count-страты — в `varvara/anza_lira_local_results/ct28_probability_consistency_v1/`, главный итог — `metrics.json`.

## PAIR: нелинейное сравнение кандидатов

Одна identity-абляция: бинарный HGB сравнивает два PAIR-кандидата одной сцены по разности и среднему их 45 geometry+CT признаков. Train — clean+10 joint-копий с суммарными весами 0.5/0.5, одинаковым весом negatives и обеих направлений сравнения; 44 484 preference-строки. Параметры HGB: 140 итераций, 15 листьев, lr 0.06, L2=2, seed 0. На inference score — средняя симметризованная вероятность победы над другими кандидатами; это utility для identity, не калиброванная вероятность реального PAIR.

Вероятности двух relation heads и JUNCTION identity заморожены. Candidate pool, tau и consistency прежние; test закрыт. Clean PAIR top-1 снизился `167/170→158/170`, joint-noise top-1 — `1793/2550→1736/2550`.

**Основной HGB K10/15:**

| PAIR identity | Exact / false до gate | Exact / false после | Точные `pair_only` | JUNCTION / mixed |
|---|---:|---:|---:|---:|
| Pointwise fusion, контроль | 155 / 11 | **52 / 0** | **48** | 2 / 2 |
| Preference HGB | 150 / 16 | 49 / 1 | 47 | 2 / 0 |

**Отдельный MLP CE+JS претендент:**

| PAIR identity | Exact / false до gate | Exact / false после | Точные `pair_only` | JUNCTION / mixed |
|---|---:|---:|---:|---:|
| Pointwise fusion, контроль | 141 / 12 | **54 / 0** | **49** | 4 / 1 |
| Preference HGB | 136 / 17 | 52 / 1 | 46 | 4 / 2 |

**Решение: отклонить preference HGB.** Обе головы приняли неверный `PAIR:E02+E04` в `CT28S:val:pair_only:965:15` с consistency 0.8: устойчивость выбранной связи сама по себе не гарантирует правильность. Вариант улучшал отдельные результаты пациента 957, но ухудшал 966. Прежние pointwise identity scores сохранены.

Код — `scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_preference.py`. Модель, raw feature cache, SHA, top-1, structural, per-patient и count-страты — в `varvara/anza_lira_local_results/ct28_pair_preference_v1/metrics.json` и соседних CSV. Пересчитанные raw признаки воспроизводят прежние noisy fusion scores с ошибкой `1.11×10⁻¹⁶`; оба reference head и неизменность их вероятностей проверены.

## Усиление JS только для PAIR train-сцен

Опора — сохранённая **MLP CE+JS 54/0**. Проверен один фиксированный вариант: JS-коэффициенты по истинным train-классам `NONE=1, PAIR=3, JUNCTION=1, BOTH=1`. CE-веса, train-данные, нормализация, архитектура, инициализация seed 0 и бюджет L-BFGS 1 000 итераций те же; модель обучена с нуля. На inference истинный тип не требуется: одна MLP оценивает все сцены. Выбор кандидатов, tau 0.85, consistency 0.60 и 15 проверочных повторов не менялись; test закрыт.

| JS при обучении | Exact / false до gate | Exact / false после | Точные PAIR-only / JUNCTION-only / mixed |
|---|---:|---:|---:|
| 1 для всех классов, контроль | **141 / 12** | 54 / 0 | 49 / **4** / 1 |
| 3 для PAIR, 1 для остальных | 135 / 12 | **55 / 0** | **50** / 3 / **2** |

На фиксированных 58 ранее отклонённых точных PAIR-сценах (870 повторов) правильный тип проходит tau в `56.21→60.46%`, средняя вероятность правильного типа `0.7748→0.8007`, средняя JS-разница с clean `0.0785→0.0483`. Усиление действительно стабилизирует PAIR-вероятности, но итоговый прирост невелик: приняты 6 новых точных сцен и потеряны 5 прежних. По пациентам `957/961/965/966/974`: `13/13/7/16/5 → 12/14/6/18/5`; ложных 0. Pre-gate exact снижается; среди принятых на один `junction_only` меньше и на один `mixed` больше.

**Статус:** отдельный PAIR-претендент 55/0; контроль 54/0 с четырьмя JUNCTION сохранён, основной HGB K10/15 не заменён. Разница в одну сцену на пяти validation-пациентах не устанавливает превосходство. Оптимизатор снова достиг лимита 1 000 итераций; полной сходимости не заявляем. Проверены численный градиент, неизменность loss остальных классов и точное воспроизведение контроля 54/0.

Код — `scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_js.py`; общий loss получил необязательные class-множители с проверкой, что единичные множители точно сохраняют прежние loss/gradient. Модель, SHA, confidence, per-patient и count-страты — `varvara/anza_lira_local_results/ct28_probability_consistency_pair3_v1/`, главный итог — `metrics.json`. Reference blocked cohort здесь — **87 сцен модели 54/0**, не прежние 103 сцены HGB.
