# CT28 Graph-LIRA: невыбранные абляции

Здесь только train/validation; held-out test не открывался. Текущий вариант — **K10/15: 52 точных / 0 ложных после gate**, включая 2 `junction_only` и 2 `mixed` ([основной журнал](../README.md)). Таблицы ниже относятся к разным опорным моделям; clean и Stress30 разделены. `accepted` включает также ложные и неполные repair.

Подробные outputs (309 файлов) находятся в локальном `varvara/anza_lira_local_results/ct28_rejected_archive/rejected_validation_outputs.tar.gz` (SHA256 `3951fab68c9e4698b7198a64569f261a934fb565c3525af955c09a2b7089078c`). Пути и SHA каждого файла — в `archive_manifest.json`. 27 скриптов невыбранных экспериментов сохранены в `rejected_experiment_scripts.tar.gz`; ранние отслеживаемые baseline лежат в `../local_hgb/` и `../stress30_jitter1/`.

```bash
tar -xzf varvara/anza_lira_local_results/ct28_rejected_archive/rejected_validation_outputs.tar.gz -C .
tar -xzf artifacts/varvara/ct28_graph_lira/ablations/rejected/rejected_experiment_scripts.tar.gz -C .
```

В архив перенесены также factorized recovery, pairwise/nonlinear rankers, confidence ensemble, MLP CE/JS, PAIR-JS и оба train-only path-evidence пилота. Их код не нужен для запуска K10/15. Указанные в отчётах исходные пути скриптов доступны после восстановления второй командой выше; вместе восстанавливаются их вспомогательные модули.

Отдельные отрицательные проверки, не повторённые в таблицах ниже:

Поздняя проверка поверх K10/15 — factorized relation HGB и линейные pairwise identity rankers — описана в [consistency diagnostics](../../consistency_diagnostics/README.md#проверка-восстановления-отклонённых-repair-2026-11-02). Все варианты отклонены; K10/15 сохранён.

Фиксированный ансамбль пяти relation HGB также отклонён; протокол и сравнение confidence — [в том же отчёте](../../consistency_diagnostics/README.md#усреднение-relation-confidence-фиксированный-ансамбль).

Нелинейный PAIR preference HGB проверен с обеими relation heads и отклонён: [протокол и отдельные таблицы](../../consistency_diagnostics/README.md#pair-нелинейное-сравнение-кандидатов).

Train-only пилот CT path evidence также не перенесён в модели: [matched true/false, устойчивость и endpoint-диагноз](../../path/README.md#train-only-пилот-path-evidence-для-кандидатов). Отдельная CT-привязка endpoints со штрафом тоже не перенесена: noisy JUNCTION AUROC 0.550→0.605, но paired-win 45/85→48/85 и bootstrap-интервал прироста включает ноль; [полное сравнение и ограничения](../../path/README.md#ct-привязка-endpoints-train-only-абляция).

| Проверка | Итог | Источник |
|---|---|---|
| Усреднение JUNCTION score с пятью inference-time jitter-копиями | Clean top-1 `34/36 → 32/36`, joint `367/540 → 363/540`; отклонено. | `varvara/anza_lira_local_results/ct28_junction_score_averaging_v1/` |
| Замена только JUNCTION local HGB на LogisticRegression | Clean top-1 `34/36 → 29/36`, position-noisy `375/540 → 341/540`; полный Graph-LIRA не запускался. | `varvara/anza_lira_local_results/ct28_junction_local_logistic_diagnostic/` |
| Удаление пяти local position-копий | Не улучшило выбранную линию; Stress30 в том каталоге имеет иные условия обучения. | `varvara/anza_lira_local_results/ct28_no_local_augmentation_mixed_identity/` |
| Удаление `junction_span_mm` с переобучением | Noisy top-1 `0.765 → 0.732`, clean accepted 0. | `varvara/anza_lira_local_results/ct28_selected_relation_augmentation/drop_junction_span/` |
| Удаление JUNCTION geometry stream | K10/15 `52/0 → 22/0`, принятых JUNCTION `2 → 0`. | `varvara/anza_lira_local_results/ct28_copy_count_ablations/no_junction_geometry_selected/` |
| Joint-копии только для JUNCTION geometry | K10/15 `52/0 → 49/0`, принятых JUNCTION `2 → 0`. | `varvara/anza_lira_local_results/ct28_copy_count_ablations/joint_junction_geometry_selected/` |
| Position-robust JUNCTION fusion вместо geometry identity | Полный Graph-LIRA `52/0 → 51/0`; mixed `2 → 1`. | `varvara/anza_lira_local_results/ct28_junction_fusion_selected_v1/` |
| Удаление прямых counts в раннем Stress30 HGB | Accepted `53 → 49`, принятых JUNCTION 0. | `ablations/stress30_jitter1/local_hgb/no_cardinality/` |

## Числовые переходы прежних вариантов

Пути — от корня репозитория. Clean и Stress30 — разные train/validation режимы. В Stress30 базовая сцена имеет `1 мм + 30°`, а consistency добавляет 15 повторов `0.8 мм + 20°`. `Exact до` — из 206 repair-needed, `false до` — из 368 сцен; после gate указаны числа сцен.

### Ранние модели: clean train и clean validation

Внутри каждого local family опора — выбор identity по geometry; вторая строка меняет только identity на geometry+CT.

| Local family / identity | Exact до/206 | False до/368 | После: exact / false / incomplete | Accepted |
|---|---:|---:|---:|---:|
| Logistic / geometry для обоих | 60 | 103 | 2 / 0 / 0 | 2 |
| Logistic / geometry+CT для обоих | 105 | 57 | 14 / 0 / 0 | 14 |
| HGB / geometry для обоих | 163 | 26 | 0 / 0 / 0 | 0 |
| HGB / geometry+CT для обоих | 176 | 13 | 0 / 0 / 0 | 0 |

**Clean:** Logistic выиграла от fusion identity (2→14 точных после gate); HGB осталась с 0. Ни одна ранняя модель не заменяет K10/15.

### Ранние модели: Stress30 train и Stress30 validation

Базовые сцены здесь смещены на `1 мм + 30°`; это отдельный режим обучения и проверки, а не дополнительные повторы clean validation. Опора внутри каждого local family — geometry identity.

| Local family / identity | Exact до/206 | False до/368 | После: exact / false / incomplete | Accepted |
|---|---:|---:|---:|---:|
| Logistic / geometry для обоих | 48 | 72 | 15 / 12 / 0 | 27 |
| Logistic / geometry+CT для обоих | 65 | 54 | 35 / 9 / 1 | 45 |
| HGB / geometry для обоих | 51 | 66 | 12 / 6 / 0 | 18 |
| HGB / geometry+CT для обоих | 79 | 36 | 47 / 4 / 2 | 53 |

**Stress30:** fusion identity улучшила exact, но HGB приняла 4 ложных и 2 неполных из 53. Это отдельный режим, не clean-контроль.

При той же relation head переход `geometry → geometry+CT` менял identity в 84/21 clean Logistic/HGB и 51/52 Stress30 Logistic/HGB сценах. PAIR top-1: `50→100`, `153→167`, `62→94`, `76→116` из 170; JUNCTION: `62→64`, `68→68`, `47→49`, `43→42` из 72. Источник — архивный `ablations/ct_identity_comparison.json`.

### Позднейший clean контроль перед relation augmentation

Поздний clean контроль: PAIR fusion, JUNCTION geometry; обе geometry HGB обучены с 5 position-only копиями, CT-only/fusion — без копий. Новые OOF scores и relation HGB дали **166/206 exact, 9/368 false до gate; 4/0/0 после**. Это новая обученная линия. Далее проверяли relation clean+15 joint с равными весами `1/16`, затем `0.5/0.5`; K10/15 включает другую PAIR fusion модель.

| Раннее relation обучение с равными весами 1/16 | Exact до/206 | False до/368 | После: exact / false / incomplete | Accepted |
|---|---:|---:|---:|---:|
| Clean validation | 87 | 9 | 8 / 0 / 0 | 8 |
| Stress30 validation той же модели | 37 | 46 | 8 / 18 / 0 | 26 |

**Опора:** та же модель на clean validation. Stress30 — проверка устойчивости; равные веса `1/16` не вошли в K10/15.

В этой **ранней** равновесной ветке оба типа identity выбирались по fusion; это отдельный контроль, а не единственное изменение относительно выбранного mixed-identity контроля. Источники: `artifacts/varvara/ct28_graph_lira/ablations/relation_perturbation_augmented_hgb/` и Stress30 pack.

Когда PAIR/JUNCTION geometry уже имели пять position-only train-копий, а PAIR/JUNCTION identity были fusion/geometry, проверили только способ обучения relation HGB. Каждая строка ниже использует одну обученную модель для clean и Stress30 validation; Stress30 не является отдельным переобучением этой строки. Joint означает одновременный position и tangent noise в **15 train-копиях сцены**.

| Relation train-копии | Clean exact / false после | Stress30 exact / false после | Принято clean JUNCTION |
|---|---:|---:|---:|
| Нет, только clean | 4 / 0 | 2 / 1 | 0 |
| 15 position-only, веса 0.5/0.5 | 12 / 0 | 4 / 12 | 2 |
| 15 tangent-only, веса 0.5/0.5 | 10 / 0 | 3 / 3 | — |
| 15 joint, веса 0.5/0.5 | 14 / 0 | 11 / 20 | 2 |
| 15 joint, равные веса 1/16 | 13 / 0 | 10 / 19 | — |

**Опора:** relation без копий (4/0 clean, 2/1 Stress30). Joint 0.5/0.5 дал 14/0 clean, но 11/20 Stress30. В K10/15 перенесены веса, не эта модель.

Результат 14/0 был промежуточным претендентом, но на Stress30 у него 20 ложных принятых repair: 19 `pair_only` с правильным типом PAIR и неверным PAIR candidate, ещё одна NONE-сцена с ошибочно принятым PAIR. Это мотивировало следующий переход — **менять local PAIR fusion для candidate identity**, сохранив relation head и JUNCTION identity. Источник: `varvara/anza_lira_local_results/ct28_selected_relation_augmentation/`.

| Пять train-копий fusion identity, scope и вид шума | Clean exact / false после | Stress30 exact / false после |
|---|---:|---:|
| Без переобучения fusion, промежуточный претендент | 14 / 0 | 11 / 20 |
| Position-only, только PAIR | 47 / 0 | 26 / 4 |
| Position-only, только JUNCTION | 14 / 0 | 11 / 20 |
| Position-only, оба | 46 / 0 | 26 / 4 |
| Tangent-only, только PAIR | 18 / 0 | 11 / 20 |
| Tangent-only, только JUNCTION | 12 / 0 | 10 / 20 |
| Tangent-only, оба | 16 / 0 | 10 / 20 |
| Joint, только PAIR | **48 / 0** | **28 / 5** |
| Joint, только JUNCTION | 14 / 0 | 11 / 20 |
| Joint, оба | **48 / 0** | **28 / 5** |

**Опора:** fusion без train-копий (14/0 clean). Joint PAIR дал 48/0; только его подход вошёл в K10/15, уже с K=10.

В JUNCTION fusion-абляции PAIR fusion и relation head не менялись; выбранный K10/15 по-прежнему ранжирует JUNCTION по geometry. Источник: `varvara/anza_lira_local_results/ct28_fusion_identity_augmentation/`.

| Пять копий PAIR fusion | Validation | Exact до/206 | False до/368 | Exact после | False после |
|---|---|---:|---:|---:|---:|
| Position-only | Clean | 154 | 12 | 47 | 0 |
| Position-only | Stress30 | 59 | 31 | 26 | 4 |
| Joint | Clean | 153 | 13 | 48 | 0 |
| Joint | Stress30 | 56 | 34 | 28 | 5 |
| Joint, fusion обоих типов | Clean | 151 | 15 | 48 | 0 |
| Joint, fusion обоих типов | Stress30 | 56 | 34 | 28 | 5 |

**Опора:** fusion без копий. В K10/15 перенесён joint PAIR с K=10; показанные K=5 строки — ранняя проверка.

Следом отдельно проверили, можно ли улучшить входы relation head, меняя **geometry local models**. Во всех строках следующей матрицы relation head обучена с clean+15 joint train-сценами; local geometry использует пять копий указанного вида только в указанных типах. Это не та же абляция, что fusion identity выше.

| Шум в PAIR/JUNCTION geometry local | Exact до/206 | False до/368 | Exact / false после |
|---|---:|---:|---:|
| Без копий | 153 | 10 | 13 / 0 |
| Position-only PAIR | 154 | 10 | 8 / 0 |
| Position-only JUNCTION | 157 | 12 | 12 / 0 |
| Position-only оба | 154 | 13 | 12 / 0 |
| Tangent-only PAIR | 142 | 9 | 13 / 0 |
| Tangent-only JUNCTION | 152 | 10 | 18 / 0 |
| Tangent-only оба | 146 | 8 | 14 / 0 |
| Joint PAIR | 169 | 10 | 13 / 0 |
| Joint JUNCTION | 159 | 11 | 12 / 0 |
| Joint оба | 165 | 8 | 18 / 0 |

**Опора:** geometry без копий (13/0). Максимум здесь 18/0, но весь опыт предшествует PAIR fusion K=5/10 и не заменяет K10/15.

Эта матрица **предшествовала** PAIR fusion K=5/10 и потому её максимум 18/0 не сравнивается с поздним 48/0 или K10/15 как замена одного компонента. Источник: `varvara/anza_lira_local_results/ct28_geometry_augmentation_matrix/comparison_clean_val.csv`.

С уже joint-augmented PAIR fusion K=5 при сравнении PAIR/JUNCTION geometry снова менялись patient-level OOF и relation head. Тут лучший суммарный счёт `51/1` получился при joint geometry обоих, но `junction_only` исчезли; исходные position-only geometry имели `48/0` и 2 точных JUNCTION. Поэтому `51/1` не стал основой выбранного варианта:

| Joint geometry вместо position-only | Exact до/206 | False до/368 | Exact / false после | Accepted | Точные JUNCTION |
|---|---:|---:|---:|---:|---:|
| Без замены | 153 | 13 | 48 / 0 | 48 | 2 |
| Только PAIR | 162 | 8 | 47 / 1 | 48 | 2 |
| Только JUNCTION | 160 | 10 | 47 / 0 | 47 | 0 |
| Оба | 160 | 8 | 51 / 1 | 52 | 0 |

**Опора:** position-only geometry (48/0, J=2). Joint обоих дал 51/1, J=0; замену отклонили.

На строке `Оба` отдельно проверили PAIR/JUNCTION fusion selector и relation-noise виды. Это **другая опорная точка**, а не текущий K10/15:

| Fusion identity на joint geometry обоих | Exact до/206 | False до/368 | Exact / false после |
|---|---:|---:|---:|
| Joint fusion только PAIR | 160 | 8 | 51 / 1 |
| Joint fusion оба | 158 | 10 | 51 / 1 |
| Position-only fusion PAIR | 161 | 7 | 48 / 0 |

**Опора:** joint geometry обоих + joint fusion PAIR (51/1). Варианты не улучшили её и не вошли в K10/15.

| Relation train-noise при joint geometry обоих | Exact / false после | Точные JUNCTION |
|---|---:|---:|
| Нет | 12 / 0 | 0 |
| Только position | 45 / 1 | 1 |
| Только tangent | 39 / 0 | 0 |
| Joint | 51 / 1 | 0 |

**Опора:** joint geometry обоих без relation noise (12/0). Joint дал 51/1, но J=0; вся ветка отклонена.

Источники: `varvara/anza_lira_local_results/ct28_geometry_pair_fusion_comparison/` и `varvara/anza_lira_local_results/ct28_joint_geometry_ablations/`. Последующие K-sweeps, где PAIR fusion K=10 дал 52/0, находятся в `../README.md`.

## Отвергнутые локальные JUNCTION исправления и scene shortcut

Эти ранние проверки относятся к **разным веткам** HGB до K10/15:

| Проверка | Проверенный результат | Статус |
|---|---|---|
| Девять geometry-входов JUNCTION fusion | Контрфактическая фиксация clean `junction_span_mm`: top-1 **453/975 → 798/975** при position noise. | Только диагностика: clean span при inference неизвестен. |
| Position-robust JUNCTION geometry для identity | Exact до gate **83.01% → 84.47%**, false **2.99% → 2.17%**; после gate **0** принято. | Компонент использовался в поздней линии, но этот ранний прогон не равен K10/15. |
| Position-robust JUNCTION geometry+CT fusion | Clean competitive top-1 **59/65 → 61/65**, noisy **453/975 → 721/975**; exact до gate **83.01% → 83.98%**, после gate **0** принято. | Отдельная невыбранная fusion-ветка. |
| Переключение типа при joint noise | Исходно верный JUNCTION сохранялся в **108/420** повторов, BOTH — в **48/450**. | Ранние сцены; не путать с 345/360 повторами выбранного K10/15. |

**Основа:** ранние независимые диагностики; clean-span подстановка недоступна при inference. Из этой таблицы не выбирали новую модель.

Fusion predictions/metrics — в архивном `ablations/junction_fusion_position_robustness/`. Position-robust geometry и input-attribution outputs — в других одноимённых членах архива. Удаление округления и повторное обучение произошли позднее.

| PAIR geometry, ранний position-only опыт | Прежний вариант | Position-robust |
|---|---:|---:|
| Top-1 на 170 clean сценах | 157/170 | 156/170 |
| Top-1 в 2 550 position-only повторах | 896/2550 | **1217/2550** |
| Exact до gate | 84.47% | 79.13% |
| Exact / false после gate | 0 / 0 | 3 / 0 |

**Опора:** ранняя PAIR geometry без копий. Position-only улучшила noisy top-1, но снизила clean exact до gate; ранняя модель целиком не вошла в K10/15.

Top-1 подтверждён архивным `pair_junction_geometry_robust/pair_position_ranking_val.json`; structural сравнение — `pair_junction_geometry_robust/graph_lira/`. Улучшение ranking не означало улучшения exact до gate. Позже численное представление local features было унифицировано без округления, поэтому ранние проценты не служат прямым контролем K10/15.

На K10/15 удаление восьми прямых counts дало `150/206 exact`, `10/368 false` до gate и `49/0` после, без принятых JUNCTION; полные 84 признака — `155/11`, `52/0`, J=2, mixed=2. Count-only правило угадывало ~73.6% validation классов. Counts сохранены, shortcut-риск остаётся; подробности — в `../../consistency_diagnostics/README.md`.

### Ранний разбор JUNCTION geometry и Stress30

В 72 validation-сценах с true JUNCTION (36 `junction_only`, 36 `mixed`) 65 содержат false-конкурента. В этой **ранней fusion-selector** диагностике 15 position-only повторов каждой сцены дали:

| JUNCTION selector | Top-1, все 1080 повторы | Top-1, 975 competitive повторов |
|---|---:|---:|
| Position-robust geometry | 825 | 720 |
| CT | 547 | 442 |
| Старый geometry+CT fusion | 558 | 453 |

**Опора:** старый fusion selector. Position-robust geometry лучше по noisy top-1; в K10/15 сохранён geometry selector, но это другие обученные веса.

Ранняя fusion-диагностика на 975 competitive position-only повторах: all-perturbed `453` правильных top-1, clean CT `485`, clean geometry `899`; на всех 1080 повторах — `558`, `590`, `1004`. Подстановка clean `junction_span_mm` дала `798`, `junction_forward_mean_mm` — `611`, `junction_center_distance_std_mm` — `514`, остальные шесть — `453–484`; все 152 clean CT inputs — `485`. Медиана span `3.76→4.48 мм`, максимум `7.54 мм`. Clean-подстановка недоступна при inference. Источник: архивный `ablations/junction_identity_diagnostics/`.

| 36 ранних clean-HGB `junction_only` | Принято | Exact |
|---|---:|---:|
| Joint noise, динамические scores | 0 | 0 |
| Position-only, динамические | 0 | 0 |
| Tangent-only, динамические | 15 | 15 |
| Joint, CT score stream фиксирован | 0 | 0 |
| Joint, geometry score streams фиксированы | 10 | 10 |
| Joint, только identity scores фиксированы | 1 | 0 |

**Опора:** ранний joint noise с динамическими scores. Tangent-only проверяет другой шум; фиксация clean streams недоступна при inference.

| True JUNCTION geometry feature | Clean median | Position-only median | Центральные 90% train positives |
|---|---:|---:|---:|
| Span, мм | 3.762 | 4.463 | 3.031–3.940 |
| Residual mean, мм | 0.226 | 0.681 | 0.075–0.447 |
| Residual max, мм | 0.309 | 0.919 | 0.102–0.514 |
| Alignment min | 0.976 | 0.779 | 0.928–0.999 |

**Основа:** clean positives, их position-only копии и train-диапазон; таблица показывает сдвиг, а не новую модель.

Position-only выводил за центральные 90% train positives `79.26%` residual-mean и `80.56%` alignment-min. Медианный сдвиг центра: `1.00/0.59/1.21 мм` при position/tangent/joint. Фиксация CT не вернула принятие. Нулевой пересчёт выявил максимальный HGB score shift `0.01820` clean и `0.26034` Stress30 на bin boundary; tau и candidate keys в 100 сценах сохранились. После этого fit/inference используют одинаковую точность. Это ранние 100 сцен (`36 junction_only`, `36 mixed`, `28 none_incomplete_junction`), не все 368; источник — архивный `ablations/junction_input_diagnostics/`.

| Отдельная Stress30 HGB robustness абляция | JUNCTION после tau | BOTH после tau | Accepted | Exact | False | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Stress30 HGB reference | 5/36 | 6/36 | 53 | 47 | 4 | 14.40% |
| Robust local + base relation | 7/36 | 7/36 | 54 | 47 | 7 | 14.67% |
| Robust local + weighted relation | 5/36 | 3/36 | 25 | 24 | 1 | 6.79% |

**Опора:** ранний Stress30 HGB (47 exact, 4 false). Robust local повышала false; weighted relation уменьшала и false, и exact. В K10/15 не вошли.

Во второй строке local train: clean вес `0.5` + 5 копий по `0.1`, relation — base scores; в третьей relation тоже получает base `0.5` + 5 копий суммарно `0.5`. `junction_only` не принято. Type consistency для исходного JUNCTION: `30.67/29.70/30.37%`, identity при том же типе `36.16/32.32/38.40%`, exact agreement `6.67/6.06/6.67%`. Источник: `ablations/stress30_jitter1/local_hgb_robustness/`.
