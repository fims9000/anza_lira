# CT28 Graph-LIRA: история версий и выбранный validation-вариант

Все сравнения ниже — development на тех же 17 train и 5 validation пациентах. Held-out test не открывался. Для structural оценки: `Exact до/206` — точные решения среди 206 сцен, требующих repair; `False до/368` — ложные связи среди всех 368; после gate — принятые точные и ложные сцены. `J` и `mixed` — точные принятые `junction_only` и `mixed`.

## Что осталось текущим

**Один текущий вариант: K10/15** — `varvara/anza_lira_local_results/ct28_copy_count_ablations/joint_check/fusion_10_relation_15/`. PAIR/JUNCTION geometry HGB обучены на clean + 5 position-only копиях; CT-only HGB на clean; PAIR geometry+CT identity HGB на clean + 10 joint-копиях; JUNCTION identity выбирается по geometry score; 84-feature CT-conditioned relation HGB обучена на clean + 15 joint-копиях. Все train scores relation head — patient-level OOF. `tau=0.85`, consistency `0.60` и 15 оценочных perturbations не подбирались в этой серии. Полная версия и SHA указаны в корневом `README.md`.

| Этап | Exact до/206 | False до/368 | После gate: exact / false | Точные J / mixed | Статус и evidence |
|---|---:|---:|---:|---:|---|
| Исходный clean Logistic CT-conditioned | 60 | 103 | 2 / 0 | 0 / 0 | Ранний CT28 baseline: корневые `relation_*` и `graph_lira_*`; не исторические веса Graph-LIRA. |
| Clean HGB, geometry+CT identity для обоих | 176 | 13 | 0 / 0 | 0 / 0 | Сильный pre-gate, нулевой coverage: `local_hgb/graph_lira_ct_identity/`. |
| Position-robust geometry PAIR+JUNCTION, unrounded; PAIR fusion identity, JUNCTION geometry identity | 166 | 9 | 4 / 0 | 0 / 0 | Рабочий контроль до relation augmentation: `pair_junction_geometry_robust_unrounded/`, `working_baseline/`. |
| Relation clean + 15 joint-копий, PAIR fusion K=5 | 153 | 13 | 48 / 0 | 2 / 1 | Промежуточная база copy sweeps: `varvara/anza_lira_local_results/ct28_copy_count_ablations/joint_check/fusion_05_relation_15/`. |
| **PAIR fusion K=10, relation K=15, geometry K=5/5** | **155** | **11** | **52 / 0** | **2 / 2** | **Текущий K10/15**, полный `clean_eval/` и `relation_model_protocol.json` по указанному выше пути. |

**Сравнение:** первая строка — ранний clean Logistic CT28, промежуточный рабочий контроль — строка 166/9 → 4/0. Максимум pre-gate exact у clean HGB (176/206), но gate принял 0; лучший выбранный баланс после gate — K10/15 (52/0, J=2, mixed=2). Только **K10/15** стал текущим validation-претендентом.

Ранние этапы показывают ход разработки, но не являются равноправными финальными моделями: менялись local family, identity score или train augmentation. Stress30-строки обучались/проверялись на отдельно смещённой базовой сцене и не смешиваются с clean-таблицей. Вся история не повышает status K10/15 до frozen test-конфигурации.

## Проверенные альтернативы рядом с выбранной версией

| Изменение относительно исходной точки sweep (geometry K=5/5, fusion K=5, relation K=15) | После gate exact / false | J / mixed | Где полная таблица |
|---|---:|---:|---|
| Relation K=0,3,5,10,15 при fusion K=5 | 33/0; 49/1; 50/1; 48/0; 48/0 | От 0/0 до 2/1 | `varvara/anza_lira_local_results/ct28_copy_count_ablations/relation/comparison_clean_val.csv` |
| PAIR fusion K=0,3,5,10,15 при relation K=15 | 14/0; 49/0; 48/0; **52/0**; 48/0 | K=10: 2/2 | `varvara/anza_lira_local_results/ct28_copy_count_ablations/fusion/comparison_clean_val.csv` |
| PAIR и JUNCTION geometry K менялись вместе: 0,3,5,10,15, при fusion K=5 | 46/0; 48/0; 48/0; 49/1; 48/0 | K=5: 2/1 | `varvara/anza_lira_local_results/ct28_copy_count_ablations/geometry/comparison_clean_val.csv` |
| PAIR fusion K=10 + relation K=5 | **54/0** | 1/1 | `varvara/anza_lira_local_results/ct28_copy_count_ablations/joint_check/fusion_10_relation_05/` |
| Раздельные geometry K: лучший общий счёт PAIR=3/JUNCTION=5, при fusion K=10/relation K=15 | 53/0 | 1/1 | `varvara/anza_lira_local_results/ct28_independent_geometry_copy_sweep/comparison_clean_val.csv` |

**Сравнение:** опора для первых трёх строк — geometry K=5/5, fusion K=5, relation K=15 (48/0, J=2, mixed=1); для последних двух — выбранный K10/15 (52/0, J=2, mixed=2). Максимум общего числа точных здесь 54/0 при relation K=5, но он теряет по одному JUNCTION и mixed. Ни этот вариант, ни 53/0 **не заменили** K10/15; fusion K=10 из sweep вошёл в него.

K10/15 оставлен текущим из-за двух принятых `junction_only` и двух `mixed` при нуле ложных; варианты 54/0 и 53/0 теряют по одному JUNCTION и mixed. Сравнение сделано всего на пяти validation пациентах; разница в 1–2 сцены не доказывает превосходство на новых пациентах. Другие сочетания и состав копий находятся в указанных comparison CSV; их выводы и причины отклонения собраны в `rejected/README.md`.

## Полные sweep-таблицы на пути к K10/15

В трёх первых таблицах меняется **один уровень** при фиксированных остальных: исходная точка — geometry K=5/5, PAIR fusion K=5, relation K=15. K означает число возмущённых **train-копий** на исходный candidate или сцену, а не 15 повторов при validation consistency. Для K>0 clean и копии имеют суммарные веса 0.5/0.5; K=0 означает только clean. `J` и `B` — точные принятые `junction_only` и `mixed`.

| Relation K, joint | Exact до/206 | False до/368 | Exact после | False после | Принято | J | B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 164 | 11 | 33 | 0 | 33 | 0 | 0 |
| 3 | 156 | 12 | 49 | 1 | 50 | 0 | 0 |
| 5 | 156 | 12 | 50 | 1 | 51 | 1 | 0 |
| 10 | 153 | 12 | 48 | 0 | 48 | 1 | 0 |
| 15 | 153 | 13 | 48 | 0 | 48 | 2 | 1 |

**Сравнение:** опора — K=0 (33/0 после gate); по одному числу exact максимум K=5 (50/1), но с ложным repair. При условии 0 ложных и сохранения JUNCTION/mixed выбран K=15 (48/0, J=2, B=1). **K=15 вошёл** в текущий K10/15; это выбор внутри sweep при fusion K=5, а не конечные 52/0.

| PAIR fusion K, joint | Exact до/206 | False до/368 | Exact после | False после | Принято | J | B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 155 | 11 | 14 | 0 | 14 | 2 | 0 |
| 3 | 156 | 10 | 49 | 0 | 49 | 2 | 1 |
| 5 | 153 | 13 | 48 | 0 | 48 | 2 | 1 |
| 10 | 155 | 11 | **52** | 0 | 52 | **2** | **2** |
| 15 | 153 | 13 | 48 | 0 | 48 | 2 | 1 |

**Сравнение:** опора — fusion K=5 (48/0, J=2, B=1); лучший K=10 (52/0, J=2, B=2). **PAIR fusion K=10 заменил K=5** в текущем K10/15.

| Geometry K, position-only для PAIR и JUNCTION | Exact до/206 | False до/368 | Exact после | False после | Принято | J | B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 153 | 10 | 46 | 0 | 46 | 0 | 0 |
| 3 | 150 | 12 | 48 | 0 | 48 | 0 | 1 |
| 5 | 153 | 13 | 48 | 0 | 48 | 2 | 1 |
| 10 | 151 | 13 | 49 | 1 | 50 | 0 | 3 |
| 15 | 148 | 10 | 48 | 0 | 48 | 0 | 1 |

**Сравнение:** опора — geometry K=5/5 (48/0, J=2, B=1); наибольшее число exact у K=10 (49), но с одним ложным repair и без JUNCTION. По критерию 0 ложных с сохранением JUNCTION **K=5/5 оставлен** в текущем K10/15. Эта таблица оценивалась при fusion K=5.

Relation K=30 не запускался: к K=15 число точных после gate перестало расти. Состав 15 relation-копий `5 position + 5 tangent + 5 joint` при исходном PAIR fusion K=5 дал 153/206 exact и 12/368 false до gate, **51/0** после gate, J=2 и B=0; он не заменил joint-only K=15 из-за потери mixed. Машинные источники: `varvara/anza_lira_local_results/ct28_copy_count_ablations/{relation,fusion,geometry,noise_mix}/comparison_clean_val.csv`.

После этих однопараметрических проверок выполнен маленький 2×2, а затем раздельный sweep geometry K. Здесь PAIR fusion K=10 и relation K=15 дают единственный выбранный вариант:

| PAIR fusion K / relation K | Exact до/206 | False до/368 | Exact после | False после | J | B |
|---|---:|---:|---:|---:|---:|---:|
| 5 / 5 | 156 | 12 | 50 | 1 | 1 | 0 |
| 5 / 15 | 153 | 13 | 48 | 0 | 2 | 1 |
| 10 / 5 | 158 | 10 | **54** | 0 | 1 | 1 |
| **10 / 15** | **155** | **11** | **52** | **0** | **2** | **2** |

**Сравнение:** опора — fusion K=5 / relation K=15 (48/0, J=2, B=1); максимум exact — K=10/5 (54/0), но J/B=1/1. **Выбран K=10/15** (52/0, J/B=2/2), он заменил опору.

| PAIR geometry K / JUNCTION geometry K | Exact до/206 | False до/368 | Exact после | False после | J | B |
|---|---:|---:|---:|---:|---:|---:|
| 0 / 5 | 157 | 14 | 42 | 0 | 0 | 1 |
| 3 / 5 | 151 | 11 | 53 | 0 | 1 | 1 |
| 5 / 0 | 153 | 8 | 44 | 0 | 0 | 0 |
| 5 / 3 | 156 | 10 | 49 | 0 | 1 | 1 |
| **5 / 5** | **155** | **11** | **52** | **0** | **2** | **2** |
| 5 / 10 | 154 | 12 | 51 | 0 | 0 | 1 |
| 10 / 5 | 154 | 12 | 49 | 0 | 1 | 0 |
| 10 / 10 | 153 | 11 | 51 | 0 | 0 | 4 |

**Сравнение:** опора — geometry K=5/5 (52/0, J=2, B=2). Числовой максимум exact — K=3/5 (53/0), но J/B=1/1. **K=5/5 оставлен** в текущем варианте; раздельный sweep не изменил его.

Источники: `varvara/anza_lira_local_results/ct28_copy_count_ablations/joint_check/comparison_clean_val.csv` и `varvara/anza_lira_local_results/ct28_independent_geometry_copy_sweep/comparison_clean_val.csv`. Все строки — exploratory validation; подбирать по ним test нельзя.

## Граница результатов

Поздний эксперимент с явным штрафом за изменение вероятностей выделил MLP CE+JS как отдельного JUNCTION-претендента. Контроль архитектуры, таблица результатов и ограничения — в [consistency diagnostics](../consistency_diagnostics/README.md#явный-штраф-за-изменение-вероятностей-mlp-ce-против-cejs). Основной K10/15 сохранён из-за потери mixed и снижения pre-gate exact у нового варианта.

Усиление JS только на PAIR train-сценах сохранено как отдельное сравнение с MLP 54/0: [протокол, результат и обмен PAIR/JUNCTION/mixed](../consistency_diagnostics/README.md#усиление-js-только-для-pair-train-сцен). Нового выбора основной модели по этой одноточечной абляции нет.

На выбранном K10/15 path stage запускался только **после** gate: маршруты построены для всех 52 принятых сцен. Это не добавляет принятых сцен и не подтверждает анатомическую точность каждого маршрута; детали — в `../path/README.md`. До test остаются качество JUNCTION/BOTH при perturbations, отсутствие принятого нового path protocol как исторической репликации и ограничение validation пятью пациентами.
