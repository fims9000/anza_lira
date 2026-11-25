# CT28 ImageCAS · CNN и ANZA · seed 20261102

**Итог на пяти масках.** Только CNN с весом ранжирования 0,5 восстановила 1 из 15 разрывов без ложных связей.

Ветка `experiment/ccta3d-topology-anisotropy-2026-11-11` (`81fc7ae`). 17 train и 5 development-validation пациентов, 6 held-out закрыты.

## Данные и обучение

- **Данные.** Доступны ImageCAS, маски и веса K10/15. Размеры и affine совпали у всех 22 пациентов. Все концы лежат на маске (5 180 train, 1 840 validation). У 62/72 истинных JUNCTION два конца ближе 3 мм. Раздельные карты различили все три конца в 72/72 случаях, объединённая — в 20/72. [Preflight](../../varvara/anza_lira_local_results/ct28_structure_step1_20261116/preflight.json) · [Близкие концы](../../varvara/anza_lira_local_results/ct28_structure_step1_20261116/close_tips.json)
- **Патчи.** 15 092 патча 24³, шаг 1 мм, `fixed_lps`. Вход — HU и три карты концов, у PAIR третья нулевая. Покрытие КТ <75% у пяти исходных train-патчей (982) и 67 возмущённых (66 у 982, один у 956), на validation — 0. Заполнение −1000 HU только за границей КТ. [Аудит патчей](../../varvara/anza_lira_local_results/ct28_structure_step1_20261116/fixed_lps_patch_geometry.json)
- **MPS.** Для обеих сетей и типов кандидатов проверены обучение, инференс и загрузка весов (расхождение скоров ≤ 5,96 × 10⁻⁸). [Короткие прогоны](../../varvara/anza_lira_local_results/ct28_structure_step2_20261116/summary.json)
- **Энкодеры.** `conv3d_sep_structure` и `anza3d_sep_structure` отдельно для PAIR и JUNCTION. Маски только на train (dense loss 0,35, path loss 0,15). AdamW, lr 0,001, weight decay 0,0001, 8 эпох, batch 2, `ranking_weight=0`. Пять OOF-фолдов по пациентам, финальные веса на всех 17 train-пациентах. Обучение и подготовка HGB — 28,48 ч. [Протоколы и веса](../../varvara/anza_lira_local_results/ct28_structure_step3_seed_20261102/outputs/) · [План](CT28_ANZA_STRUCTURE_EXPERIMENT_2026-11-11.md)
- **Каскад.** Relation HGB обучена на 84 признаках для 1 036 сцен и 15 копий каждой (0,8 мм, 20°, seed `20260920`, веса оригиналов/копий 0,5/0,5). `neural_pair_junction` выбирает PAIR и JUNCTION по скору сети. Для принятия нужны совместимые связи, вероятность HGB ≥ 0,85 и тот же набор связей в ≥ 9 из 15 повторов.

## 368 готовых validation-сцен · выбор связей

Top-1 — сцены с истинным кандидатом на первом месте. AUROC и AUPRC по 2 298 PAIR и 1 773 JUNCTION. Сравнение прежних и текущих энкодеров. [Предыдущий запуск](../../artifacts/varvara/ct28_graph_lira/ccta3d/README.md)

| Кандидат | Модель | Top-1, было → стало | AUROC, было → стало | AUPRC, было → стало |
| --- | --- | ---: | ---: | ---: |
| PAIR | CNN | 106/170 → 149/170 (+43) | 0,7133 → 0,9635 | 0,2536 → 0,6955 |
| PAIR | ANZA | 76/170 → 145/170 (+69) | 0,5924 → 0,9618 | 0,0980 → 0,6798 |
| JUNCTION | CNN | 55/72 → 57/72 (+2) | 0,8042 → 0,9733 | 0,1181 → 0,5944 |
| JUNCTION | ANZA | 61/72 → 55/72 (−6) | 0,7866 → 0,9321 | 0,1333 → 0,3098 |

Train OOF на 17 пациентах. Каждый скор получен моделью, обученной без этого пациента.

| Кандидат на train OOF | CNN top-1 | CNN AUROC / AUPRC | ANZA top-1 | ANZA AUROC / AUPRC |
| --- | ---: | ---: | ---: | ---: |
| PAIR | 417/486 | 0,9533 / 0,6330 | 413/486 | 0,9298 / 0,5716 |
| JUNCTION | 177/214 | 0,9247 / 0,2930 | 165/214 | 0,8918 / 0,2026 |

Связь нужна в 206 из 368 сцен, ложные решения считаются по всем 368. Неполное решение включает часть требуемых связей.

| 368 сцен | До consistency, точные / ложные / неполные | После consistency, точные / ложные / неполные | Точные после consistency, PAIR / JUNCTION / BOTH |
| --- | ---: | ---: | ---: |
| K10/15, замороженный контроль | 155 / 11 / 1 | 52 / 0 / 0 | 48 / 2 / 2 |
| CNN, исходная HGB | 146 / 23 / 1 | 27 / 0 / 0 | 26 / 1 / 0 |
| CNN, калиброванные входы HGB | 145 / 22 / 0 | 29 / 0 / 0 | 28 / 1 / 0 |
| ANZA, исходная HGB | 137 / 25 / 0 | 27 / 1 / 0 | 26 / 1 / 0 |
| ANZA, калиброванные входы HGB | 138 / 25 / 0 | 28 / 1 / 0 | 27 / 1 / 0 |

В прежнем `neural_pair` после consistency у CNN 52/13/1, у ANZA 31/7/0. Изменились входы, обучение и выбор JUNCTION. У текущих CNN и ANZA 21 общее точное решение и по 6 своих. Ложный PAIR у ANZA (974) подтвердился в 9/15 повторах.

### Где теряется правильная связь в основном сравнении

При тех же вероятностях HGB выбор по сети вместо K10/15 изменил ID PAIR / JUNCTION у CNN в 5 / 11 сценах и 553 / 224 повторах, у ANZA — в 7 / 11 сценах и 583 / 193 повторах.

| Истинный кандидат занял первое место на исходной validation-сцене | HGB исключила тип | Ниже 0,85 | Другие концы или несовместимость | Отказ consistency | Истинный кандидат принят |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN · PAIR | 2 | 21 | 1 | 99 | 26 |
| CNN · JUNCTION | 2 | 17 | 0 | 37 | 1 |
| ANZA · PAIR | 3 | 25 | 1 | 90 | 26 |
| ANZA · JUNCTION | 1 | 21 | 1 | 31 | 1 |

- **Все сцены (CNN / ANZA).** Тип верен до порога в 89,1% / 91,3%. Отказы по порогу — 79 / 79, NONE — 119 / 127, отказы по consistency — 143 / 134.
- **Точные до consistency (CNN / ANZA).** Отклонены 119 / 110 сцен. В несовпавших повторах с исходным типом уверенность упала 597 / 541 раз, ID сменился при уверенности выше порога 420 / 399 раз.

[Скоры кандидатов](../../varvara/anza_lira_local_results/ct28_structure_step4_seed_20261102/candidate_scores.csv.gz) · [Решения и отказы](../../varvara/anza_lira_local_results/ct28_structure_step4_seed_20261102/scene_decisions.csv.gz) · [Повторы](../../varvara/anza_lira_local_results/ct28_structure_step4_seed_20261102/repeat_decisions.csv.gz)

## Пять полных масок · восстановление сосудов

- **Вход.** Пять общих для всех моделей повреждённых масок, 22 окна и 15 искусственных PAIR-разрывов. У всех разрывов есть подходящий кандидат. Кандидаты JUNCTION есть в двух окнах. Сверены SHA входов, состав окон, размеры, affine и новые воксели NIfTI.
- **Путь и проверка.** Path v2, радиус 0 мм. Разрыв засчитывается при верных компонентах, концах в пределах 3 мм от краёв и пересечении удалённого просвета. Ошибки внутри одного дерева считаются отдельно.

| Модель на 22 окнах | Отказы по tau / consistency, исходная HGB | Отказы по tau / consistency, калиброванные входы HGB |
| --- | ---: | ---: |
| K10/15, контроль | 22 / 0 | — |
| CNN | 19 / 3 | 21 / 1 |
| ANZA | 21 / 1 | 21 / 1 |

K10/15 и обе версии HGB оставили маски без добавлений. Покрытие разрывов — 0/1 339 вокселей просвета и 0/96 центральной линии. Dice/clDice по целому эталону, время и RSS на CPU с тремя потоками.

| Пациент | Окна / разрывы | Dice до = после | clDice до = после | Время, с K10/15 / CNN / ANZA | Пик RSS, МиБ K10/15 / CNN / ANZA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 957 | 5 / 3 | 0,994458 | 0,994410 | 6,46 / 8,32 / 14,11 | 928 / 950 / 950 |
| 961 | 5 / 3 | 0,995424 | 0,994055 | 5,79 / 7,06 / 11,23 | 902 / 948 / 936 |
| 965 | 3 / 3 | 0,995530 | 0,991842 | 5,46 / 6,01 / 8,68 | 920 / 893 / 932 |
| 966 | 5 / 3 | 0,996692 | 0,993341 | 7,98 / 10,58 / 19,66 | 1 036 / 924 / 1 000 |
| 974 | 4 / 3 | 0,997827 | 0,994289 | 6,31 / 7,93 / 13,54 | 942 / 1 056 / 976 |

**Train-пилот 953.** Все три модели отклонили 4/4 окна. Dice 0,995345 и clDice 0,995752 сохранились. [Метрики по пациентам](../../varvara/anza_lira_local_results/ct28_structure_step5_seed_20261102/patient_metrics.csv) · [Выходные маски](../../varvara/anza_lira_local_results/ct28_structure_step5_seed_20261102/validation/)

### Три случая отказа

- **957, разрыв 2.** Прежняя ANZA подтвердила E0010 + E0016 в 15/15 повторах и добавила 6 вокселей внутри эталона. У текущих CNN и ANZA — 0/15 с калибровкой и без неё. [Окна и вероятности](../../varvara/anza_lira_local_results/ct28_structure_step5_seed_20261102/patient_957_anza_replay.csv)
- **966, разрыв 3.** CNN выбрала верную пару (0,864, совпадение в 7/15 повторах), ANZA остановилась на пороге (0,815). После калибровки у обеих сетей порог прошёл только W0002. CNN подтвердила пару 3/15 раз, ANZA — 2/15 при минимуме 9/15. CNN с весом ранжирования 0,5 восстановила разрыв и добавила 8 вокселей внутри эталона. [Аудит связи](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/ranking/masks/full_mask_audit.json)
- **965, разрыв 2.** Скоры правильной пары у CNN / ANZA — 0,885 / 0,871. Relation HGB выбрала NONE для CNN и дала PAIR вероятность 0,616 для ANZA. [Три случая на КТ](../../varvara/anza_lira_local_results/ct28_structure_step5_seed_20261102/failure_cases.png)

## Калибровка локальных скоров

Калибровка Платта `p = sigmoid(a × logit(score) + b)` сохраняет ранги при `a > 0`. Для train-пациента и копий параметры найдены по остальным 16 пациентам, для validation — по всем 17 train. Метки validation использованы для оценки. Возможна косвенная зависимость внутри train через энкодеры других OOF-фолдов.

| Локальные скоры | Brier train OOF до → после | Brier validation до → после | Средний прогноз на validation до → после | Доля истинных кандидатов на validation |
| --- | ---: | ---: | ---: | ---: |
| PAIR · CNN | 0,0853 → 0,0421 | 0,0825 → 0,0366 | 19,60 → 6,87% | 170/2298 (7,40%) |
| PAIR · ANZA | 0,0854 → 0,0462 | 0,0751 → 0,0377 | 18,59 → 7,27% | 170/2298 (7,40%) |
| JUNCTION · CNN | 0,1136 → 0,0369 | 0,0868 → 0,0235 | 19,17 → 4,27% | 72/1773 (4,06%) |
| JUNCTION · ANZA | 0,1252 → 0,0407 | 0,1376 → 0,0329 | 28,86 → 5,70% | 72/1773 (4,06%) |

Brier снизился у всех пяти validation-пациентов. Переобучили только HGB, концы выбрали по исходным скорам.

[Калибровка](../../varvara/anza_lira_local_results/ct28_structure_step3_seed_20261102/calibration/metrics.json) · [Каскад](../../varvara/anza_lira_local_results/ct28_calibrated_hgb_20261102/summary.json) · [Исходные OOF-скоры на 15 train-возмущениях](../../varvara/anza_lira_local_results/ct28_calibrated_hgb_20261102/raw_train_scores.sqlite3)

<!-- ct28-seed-20261102-ablations -->
## Абляции на seed 20261102

Контроль — базовые CNN и ANZA. В `neural_pair` PAIR выбирается по скору сети, JUNCTION — по геометрии.

### Вклад КТ в ранжирование кандидатов

Инференс на CPU с исходными весами (OOF для train, финальные для validation). Перемешивание сохраняет HU патча, «только концы» обнуляет нормализованный канал КТ. Ниже — top-1 / AUROC / AUPRC на development-validation.

| Вход | PAIR · CNN | PAIR · ANZA | JUNCTION · CNN | JUNCTION · ANZA |
| --- | --- | --- | --- | --- |
| КТ | 149/170 / 0,9635 / 0,6955 | 145/170 / 0,9618 / 0,6798 | 57/72 / 0,9733 / 0,5943 | 55/72 / 0,9321 / 0,3098 |
| Перемешанные HU | 104/170 / 0,8715 / 0,3100 | 104/170 / 0,8591 / 0,2968 | 52/72 / 0,9104 / 0,2805 | 47/72 / 0,8656 / 0,1616 |
| Только концы | 27/170 / 0,3437 / 0,0553 | 7/170 / 0,2257 / 0,0445 | 59/72 / 0,9124 / 0,2149 | 7/72 / 0,1392 / 0,0233 |

Удаление КТ ухудшило PAIR у обеих сетей и JUNCTION у ANZA. У JUNCTION CNN вырос top-1 и снизился AUPRC.

[Метрики validation](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/input_controls/val_metrics.csv) · [Метрики train OOF](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/input_controls/train_metrics.csv)

### Сравнение каскадов

| Условие | Модель | 368 сцен после consistency, точные / ложные / неполные | Пять масок, восстановлено разрывов / ложных связей |
| --- | --- | --- | --- |
| Исходный | CNN | 27 / 0 / 0 | 0/15 / 0 |
| Исходный | ANZA | 27 / 1 / 0 | 0/15 / 0 |
| Ranking 0,5 | CNN | 45 / 0 / 0 | 1/15 / 0 |
| Ranking 0,5 | ANZA | 46 / 2 / 0 | 0/15 / 0 |
| ANZA без осевой анизотропии | ANZA | 25 / 0 / 0 | 0/15 / 0 |
| DeepSets | CNN | 35 / 1 / 0 | 0/15 / 0 |
| DeepSets | ANZA | 46 / 2 / 0 | 0/15 / 0 |
| Выбор JUNCTION по геометрии | CNN | 27 / 0 / 0 | 0/15 / 0 |
| Выбор JUNCTION по геометрии | ANZA | 27 / 1 / 0 | 0/15 / 0 |

**Локальные метрики, вес ранжирования 0 → 0,5**

| Кандидат | Модель | Выборка | Кандидатов | Top-1 | AUROC | AUPRC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| PAIR | CNN | Train OOF | 6336 | 417/486 → 432/486 | 0,9533 → 0,9589 | 0,6330 → 0,6777 |
| PAIR | CNN | Development-validation | 2298 | 149/170 → 154/170 | 0,9635 → 0,9620 | 0,6955 → 0,7543 |
| PAIR | ANZA | Train OOF | 6336 | 413/486 → 411/486 | 0,9298 → 0,9307 | 0,5716 → 0,5991 |
| PAIR | ANZA | Development-validation | 2298 | 145/170 → 148/170 | 0,9618 → 0,9478 | 0,6798 → 0,7213 |
| JUNCTION | CNN | Train OOF | 4685 | 177/214 → 184/214 | 0,9247 → 0,9602 | 0,2930 → 0,4734 |
| JUNCTION | CNN | Development-validation | 1773 | 57/72 → 59/72 | 0,9733 → 0,9476 | 0,5944 → 0,4866 |
| JUNCTION | ANZA | Train OOF | 4685 | 165/214 → 164/214 | 0,8918 → 0,8975 | 0,2026 → 0,2307 |
| JUNCTION | ANZA | Development-validation | 1773 | 55/72 → 55/72 | 0,9321 → 0,9195 | 0,3098 → 0,2038 |

[Протоколы с весом 0](../../varvara/anza_lira_local_results/ct28_structure_step3_seed_20261102/outputs/) · [Протоколы с весом 0,5](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/ranking/outputs/)

Выбор JUNCTION по геометрии изменил развилку в 11 из 368 исходных сцен у каждой сети. После consistency совпали и принятые сцены, и развилки.

Остальные энкодеры, top-1 / AUROC / AUPRC.

| Условие | Модель | Кандидат | Train OOF, 5 фолдов | Development-validation |
| --- | --- | --- | --- | --- |
| ANZA без осевой анизотропии | ANZA | PAIR | 415/486 / 0,9296 / 0,5919 | 147/170 / 0,9653 / 0,6995 |
| ANZA без осевой анизотропии | ANZA | JUNCTION | 169/214 / 0,8986 / 0,2121 | 56/72 / 0,9333 / 0,3112 |
| DeepSets | CNN | PAIR | 433/486 / 0,9617 / 0,6791 | 155/170 / 0,9705 / 0,8068 |
| DeepSets | CNN | JUNCTION | 180/214 / 0,9641 / 0,4542 | 61/72 / 0,9660 / 0,4976 |
| DeepSets | ANZA | PAIR | 398/486 / 0,9135 / 0,5371 | 147/170 / 0,9439 / 0,5979 |
| DeepSets | ANZA | JUNCTION | 190/214 / 0,9840 / 0,6868 | 63/72 / 0,9877 / 0,7403 |

> **Артефакты в `varvara/` хранятся локально. На GitHub их нет.**

- Ranking 0,5 — [решения до и после gate](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/ranking/scenes/summary.json) · [маски и ошибки связности](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/ranking/masks/patient_metrics.csv)
- Без осевой анизотропии — [решения](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/noaxial/scenes/summary.json) · [маски](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/noaxial/masks/patient_metrics.csv)
- DeepSets — [решения](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/deepsets/scenes/summary.json) · [маски](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/deepsets/masks/patient_metrics.csv)
- JUNCTION по геометрии — [решения](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/historical_identity/scenes/summary.json) · [маски](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/historical_identity/masks/patient_metrics.csv)

[Конфигурации и порядок прогонов](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/run_matrix.py) · [Основная очередь](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/queue_status.json) · [Дополнительные сравнения](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/extension_status.json) · [Журнал возобновления](../../varvara/anza_lira_local_results/ct28_structure_ablations_20261102/resume_queue.log)
<!-- end-ct28-seed-20261102-ablations -->

## Что остаётся по исходному заданию

- Повторить полный каскад на seed `20261103` и `20261104` и свести результаты пяти пациентов по трём seed, чтобы проверить устойчивость различий между моделями.

## Отложенные абляции

- **Ориентация патча (`20261102`).** `canonical` против `fixed_lps`.
- **Карты концов (`20261102`).** Объединённая карта против раздельных.
- **Контрастная ANZA (опционально после основного сравнения).** Сравнить с обычной ANZA в полном каскаде CT28. Пока проверены скор и градиенты на искусственном входе.
