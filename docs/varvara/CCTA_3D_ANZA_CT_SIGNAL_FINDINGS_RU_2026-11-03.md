# CT28: где именно КТ добавляет информацию и что из этого следует для 3D ANZA

**Дата:** 03.11.2026 · **Источник:** сохранённые результаты исторического `local_hgb` на 17 train / 5 development-val пациентах. **Это НЕ выбранная K10/15 и НЕ результат обученной ANZA.** Контур закрытого теста (954, 958, 972, 973, 980, 984) не открывался.

Результаты фактически пересчитаны в GitHub Actions, используя сохранённые 3D-CT-derived признаки, настоящий сериализованный HGB model pack, source CSV и исходные validation labels.

- [Воспроизводимый GitHub Actions run](https://github.com/fims9000/anza_lira/actions/runs/41609984006)
- [Исходный скрипт relation-type CT sham](../../scripts/research/ccta_graph_lira_safe_repair/ct28_saved_ct_signal.py)
- [Исходный скрипт PAIR identity CT sham](../../scripts/research/ccta_graph_lira_safe_repair/ct28_saved_pair_ct_signal.py)

## 1. Relation-class prediction: CT-score даёт небольшой вклад

Проверка `NONE / PAIR / JUNCTION / BOTH` на 368 validation сценах и 4071 кандидатах:

| Режим | Raw accuracy | Log loss |
|---|---:|---:|
| Оригинальный исторический local-HGB | **341/368 = 92.663%** | **0.22403** |
| 30 перемешиваний `ct_score` внутри каждого пациента и типа кандидата, геометрия сохранена | В среднем **91.649%** | В среднем **0.25235** |

Средняя разница accuracy составила **+1.014 процентного пункта** в пользу исходного `ct_score`. В **4 из 30** независимых перестановок точность не ниже оригинала. Это ограниченный результат: перемешивался **только `ct_score` relation head**, но не `geometry_plus_ct_score` PAIR identity, который используется на следующем этапе. Relation accuracy нельзя смешивать с exact structural repair.

## 2. PAIR identity: CT-фичи существенно улучшают ранжирование

Frozen `local_hgb` на тех же пяти validation пациентах: **2298 PAIR кандидатов**, **170 истинных** в 170 сценах.

| Модель/контроль | ROC-AUC | AUPRC | Истинный PAIR в top-1 |
|---|---:|---:|---:|
| Geometry-only | 0.97698 | **0.71745** | **153/170 = 90.00%** |
| Geometry + CT | **0.99102** | **0.90117** | **167/170 = 98.24%** |
| Geometry + CT, но 38 исходных CT-признаков переставлены внутри каждого пациента (20 перестановок) | В среднем 0.91093 | В среднем **0.41556** | В среднем **70.88%** |

PAIR top-1 discordance исходных geometry и geometry+CT: **15** сцен исправила fusion, **1** сцена ухудшилась относительно geometry, **152** были правильными у обеих, **2** ошибочными у обеих. Чистый выигрыш **14/170 = +8.235 процентного пункта**.

По пациентам:

| ID | PAIR-сцен | Geometry top-1 | Geometry+CT top-1 | Разница |
|---|---:|---:|---:|---:|
| 957 | 34 | 28 | 34 | +6 |
| 961 | 41 | 39 | 40 | +1 |
| 965 | 27 | 23 | 25 | +2 |
| 966 | 42 | 40 | 42 | +2 |
| 974 | 26 | 23 | 26 | +3 |
| **Всего** | **170** | **153** | **167** | **+14** |

Постфактум descriptive bootstrap с ресемплированием **пяти пациентов** (5000 реплик, seed 20261103): разница PAIR top-1 **+8.24 п.п.**, интервал **[+4.15, +13.61] п.п.** Это **НЕ** доверительный интервал независимой проверки из-за выбранных по validation моделей и малого числа пациентов.

### Что именно доказывает shuffle

Перестановка в PAIR identity производится на уровне **целого 38-мерного CT feature vector** в пределах пациента. Тем самым сохраняются размеры классов, геометрические признаки, численность/идентичность кандидатов, модель, эмпирическое CT-feature распределение в каждом пациенте. Разрушается соответствие конкретного CT-вектора конкретному кандидату. Сильное падение AP указывает, что конкретные CT-фичи существенно влияют на PAIR identity **в этих синтетических сценах**. Однако shuffled пары могут быть физически несогласованными, что само по себе ухудшает HGB; это не причинное доказательство клинической значимости.

## 3. Исправление исследовательской архитектуры

До этой редакции 3D ANN (`conv3d`/`anza3d`) подставлялась **только в `ct_score` для relation head**, тогда как финальный выбор **PAIR identity** продолжал использовать прежний HGB `geometry_plus_ct_score`. Таким образом предыдущий протокол не мог честно проверить, улучшает ли ANZA выбор истинной пары концов.

Теперь в `NeuralCCTA3dInference` есть две **явные, протоколируемые** политики:

- `--pair-identity-source frozen_fusion` — предыдущее поведение: PAIR identity использует выбранную HGB geometry+CT, ANN влияет на relation HGB.
- `--pair-identity-source neural_pair` — ANN score используется и для CT stream relation head, и непосредственно как **PAIR identity** при ранжировании кандидатов. JUNCTION identity по-прежнему geometry-first. Выбор направления конфликта глобальных реконструкций и 15x consistency выполняются по той же policy/solvers; никакой скрытой активации режима.

Единичный synthetic regression проверяет **фактическую смену выбранного PAIR candidate ID** при переключении policy, а не только смену чисел в объекте.

Это создаёт честное **ANN-vs-CNN** сравнение по PAIR identity при одинаковом протоколе и тех же кандидатах. Однако сравнение этих NN с выбранной K10/15 всё ещё смешивает разные формы локальных features/augmentations; его нужно обозначать как *development baseline*, пока ANN relation head не переобучен на идентичных 15 train perturbation copies.

### Рекомендуемый режим 3D ANN vs CNN на TRAIN

После локального получения модели/CT28 патчей через готовый train OOF:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_study.py \
  --stage all --patient-id 953 --device cuda --epochs 8 --folds 5 \
  --pair-identity-source neural_pair \
  --root varvara/anza_lira_local_results/ccta_3d_runtime_pilot/anza_vs_cnn_neural_pair
```

В validation использовать **отдельный `--output-root`** и `--pair-identity-source neural_pair`; скрипт сверяет эту политику с сохранёнными JSON/metrics и не позволит сопоставить отчёты с разной политикой. Следующие шаги без скрытых manual choices: OOF 3D CNN/ANZA → pair top1/AP → frozen relation type → 3D mask ΔDice/clDice + GT false bridges → только после freeze независимый test.

## 4. Научное решение

**Основная гипотеза ANZA:** нейросеть с симметричной локальной направленной метрикой и нечёткими пространственными правилами может повысить точность определения *соответствующих друг другу концов сосудов* на 3D CCTA относительно компактной 3D CNN при одинаковой ёмкости, train-only оптимизации и физической нормализации патча.

**Primary candidate metrics:** PAIR top-1 exact ID и AUPRC на пяти development-val пациентах, patient-level descriptives и control endpoint-only/CT-sham. **Primary full-volume safety metrics:** correct new GT component merges, unknown GT merges, false inter-component bridges, added voxels вне reference, ΔclDice, доля abstain.

**Stop-критерий:** если ANZA не улучшает CNN при одинаковом budget/OOF и не сохраняет риск ложных соединений, обоснование преимущества конкретной ANZA-конструкции **не подтверждается**. Тогда честная статья должна фокусироваться на selective graph repair и локальном CT evidence, а не на неподтверждённом архитектурном выигрыше.

## 5. Контракт исходных данных

Проверены source SHA256: `scene_plan_train_val.csv` = `746e2d8e48c33ea281d6858a667189ae0ffb8106c60a7ef35935a6320e360871`; `scene_candidate_features_train_val.csv` = `dc406fdac832cb55c7e59417102cb1adf8bcef3b6afa25c98c3ff6958a6cedee`; historical HGB model = `3985e5c3b81a1c1be60887adbefd2622f992c541f3ac3e472364b65ecc0ad204`; saved local candidate scores = `71cf7e7436acc49dbf166254dbe6855c4d8c5b96f56df2a46b55a4d8ba94f0a8`.

**Неизмерено:** обученная 3D ANZA/CNN на CCTA, full-mask performance на настоящих CT пациентах, natural segmentation errors, closed six-patient test. Не заменять эти поля числами из исторического HGB.
