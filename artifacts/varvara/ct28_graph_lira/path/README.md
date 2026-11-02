# CT28 path stage: max–min v1 и выбранный по train v2

Код невыбранных экспериментов перенесён в [архив](../ablations/rejected/README.md); команды ниже используют исходные пути после его восстановления.
Ниже сначала описан **новый** CT28 path protocol v1, затем выбранный по train вариант v2. Из старой Graph-LIRA взяты принцип max–min и порядок «сначала принятое структурное решение, потом путь». Вес ребра CCTA и правило общего узла JUNCTION заданы заново; они не объявляются восстановленным историческим алгоритмом. Held-out test не открывался.

## Вход и неизменное решение Graph-LIRA

Path solver получает выбранные PAIR/JUNCTION endpoints, их касательные и CCTA, прошедшую patient-level SHA256/geometry checks. Исходные VTK centerlines, названия ветвей и segmentation **не подаются** в solver; они используются только после построения для аудита на train/validation. `NONE` и abstain не дают маршрута. Path stage не меняет выбранный candidate ID, relation type, `tau=0.85`, consistency `0.60` или структурные счётчики.

## Как строится маршрут

| Конструкция | Правило |
|---|---|
| 3D граф | Воксели CT в локальной области, 26 соседей. |
| PAIR | Коридор радиуса `r` вокруг прямого отрезка между endpoints; ищется max–min путь. |
| Degree-3 JUNCTION | Три коридора от endpoints к центру, оценённому по касательным; общий CT-воксель разрешён в пределах `c` мм. Все три arm-пути непустые и заканчиваются в **одном** узле; перекрытие допускается только в пределах 0.75 мм от него. |
| BOTH | PAIR и JUNCTION строятся отдельно; проверяется отсутствие дополнительного контакта их маршрутов. |

**Основа таблицы:** это правила одного нового `ct28_path_protocol_v1`, применённого к текущему K10/15. PAIR и JUNCTION — разные конструкции, победителя между ними нет; исторический CT28 path protocol не подменялся этим как «восстановленный».

Опорная яркость видимого сосуда `Hv` — медиана HU в 0.5, 1, 1.5 и 2 мм **снаружи** каждого endpoint. Фон `Hb` — медиана на кольцах радиуса 1.5 и 2.5 мм вокруг тех же видимых участков. Перед медианой HU обрезаются до `[-300,1200]`. Для вокселя `v`:

```text
sv = max(80, 0.5 * max(Hv-Hb, 0))
sb = max(40, 0.25 * max(Hv-Hb, 0))
s(v) = exp(-0.5 * ((HU(v)-Hv)/sv)^2) * sigmoid((HU(v)-Hb)/sb)
a(u,v) = min(s(u), s(v)) * exp(-abs(HU(u)-HU(v))/d)
```

Путь максимизирует **минимальный** вес ребра `a`; при равном bottleneck выбирается меньшая физическая длина. Для JUNCTION выбирается один общий узел, максимизирующий худшую поддержку из трёх arm-путей, затем меньшую суммарную длину. Исправлена ошибка, при которой общий узел совпадал с endpoint и одна arm имела нулевую длину.

Топологический успех означает: достигнуты все **выбранные** endpoints, шаги соответствуют 26-соседству, маршрут не касается невыбранного **endpoint сцены** или второй конструкции BOTH ближе 0.75 мм. Это не проверка всех сосудов в полном объёме CT.

## Подбор только на train

Заранее проверены 12 сочетаний: `r∈{1.5,2,2.5}` мм, `c∈{1,2}` мм, `d∈{100,200}` HU. По одному `pair_only` и `junction_only` от каждого из 17 train-пациентов — 34 сцены. Цель учитывает близость к исходной маске только **после** маршрутизации, неудачи построения, касания чужих scene endpoints и чрезмерную длину. Выбраны `r=2 мм`, `c=2 мм`, `d=200 HU`; две лучшие aggregate оценки очень близки, `0.721636` против `0.721615`, поэтому точный оптимум не установлен. На 34 train-сценах 29 маршрутов прошли топологическую проверку; пять неудач — JUNCTION (четыре касания другого scene endpoint, одно перекрытие arm вне узла). Машинные источники: `varvara/anza_lira_local_results/ct28_path_protocol_v1/chosen_config_train.json` и `tuning_summary_train.csv`.

## Проверка на неизменной validation

Вход — сохранённое решение CT-conditioned K10/15: из 368 сцен приняты 52 exact structural repairs, 48 `pair_only`, 2 `junction_only`, 2 `mixed`. Маршруты построены для всех 52; все связны, ненулевые, без лишнего касания других **scene endpoints** и без контакта конструкций BOTH. Число принятых repair от построения пути не выросло.

Близость к **целой** исходной segmentation на допуске 0.75 мм: средняя доля маршрута 0.9971, минимум 0.9048. Медиана отношения длины пути к прямой — 1.81, p90 3.24, максимум 5.43. Четыре оценённых общих JUNCTION-узла имеют ошибку 1.42–1.80 мм относительно размеченного branch point. Эти метрики не доказывают, что путь проходит **по нужной ветви**.

Поэтому сделан отдельный post-hoc аудит **намеченных centerline-дуг**. Все 50 PAIR-маршрутов сопоставлены одной исходной VTK polyline между выбранными endpoints; все 12 arms четырёх JUNCTION-конструкций — размеченным incident polylines от exposed endpoint до true branch point. Дуги семплировались через 0.25 мм; измерялись расстояния в обе стороны, покрытие и длина. Ничего из этих reference-дуг не подавалось в solver.

| Анатомический аудит | PAIR | JUNCTION arms |
|---|---:|---:|
| Средняя доля маршрута ближе 0.75 мм к нужной дуге | 0.863 | 0.651 |
| Среднее покрытие reference-дуги при 0.75 мм | 0.943 | 0.702 |
| Доля маршрута ближе 1.5 мм к нужной дуге | 0.991 | 0.913 |
| Медиана длины маршрута / длины нужной дуги | 1.72 | 2.62 |

**Основа сравнения:** все 52 маршрута построены после принятия текущего K10/15; прежнего CT28 path-baseline на тех же сценах нет. PAIR и JUNCTION имеют разные задачи и объёмы (50 маршрутов против 12 arms), поэтому 0.863 против 0.651 — **не победа одной модели над другой**. Для JUNCTION близость к нужным дугам остаётся слабее; параметры path stage по validation не заменялись.

Три JUNCTION arms в одной mixed-сцене подходят ближе 0.75 мм к другой VTK polyline, которая сама разделяет исходный узел с выбранной arm; это записано как adjacent-line proximity, **не** объявлено ложным repair. Близости к топологически несоседней polyline в этом аудите нет. Нельзя заявить «52 анатомически правильных пути»: 52/52 означает лишь топологический успех по локальным проверкам, а JUNCTION-маршруты часто отклоняются от нужной дуги. Расстояние до centerline также не является полной клинической проверкой, поскольку корректный путь может проходить внутри просвета на некотором расстоянии от осевой линии.

Выходы в `varvara/anza_lira_local_results/ct28_path_protocol_v1/`: `path_metrics_val.csv` — все 368 сцен, включая skips; `paths_val.jsonl` — 52 маршрута; `metrics_val.json` — компактные метрики; `path_anatomy_metrics_val.csv` — каждая arm с true line ID, расстояниями/покрытием/длиной; `anatomy_summary_val.json` — агрегаты и SHA. Повторный запуск из корня репозитория: `run_ct28_path_stage.py --phase tune`, затем `--phase evaluate`, затем `audit_ct28_path_anatomy.py` из `scripts/research/ccta_graph_lira_safe_repair/` существующим `.venv/bin/python`.

```bash
.venv/bin/python scripts/research/ccta_graph_lira_safe_repair/run_ct28_path_stage.py --phase tune
.venv/bin/python scripts/research/ccta_graph_lira_safe_repair/run_ct28_path_stage.py --phase evaluate
.venv/bin/python scripts/research/ccta_graph_lira_safe_repair/audit_ct28_path_anatomy.py
```

## Выбранный по train вариант v2

V1 выше сохранён как исходный max–min контроль. Для v2 на тех же 34 train-сценах проверены допуски к лучшему bottleneck `0`, `5`, `10` и `20%`. Выбраны `5%`: маршрут остаётся на CT-рёбрах с поддержкой не ниже `0.95` от max–min bottleneck, а среди них минимизирует физическую длину. Для JUNCTION один общий узел выбирается по суммарной длине трёх допустимых arms; остальные ограничения v1 сохраняются. Это **новое** правило, а не восстановленный исторический алгоритм.

V2 также исключает воксели около невыбранных endpoint’ов сцены, кроме области начала выбранной arm. Контакт с decoy на расстоянии `<=0.75 мм` от выбранного endpoint помечается отдельно: запретить его в начальной точке геометрически невозможно. В train это изменило топологическую оценку с `29/34` по старому правилу на `33/34` по новому; четыре прежних отказа были связаны с почти совпадающими selected/decoy endpoints. Оставшийся отказ — перекрытие JUNCTION arms вне общего узла у пациента 964. Разница `29→33` включает **изменение правила аудита**, поэтому её нельзя целиком приписать поиску более короткого пути.

| Проверка на тех же принятых validation-сценах | V1 max–min | V2, выбранный по train |
|---|---:|---:|
| Топологически построены | 52/52 | 52/52 |
| PAIR: средняя доля пути рядом с нужной дугой, 0.75 мм | 0.863 | 0.955 |
| PAIR: медиана длины / длины нужной дуги | 1.72 | 1.16 |
| JUNCTION: средняя доля arm рядом с нужной дугой, 0.75 мм | 0.651 | 0.870 |
| JUNCTION: среднее покрытие нужной дуги, 0.75 мм | 0.702 | 0.889 |
| JUNCTION: медиана длины arm / длины нужной дуги | 2.62 | 1.35 |
| JUNCTION: медианная ошибка общего узла | 1.64 мм | 0.86 мм |
| Касания несмежной VTK polyline, post-hoc | 0 | 0 |

**Основа сравнения:** одни и те же 52 принятые K10/15 validation-сцены; v1 — контроль, v2 — текущий path-кандидат. V2 выбран по train, validation использована для проверки. Только четыре JUNCTION-конструкции дали 12 arms, поэтому улучшение анатомической близости пока имеет большой статистический разброс. Само структурное решение остаётся **52 точных / 0 ложных**; path stage не прибавляет принятых repair и не создаёт финальную исправленную segmentation mask.

У всех 52 validation-маршрутов фактический bottleneck составил не меньше `0.95025` от v1; это проверяет ограничение `>=0.95` после построения, а не только в настройке алгоритма.

Машинный протокол с SHA256: `selected_path_protocol.json`. Полные v2 результаты находятся в игнорируемой Git папке `varvara/anza_lira_local_results/ct28_pretest_audit/{path_v3_train,path_v3_val}/`. Воспроизведение из корня репозитория после восстановления raw CT и текущего K10/15:

```bash
.venv/bin/python scripts/research/ccta_graph_lira_safe_repair/audit_ct28_train_path_anatomy.py --slack-fraction 0.05 --ignore-unavoidable-endpoint-contact --avoid-other-endpoints --out-dir varvara/anza_lira_local_results/ct28_pretest_audit/path_v3_train
.venv/bin/python scripts/research/ccta_graph_lira_safe_repair/run_ct28_path_stage.py --phase evaluate --chosen-config varvara/anza_lira_local_results/ct28_path_protocol_v1/chosen_config_train.json --slack-fraction 0.05 --ignore-unavoidable-endpoint-contact --avoid-other-endpoints --out-dir varvara/anza_lira_local_results/ct28_pretest_audit/path_v3_val
.venv/bin/python scripts/research/ccta_graph_lira_safe_repair/audit_ct28_path_anatomy.py --path-dir varvara/anza_lira_local_results/ct28_pretest_audit/path_v3_val
```

## Train-only пилот path evidence для кандидатов

Проверен новый признак **до relation head**, отдельно от построения путей после принятых repair. Выборка зафиксирована до CT-расчётов: первая eligible `pair_only` и `junction_only` сцена каждого из 17 train-пациентов; true и false с максимальным geometry OOF score. Итого 34 matched sources, 68 кандидатов, clean+5 position-only смещений `σ=0.8 мм` с общим scene jitter. Это диагностическая выборка 1:1, не новый баланс полного candidate pool. Фильтры после смещения не применялись.

Solver получает только наблюдаемые endpoints/касательные и проверенный CT. VTK и segmentation не открываются. Параметры уже зафиксированы по train: радиус 2 мм, уточнение узла 2 мм, HU-scale 200, bottleneck slack 5%. Для candidate-only evidence маска других endpoints выключена; это отличие от repair path v2. Нового подбора параметров нет. Primary score — минимальный вес ребра; secondary — худший arm edge-q10. Длина, извилистость, HU reference и смещение центра также сохранены.

| Train-кандидаты | Bottleneck AUROC | True выше matched false | Построены true / false |
|---|---:|---:|---:|
| PAIR clean | 0.561 | 9/17 | 17/17 · 17/17 |
| PAIR position-only | 0.533 | 41/85 | 85/85 · 85/85 |
| JUNCTION clean | 0.711 | 12/17 | 16/17 · 12/17 |
| JUNCTION position-only | 0.550 | 45/85 | 85/85 · 83/85 |

Все 408 расчётов сохранены: 400 успешных маршрутов и 8 `branch_overlap`; failures дают нулевую evidence, не исключаются. Для JUNCTION clean AUROC среди только построенных маршрутов — 0.641, поэтому общий 0.711 частично отражает разницу в availability. Secondary score при смещениях также слаб: AUROC 0.532 PAIR и 0.546 JUNCTION. Patient-cluster bootstrap по 17 пациентам для noisy paired-win: PAIR 48.2% [34.1%; 62.4%], JUNCTION 52.9% [44.7%; 61.2%]. Повторы не считаются независимыми пациентами.

**Диагноз:** точная привязка к endpoint-вокселю ограничивает max–min маршрут. У true PAIR медиана минимального endpoint HU меняется `254→55`, у JUNCTION `313→108`; медианная верхняя граница support в endpoints — `0.824→0.206` и `0.862→0.058`. На noisy true routes медиана отношения bottleneck к этой границе около 0.94. Все endpoints внутри CT, SHA/shape/spacing/affine проверены у 17/17 пациентов. Это локализует чувствительность, но не является исправленным inference.

**Решение:** текущий path evidence не переносить в local models: устойчивого разделения true/hard-negative при смещениях нет. Обучение моделей, новый Graph-LIRA repair/abstain и validation/test здесь не выполнялись; tau/consistency сохранены. Пилот не меняет прежние 52/54/55 принятых сцен. CT attachment в небольшой области вокруг наблюдаемых endpoints проверен отдельно ниже.

Код — `scripts/research/ccta_graph_lira_safe_repair/pilot_ct28_path_evidence.py`. В `varvara/anza_lira_local_results/ct28_path_evidence_pilot_v1/`: `pilot_plan_train.csv`, `protocol.json` с SHA, `path_features_train.csv`, `paired_comparisons_train.csv`, `per_patient_train.csv`, `status_counts_train.csv`, `endpoint_audit_train.csv`, `metrics.json`. Primary/secondary score заданы до расчётов; learned weights и thresholds не выбирались. Из корня репозитория:

```bash
.venv/bin/python scripts/research/ccta_graph_lira_safe_repair/pilot_ct28_path_evidence.py
.venv/bin/python scripts/research/ccta_graph_lira_safe_repair/pilot_ct28_path_evidence.py --audit-endpoints
```

Скрипт возобновляет extraction по сохранённым ключам только при совпадении входов/протокола; `--check-only` проверяет синтетическую трубку с разрывом, `--summarize-only` пересобирает метрики без поиска путей.


## CT-привязка endpoints: train-only абляция

**Контроль:** exact-endpoint path evidence из предыдущего раздела. Использованы побайтово тот же план из 68 кандидатов, 17 train-пациентов и те же 408 clean/position-only представлений. Validation/test не оценивались. Это признак для кандидата, не проверка repair после gate.

Заранее задан один вариант, без подбора по результатам: endpoint разрешено сдвинуть внутри **1 мм**, сохраняя его дробное положение относительно CT-сетки. Для каждой точки оценивается CT-support с vessel reference `max(исходный наблюдаемый vessel HU, локальный HU q90)` и исходным наблюдаемым background HU. Выбирается совместное размещение endpoints в разных CT-вокселях, максимизирующее произведение `support × exp(−d²/(2σ²))`, где `σ=1 мм`. Метки true/false и clean-координаты при выборе не используются.

После привязки прежний solver заново оценивает центр и HU references по скорректированным точкам. Primary evidence — полученный bottleneck, умноженный на `exp(−Σdᵢ²/(2σ²))`; secondary q10 получает тот же штраф. Это составная абляция привязки и пересчёта локального контекста; изменение нельзя приписывать только одному из этих действий. Скорректированные точки используются только внутри этой evidence; source candidate IDs и наблюдаемые endpoints не заменяются в Graph-LIRA. Соединение исходной точки с привязанной не оценивается как анатомический repair.

| Те же train-кандидаты | AUROC: контроль → attachment | True выше matched false: контроль → attachment |
|---|---:|---:|
| PAIR clean | 0.561 → 0.564 | 9/17 → 9/17 |
| PAIR position-only | 0.533 → 0.534 | 41/85 → 46/85 |
| JUNCTION clean | 0.711 → 0.739 | 12/17 → 12/17 |
| JUNCTION position-only | 0.550 → 0.605 | 45/85 → 48/85 |

**Лучший в таблице по AUROC:** attachment во всех четырёх строках, но PAIR-разница практически нулевая. Patient-bootstrap 95% интервал изменения noisy paired-win: PAIR `+5.9 п.п. [−1.2; +13.0]`, JUNCTION `+3.5 п.п. [−7.1; +15.3]`; убедимого прироста не установлено. Существуют только 17 независимых пациентских кластеров.

У JUNCTION noisy построены 83/85 true и 74/85 false маршрутов, против 85/85 и 83/85 в контроле. AUROC среди только построенных путей: `0.539→0.561`; часть общего прироста связана с availability. По всем 408 представлениям теперь 389 успешных маршрутов и 19 `branch_overlap` (3 true, 16 false), против прежних 400/8. Failures сохранены с нулевой evidence. Без финального штрафа диагностический noisy AUROC attachment — PAIR 0.538, JUNCTION 0.604; это не альтернативный выбранный score.

Среднее максимальное смещение noisy true endpoint-наборов: PAIR 0.48 мм, JUNCTION 0.48 мм; медианный множитель штрафа — 0.839 и 0.797. Для false — 0.41/0.55 мм и 0.882/0.775. Ни одно смещение не превысило 1 мм.

При расчёте обнаружен цикл предшественников пути у нового false JUNCTION пациента 967. В `ct28_path_stage.reconstruct` добавлена защита: циклическая цепочка возвращает отсутствие пути, вместо бесконечного обхода. Воспроизведён конкретный двухвоксельный цикл; проверки cyclic/acyclic цепочек прошли. Все **408 исходных** exact-endpoint результатов после исправления воспроизведены с теми же status/bottleneck/q10 при нулевом радиусе. Также прошли проверки синтетической трубки, различных вокселей arms, физического радиуса и преобразования координат на наклонном анизотропном affine. Исходные feature CSV и выбранная K10/15 модель не изменены.

**Решение от 2026-11-02:** завершить эту линию улучшений, сохранить отдельной непринятой абляцией и оставить K10/15 рабочим baseline. Attachment частично помогает JUNCTION, но не даёт устойчивого разделения true/hard-negative; число успешных путей также снизилось. Гипотеза не дала основания заявлять рост принятых repair. Tau 0.85, consistency 0.60 и test не менялись; модели не обучались.

Код: `scripts/research/ccta_graph_lira_safe_repair/pilot_ct28_path_attachment.py`. Результаты в `varvara/anza_lira_local_results/ct28_path_attachment_pilot_v1/`: `protocol.json` с SHA исходных данных/solver, `path_features_train.csv` с исходным и penalized scores, shifts/penalty/attached positions, `comparison_train.csv`, `paired_comparisons_train.csv`, `per_patient_train.csv`, `status_counts_train.csv`, `metrics.json`. Повторный запуск безопасно возобновляет только совпадающий протокол:

```bash
.venv/bin/python scripts/research/ccta_graph_lira_safe_repair/pilot_ct28_path_attachment.py
```

Полный локальный аудит дополнительно уточнил provenance attachment-пилота: `source_sha256.path_solver` теперь указывает фактически использованный solver с защитой от циклов, а `baseline_path_solver_sha256` сохраняет историческую версию контроля. Старый protocol и точный исходник исторического solver с совпадающим SHA сохранены в локальных audit evidence; features и численные сравнения не менялись.
