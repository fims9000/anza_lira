# CT28 Graph-LIRA: состояние исследования

Статус: **train/validation development**. Held-out test пациентов 954, 958, 972, 973, 980, 984 не открывался. Здесь зафиксированы данные и текущий validation-претендент; результаты на пяти validation-пациентах не являются итоговой оценкой обобщения.

## Текущий вариант — один

**Решение от 2026-11-02:** линия улучшения через candidate CT path evidence и CT-привязку endpoints завершена без переноса в модель. Основной рабочий baseline — прежний **K10/15**; MLP CE+JS и PAIR-weighted JS остаются отдельными exploratory альтернативами. Новые эксперименты в этой линии не планируются. Фиксация рабочего baseline не означает разрешение открыть test или устранение известных ограничений.

Текущий контроль для дальнейшей работы: `varvara/anza_lira_local_results/ct28_copy_count_ablations/joint_check/fusion_10_relation_15/` (далее **K10/15**). Он выбран как баланс точных repair, нулевых ложных принятых repair и сохранения JUNCTION/BOTH. Отдельного второго «основного» варианта нет. Краткий машинный итог и копии validation predictions/metrics находятся в `selected_validation/`; полный model pack и train scores остаются в локальной игнорируемой Git папке `varvara/`.

В [README CCTA3D](ccta3d/README.md) отдельно приведены выбор связей в 368 готовых validation-сценах и восстановление масок на четырёх окнах train-пациента 953 и 22 окнах пяти development-validation пациентов.

| Уровень | Зафиксированная версия K10/15 |
|---|---|
| PAIR/JUNCTION geometry local | HGB; по 5 position-only train-копий на исходный candidate, clean вес 0.5, каждая копия 0.1. |
| PAIR/JUNCTION CT local | HGB на clean train-кандидатах. |
| PAIR geometry+CT для candidate identity | HGB; 10 joint train-копий, clean/копии суммарно 0.5/0.5. |
| JUNCTION candidate identity | `geometry_score` от position-robust JUNCTION HGB; отдельный fusion score не выбирает JUNCTION. |
| Relation type | HGB на 84 признаках из раздельных PAIR/JUNCTION geometry и CT score streams; clean train-сцена + 15 joint-копий с весами 0.5 и 0.5/15. |
| Решение | `NONE/PAIR/JUNCTION/BOTH`, `tau=0.85`, endpoint-disjoint compatibility, 15 оценочных возмущений `0.8 мм + 20°`, exact-set consistency `>=0.60`. |

**Основа таблицы:** только текущий K10/15; варианты здесь не сравниваются. Его validation-результат приведён ниже, замена модели ещё не фиксировалась по test.

Train scores для relation head получены patient-level OOF; validation scores — моделями, обученными только на 17 train-пациентах. На 368 validation-сценах (206 требуют repair) K10/15 даёт 155/206 структурно точных и 11/368 ложных до consistency; **после gate 52 точных, 0 ложных, 0 неполных**: 48 `pair_only`, 2 `junction_only`, 2 `mixed`. SHA256 выбранного `relation_models.joblib`: `1693b6e9b00fd74f2d443905351f534e6bdf745a5cdc00d0057823646aa09630`; SHA256 `clean_eval/graph_lira_predictions_val.csv`: `934711d564a56eb7cda9ef2d9aaa19de144cf723b47b5ca883dd0e77fbb5deaf`.

## Откуда берутся данные и решения

```text
ImageCAS-X centerlines + frozen CT28 split
→ controlled CT28 scenes и PAIR/JUNCTION candidates
→ geometry и проверенные CCTA-признаки
→ patient-level OOF local scores
→ 84-feature HGB relation head: NONE / PAIR / JUNCTION / BOTH
→ tau, выбор совместимых candidate IDs, consistency
→ repair / abstain
→ новый CT28 path stage только для принятых repair
```

Canonical scene protocol `ct28_scene_protocol_v1` создан заново: exact historical scene generator и его model weights не восстановлены. Сохранены документированные границы PAIR distance `<=12 мм`, JUNCTION span `<=16 мм`, четыре класса и endpoint-disjoint совместимость. Пять семейств сцен: `pair_only`, `junction_only`, `mixed`, `none_orphan_pair`, `none_incomplete_junction`. Семейства не равны классам: `mixed` имеет класс `BOTH`; оба `none_*` — `NONE`. Прямые count-признаки пока остаются в 84-feature head; диагностика shortcut и абляция без counts описаны в истории абляций.

| Canonical вход или старый baseline | Назначение |
|---|---|
| `scene_plan_train_val.csv`, `scene_endpoints_train_val.csv`, `scene_candidates_train_val.csv` | 1 404 сцены, 7 020 endpoints, 15 092 candidates; train/val и ground truth. |
| `scene_protocol.json` | Правила генерации, статистика, SHA256. |
| `scene_candidate_features_train_val.csv`, `scene_ct_alignment_train_val.csv`, `scene_candidate_feature_protocol.json` | Geometry/CT-признаки и проверка 22 CT по SHA/shape/spacing/affine. |
| Корневые `relation_*`, `graph_lira_*`, `perturbation_consistency_val.csv` | Ранний clean Logistic baseline, сохранённый для исторического сравнения; **это не текущий K10/15**. |
| `selected_validation/selected_result.json`, `graph_lira_predictions_val.csv`, `graph_lira_structural_metrics_val.csv` | Выбранный K10/15: проверяемый краткий итог и полные scene-level validation решения. |
| `ablations/local_hgb/` и `ablations/stress30_jitter1/` | Ранние HGB/Stress30 варианты; также не текущий K10/15. Остальные невыбранные outputs перенесены в локальный архив, описанный в `ablations/rejected/README.md`. |

**Основа таблицы:** это указатель на файлы, а не сравнение качества; текущий pack — `selected_validation/`, корневые метрики и ранние абляции его не заменяют.

PAIR использует 7 geometry + 38 CT-признаков; JUNCTION — 9 geometry + 152 CT-признака. У 17 train-candidates CT sampling частично выходит за объём; применяется train-fitted median imputation. На validation таких строк нет. Четыре локальных score streams идут во второй уровень раздельно; geometry+CT fusion применяется к выбору PAIR identity после relation head. Historical 42-признаковое представление одного потока включает распределение candidate scores и candidate counts; CT-conditioned head получает два таких потока, всего 84 признака.

Точнее, 42 признака одного потока состоят из 18 PAIR summary, 18 JUNCTION summary и 6 cross-stream. Для каждого типа summary включает top-1…top-7, разницы top-1/top-2, top-2/top-3, top-1/top-3, mean, std, median, p75, p90, minimum, candidate count и долю top score в сумме. Исторические 42 признака восстановлены, но веса historical head, split folds, exact scene assembly и historical consistency formula не восстановлены; CT28-модели обучены заново. Три сравниваемых relation heads: geometry-only (42 geometry), early-fusion geometry+CT (42 fusion) и CT-conditioned (84, geometry и CT раздельно). В выбранном K10/15 используется третий.

После relation head PAIR выбирает candidate с максимальным разрешённым identity score, JUNCTION — аналогично. При BOTH нельзя независимо брать два максимума, если они используют общий endpoint: перебираются endpoint-disjoint комбинации и выбирается совместимая с наибольшим общим score. Другие исторические global graph constraints не восстановлены. Если максимальная вероятность relation ниже `tau=0.85`, тип становится `NONE`; после выбора набора связь принимается только при exact-set consistency `>=0.60`.

Отдельный **candidate-level JUNCTION baseline**, предшествовавший сборке структурных сцен, находится в `artifacts/varvara/ct28_junction/` и не равен нынешним local models на 15 092 scene candidates. Его frozen train/val generator дал 143 degree-3 positives и 1 666 negatives при 100% candidate recall. `segment_label` использовался только для audit; decoys сортировались по geometry distance. Среди 36 validation junction sources было 28 `competitive` и 8 `isolated`. CT-представление одного кандидата — тензор `3×17×25`; 38 признаков на arm (`10` осевых, `24` кольцевых, `4` поддержки) симметрично агрегируются через mean/min/max/std: `38×4=152` CT-признака. Все 22 train/val CT прошли SHA/shape/spacing/affine. Ниже результаты **ранней** StandardScaler→LogisticRegression на собственных JUNCTION candidates; они не являются результатами текущего Graph-LIRA:

| JUNCTION baseline | AUROC по всем validation candidates | Top-1 среди 36 источников | Top-1 среди 28 competitive | Top-1 среди 9 geometry-ambiguous |
|---|---:|---:|---:|---:|
| Geometry (9) | 0.9949 | 35/36 | 27/28 | 8/9 |
| CT (152) | 0.9405 | 28/36 | 20/28 | 6/9 |
| Geometry+CT (161) | 0.9936 | 36/36 | 28/28 | 9/9 |

**Сравнение:** исходный candidate-level baseline — Geometry (9). По AUROC он немного выше (0.9949 против 0.9936), по Top-1 на validation лучший Geometry+CT (36/36, включая 28/28 competitive). Этот ранний результат **не заменял** local JUNCTION selector в текущем K10/15: там используется отдельная position-robust HGB geometry-модель на scene candidates.

Для раннего **candidate-level** baseline пороги выбирались **на validation**, с максимизацией recall при FPR `<=5%`: geometry `0.7567`, CT `0.8976`, geometry+CT `0.1916`. Поэтому validation recall/FPR/precision при этих порогах — показатели на данных, использованных для выбора порога, а не независимая проверка порогового правила. AUROC и Top-1 от этих порогов не зависят. Это **не** Graph-LIRA `tau=0.85`. Подробные метрики и split/group-счётчики находятся в `artifacts/varvara/ct28_junction/baseline_metrics.json`.

## Что измеряет каждый этап

| Этап | Вопрос | Основные метрики | Текущий статус |
|---|---|---|---|
| Проверка данных | Совмещены ли CT и разметка? | SHA, shape, spacing, affine | 22/22 train/val CT проверены. |
| Candidate generation | Попадает ли истинный вариант в pool? | Candidate recall/Recall@K | 100% для controlled train/val repair-сцен; отдельный candidate-level JUNCTION audit выше. |
| Candidate ranking | Истинная связь выше ложных? | AUROC, AUPRC, Top-1/Top-K | Есть для раннего JUNCTION baseline и для scene-local score ablations. |
| Local acceptance | Можно ли принять отдельный candidate? | Recall, FPR, precision при local threshold | Есть у раннего JUNCTION baseline; Graph-LIRA принимает relation на уровне сцены. |
| Relation type | Верен ли `NONE/PAIR/JUNCTION/BOTH`? | Accuracy, balanced accuracy, macro-F1 | Сохранены relation probabilities и validation metrics. |
| Graph-LIRA | Верен ли совместный набор связей? | Structural exact, false structural, incomplete | K10/15: 155/206 exact и 11/368 false до consistency. |
| Selective policy и perturbations | Какие решения безопасно принять и насколько они устойчивы? | Coverage, exact/false among accepted, type/identity consistency | K10/15: 52 точных / 0 ложных после gate. |
| Patient/anatomy audit | Результат устойчив между пациентами и подгруппами? | Per-patient, patient-cluster bootstrap, scene family, competitive/isolated | Сохранены для validation; пять пациентов ограничивают вывод. |
| Path construction | Достигает ли путь выбранных ветвей и следует ли нужной анатомии? | Topological success, поддержка CT, branch-specific proximity/length | Новый path v2: 52/52 топологически успешны; для 12 JUNCTION arms средняя близость к нужной дуге при 0.75 мм выросла с 0.651 до 0.870. |
| Финальная маска | Улучшились ли сегментация и связность? | Dice, clDice, проверка новых связей по эталону | Оценена в отдельных прогонах CCTA3D по повреждённым маскам с автоматически найденными окнами. |

**Основа таблицы:** это перечень этапов и их метрик, без сравнения моделей; текущие значения относятся к K10/15, ранний candidate-level baseline явно отмечен отдельно. Финального победителя по end-to-end сегментации пока нет.

Обучаемая часть текущего pipeline: шесть local candidate HGB (PAIR/JUNCTION × geometry/CT/geometry+CT) и CT-conditioned HGB relation head; geometry-only и early-fusion relation heads хранятся как baseline. Candidate generation, вычисление признаков, compatibility, фиксированные gates, repair/abstain и max–min path solver не обучаются. Ранний список из четырёх обучаемых частей относился к более простой версии и не описывает текущий K10/15 полностью.

## Где читать результаты

- `ablations/README.md` — последовательность версий, сравнение и единственный выбранный K10/15.
- `ablations/rejected/README.md` — отклонённые и только диагностические варианты; их числовые outputs сохранены отдельно от текущего решения.
- `consistency_diagnostics/README.md` — причины abstain и нестабильность JUNCTION/BOTH в выбранном K10/15.
- `path/README.md` — новый CT28 path protocol, train-подбор, validation-маршруты и ограничения анатомического аудита.

Код scene/evidence/relation/evaluation: `scripts/research/ccta_graph_lira_safe_repair/build_ct28_scene_plan.py`, `extract_ct28_scene_candidate_features.py`, `train_ct28_relation_head.py`, `run_ct28_graph_lira.py`. Path stage реализован отдельно. Для следующего шага сначала решается вопрос качества relation/identity на validation, затем фиксируется полный protocol и только после этого открывается test.

## Проверка перед test, 2026-11-02

Проверка выполнена на train/validation, без новых test-решений. В замороженном scene pack подтверждены 1 404 сцены, 15 092 candidates, правильные class labels, разделение пациентов и patient-level OOF. Все 22 CT train/val прошли SHA256, shape, spacing и affine. У 17 train candidates CT-сетка частично за пределами объёма, на validation таких случаев нет; imputer обучен только на train и даёт конечные признаки/scores. Машинный аудит: игнорируемый Git `varvara/anza_lira_local_results/ct28_pretest_audit/pretest_integrity.json`.

Проверка локального JUNCTION ranking на 65 competitive сценах × 15 повторов дала для текущей geometry-модели `721/975` правильных top-1 при шуме только координат и `691/975` при совместном шуме. Диагностическая подстановка исходного `junction_center_distance_max_mm` поднимала счёт до `742/975` и `730/975`, но требует недоступную при inference чистую сцену. Реальное удаление этого признака с переобучением снизило position top-1 до `709/975`; поэтому локальная модель K10/15 **не заменена**. Предыдущие проверки усреднения и переобучения JUNCTION fusion также не превзошли полный структурный результат K10/15.

У scene protocol остаётся риск shortcut: count-only lookup давал около `73.6%` точности на validation. Удаление восьми прямых count-входов из relation head снижало итог с `52/0` до `49/0` и убирало оба принятых `junction_only`. Дополнительная контрфактическая замена только этих count-входов на train-медианы меняла raw тип у `31/368` сцен, включая `15/36` JUNCTION; после `tau` — у `27/368`, включая `19/36` JUNCTION. Это **диагностика чувствительности**, не допустимый inference: полученные комбинации признаков не обязаны соответствовать реальной сцене. Полные 84 признака оставлены, отсутствие shortcut не доказано.

Для выбранного K10/15 из 36 `junction_only` верный raw тип получен у 31, после `tau` — у 23, а после consistency приняты только 2. Среди 15 joint-noise повторов исходно правильных post-`tau` сцен JUNCTION raw тип сохранялся в `293/345`; в `138/293` таких повторов вероятность того же типа падала ниже `tau`. У `mixed` соответствующие числа `269/360` и `178/269`. Поэтому главный оставшийся барьер — одновременно confidence/type stability и выбор JUNCTION identity; простое удаление counts или одного геометрического входа его не решило. Frozen `tau=0.85` и consistency `0.60` не менялись.

Path v2 выбран по train: 5% допуск к max–min bottleneck, кратчайший поддержанный CT-маршрут и исключение невыбранных scene endpoints. На validation для тех же 52 принятых K10/15 сцен все пути построены без лишних контактов; близость JUNCTION arms к правильной VTK-дуге при 0.75 мм стала `0.870` против `0.651` у v1, медианная ошибка общего узла `0.86` против `1.64 мм`. Одна из 34 train-сцен всё ещё не строится из-за перекрытия JUNCTION arms. Подробности и машинный протокол — в `path/README.md` и `path/selected_path_protocol.json`. Результаты path stage **не прибавляют** принятых структурных repair.

**Решение до test:** K10/15 остаётся единственным структурным validation-претендентом (`52/0`, включая `2` JUNCTION и `2` mixed); path v2 — выбранный train-параметрами вариант для него. Цель 80 принятых не достигнута. Нельзя заявлять готовую финальную маску или Dice/clDice: rasterization и проверка изменения segmentation ещё не реализованы. До однократного held-out test требуется явно принять ограничение scene shortcut и низкого JUNCTION coverage, а также зафиксировать полный scene/path protocol и область оценки; подбор новых параметров на test запрещён.

Рабочие скрипты для K10/15 и path stage остались в `scripts/research/ccta_graph_lira_safe_repair/`; девятнадцать скриптов отклонённых диагностик сохранены в `ablations/rejected/rejected_experiment_scripts.tar.gz`. Большие outputs этих диагностик — в локальном архиве `varvara/`; инструкция и побайтовый манифест находятся в [журнале отклонённых вариантов](ablations/rejected/README.md). Для повторного обучения K10/15 **с самого начала** некоторые ранние входы sweep также потребуется восстановить из архива. Архивирование не изменило сохранённые решения K10/15.


## Проверка согласованности файлов, 2026-11-02

Проверены 29 текущих/новых Python-модулей, ссылки отчётов, 309 архивных outputs и 19 архивных скриптов по SHA. На сохранённых validation-признаках все шесть local streams воспроизвелись с максимальной ошибкой около `1e-16`; 5 520 повторных решений восстановили прежние **155/206 exact, 11/368 false до gate и 52/0 после**. Переобучение и новые test-вычисления не выполнялись.

Исправлены падение выбора при отсутствии кандидатов предложенного типа и возможность молча использовать устаревший/неполный evaluation cache. Для трёх переиспользованных 2×2 caches записано происхождение и обновлены только source SHA после проверки эквивалентности моделей и score tables; predictions и числовые результаты не менялись. Команда воспроизведения теперь сохраняет параметры оценочного шума/seed/bootstrap и корректно экранирует пути. Для небольшого архива старых скриптов добавлено точечное исключение Git ignore.

Числовые evidence проверки — локальный `varvara/anza_lira_local_results/ct28_consistency_audit/`. Код разделён по ответственности; общие обучение, features, gates и path solver переиспользуются. Пять основных отчётов остаются раздельными по смыслу; `ARCHITECTURE_MAP.md` — только совместимый указатель из ТЗ. Известные научные ограничения (shortcut и низкое принятие JUNCTION/BOTH) этим аудитом не устранены.


Полный аудит исходной локальной `varvara/`: 1 695 файлов, 10.555 GiB, включая 99 model-файлов, 56 VTK и 28 NIfTI segmentation. Все форматы прочитаны; 28 CT gzip-members прошли CRC и сравнение с frozen SHA без получения новых test-предсказаний. Segmentation SHA/shape/spacing/affine совпали 28/28; все 471 special points совпали с VTK. Пересчитаны 443 проверки сохранённых candidate/relation/structural метрик и 28 строк copy-count сравнений; расхождений нет. Все 88 scene local-model packs воспроизвели scores с ошибкой не выше `1.11e-16`; дополнительные MLP, ensemble, factorized и ranking модели также проверены по сохранённым validation-решениям/ранжированию. Геометрия и длина маршрутов проверены во всех четырёх сохранённых JSONL bundles. Это проверки сохранённых моделей, без переобучения и нового held-out Graph-LIRA evaluation.

В metadata attachment-пилота исправлен унаследованный SHA старого path solver: фактически использованный solver и baseline solver теперь указаны отдельно; старый protocol сохранён в audit evidence. Числовые features/results не менялись. Для двух ранних экспериментов SHA исходника относится к прежней версии скрипта; сохранённые веса и inference воспроизводятся, но побайтовое повторение старого обучения по сегодняшнему коду этим не доказано. Исторические SHA не подменялись.

При повторном thresholding сохранённых scores нужен `pd.read_csv(..., float_precision="round_trip")` либо готовые `*_pred`: стандартный CSV-reader округляет один пограничный score ранней JUNCTION fusion Logistic и создаёт ложное расхождение `35→34 TP`. Модель и lossless-чтение подтверждают исходные 35 TP. NPZ arm summaries хранятся в float32 и также не подменяют frozen feature CSV при точном воспроизведении.

Две скачанные ZIP-обёртки (`801-1000.z04.zip`, `801-1000.change2zip.zip`) содержат побайтовые копии уже распакованных частей и занимают дополнительно около 5 GB; не удалялись. Остальные совпадающие файлы преимущественно повторно используемые артефакты отдельных экспериментов. Полные file inventory, hashes и результаты проверок — в локальном `varvara/anza_lira_local_results/ct28_consistency_audit/full_local_inventory.json` и соседних audit JSON. Научные ограничения K10/15 остаются прежними.
