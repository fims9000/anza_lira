# CCTA 3D runtime: запуск на настоящем объёме КТ (исследовательский RC)

**Дата:** 2026-11-02. Ветка: `feature/ccta-3d-runtime-2026-11-02`; PR #2 (stacked on RC1 PR #1). Ничего не записывается в `research/coronary-connectivity-repair`, пока Варвара не проверит и не объединит изменения.

## 0. Реальные границы результата

Теперь есть **исполняемый код** для

```text
CT (NIfTI, HU) + ПОВРЕЖДЁННАЯ ПРЕДСКАЗАННАЯ 3D-маска (NIfTI)
  -> скелет/концы из самой маски
  -> реальные 5-концевые локальные окна и PAIR/JUNCTION кандидаты
  -> локальные CT и geometry признаки, сохранённые модели K10/15
  -> relation NONE/PAIR/JUNCTION/BOTH
  -> tau=0.85, exact-set perturbation consistency 15 x (0.8 мм, 20 градусов), gate=0.60
  -> глобальное согласование использования endpoints
  -> 3D CT bottleneck+shortest paths v2 (slack 5%)
  -> транзакционный mask repair с топологическими запретами
  -> repaired.nii.gz + added_voxels.nii.gz + decisions.json
```

**Важно:** это предварительный full-volume runtime, **не** валидация модели K10/15 на реальных segmentation errors. Frozen CT28 модель обучалась на искусственных сценах из пяти endpoints и содержит опасный кандидатно-количественный shortcut; распределение автоматически построенных окон отличается. Ни Dice/clDice на настоящих 3D пациентах, ни clinical risk, ни вклад 3D ANZA пока не установлены. На всех реальных новых данных система может корректно abstain.

Код не содержит обученного 3D ANZA encoder. Он подключает оригинальный image-conditioned **HGB K10/15**, а не подменяет его эвристикой или произвольной новой моделью.

## 1. Подготовить Linux окружение

Работать из клона репозитория ANZA-LIRA в новой ветке:

```bash
git fetch origin
git switch feature/ccta-3d-runtime-2026-11-02
python3.11 -m venv .venv-ccta
source .venv-ccta/bin/activate
python -m pip install -r requirements-ccta-3d.txt
python -m pytest -q tests/test_ccta_3d_endpoints.py tests/test_ccta_3d_k10.py tests/test_ccta_3d_runtime.py tests/test_ccta_3d_cli.py tests/test_ccta_3d_train_gap_simulator.py tests/test_ccta_3d_evaluate.py tests/test_ccta_3d_export_training_ct.py
```

Новый `nibabel` нужен только для NIfTI. Если сериализованный `relation_models.joblib` нельзя открыть из-за sklearn/version mismatch, следует использовать версию sklearn из **оригинального окружения обучения Варвары**; не делать скрытого переобучения. Проверка SHA обязана пройти перед десериализацией.

## 2. Рекомендуемый быстрый запуск — теперь одной командой

Новый воспроизводимый пилот автоматически выполняет export CT, контролируемые train-only повреждения, candidate inspection, настоящий K10/15, 3D path и восстановление маски, затем постфактум Dice/clDice и manifest. При ошибке оставляет `pilot_failure.json`, не выдаёт её за успех.

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_train_pilot.py \
  --patient-id 953 \
  --mode k10 \
  --gaps 3 --gap-radius-mm 1.5 \
  --repair-radius-mm 0 --max-new-voxels 250
```

По умолчанию результаты будут в `varvara/anza_lira_local_results/ccta_3d_runtime_pilot/953_k10_g3_seed20261102/`. Там появятся `ct.nii.gz`, `broken_mask.nii.gz`, `candidates.json`, `decisions.json`, `repaired.nii.gz`, `added_voxels.nii.gz`, `posthoc_metrics.json`, `pilot_manifest.json`. Повторный запуск в ту же директорию запрещён, потому что эксперимент должен оставаться воспроизводимым.

Для первичной диагностики **без загрузки K10/15 и без ремонта**:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_train_pilot.py \
  --patient-id 953 --mode candidates --gaps 3 --gap-radius-mm 1.5
```

При сравнении результатов не смешивать pilot с неполностью наблюдаемыми естественными segmentation failures. Это controlled reference-mask corruption, а не клинический результат. Параметры 1.5 мм, 0 мм и 250 в приведённой команде — демонстрационные train-only значения, а не зафиксированные оптимальные настройки.

### Ручной пошаговый вариант

Если нужно разобрать ошибку отдельного этапа, команды ниже выполняют те же операции по шагам.

## 2. Первый запуск именно на данных CT28 train-пациента 953

Оригинальный архив CT28 намеренно игнорируется Git. Оригинальная полная сегментация используется **только для подготовки controlled damage и постфактум сравнения**, но **не для скоринга или прокладки маршрута**. Не используем validation/test для подбора правил восстановления.

```bash
mkdir -p varvara/anza_lira_local_results/ccta_3d_runtime_pilot/953
OUT=varvara/anza_lira_local_results/ccta_3d_runtime_pilot/953
REFERENCE=varvara/anza_lira_local_data/Varvara_CT28_junction_source_pack/segmentations/953.coronary.nii.gz
```

Экспортировать исходную 3D-КТ из уже использованного в проекте multipart ImageCAS-архива с проверкой SHA/affine/spacing. Архив ожидается в `varvara/anza_lira_local_data/imagecas_801_1000/`:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_export_training_ct.py \
  --scan-id 953 --output "$OUT/ct.nii.gz" --report "$OUT/ct_export.json"
```

Сгенерировать 3 топологически проверяемых train-only разрыва. При отказе генератора менять уровень сложности можно только на train, фиксируя параметры и причину:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_train_gap_simulator.py \
  --patient-id 953 --reference-mask "$REFERENCE" \
  --gaps 3 --radius-mm 2.0 --seed 20261102 \
  --output-broken-mask "$OUT/broken.nii.gz" --report "$OUT/gaps.json"
```

Первой **безопасной** командой проверить кандидаты из повреждённой маски — без чтения K10 weights и без repair:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_cli.py \
  --ct "$OUT/ct.nii.gz" --mask "$OUT/broken.nii.gz" \
  --patient-id 953 --mode candidates --min-component-voxels 12 \
  --output-mask "$OUT/preview_unchanged.nii.gz" --report "$OUT/candidates.json"
```

В JSON проверить `extraction`, `endpoints`, `scenes`, координаты LPS и количество допустимых PAIR/JUNCTION. Если найдено менее пяти endpoints, K10/15 не запускается на таких окнах: нужны другие **train-only** повреждения или новое обучение модели, не синтетическое добавление фиктивных концов.

## 3. Запустить сохранённый K10/15 на 3D CCTA

Точный файл весов на машине Варвары:

```text
varvara/anza_lira_local_results/ct28_copy_count_ablations/joint_check/fusion_10_relation_15/relation_models.joblib
```

Его замороженный SHA256: `1693b6e9b00fd74f2d443905351f534e6bdf745a5cdc00d0057823646aa09630`. Если файл отсутствует или его хэш отличается, runtime отказывается запускаться. Для запуска не используются архивные root-level модельные пакеты.

```bash
MODEL=varvara/anza_lira_local_results/ct28_copy_count_ablations/joint_check/fusion_10_relation_15/relation_models.joblib
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_cli.py \
  --ct "$OUT/ct.nii.gz" --mask "$OUT/broken.nii.gz" \
  --patient-id 953 --mode k10 --model-pack "$MODEL" \
  --radius-mm 0 --max-new-voxels 250 \
  --output-mask "$OUT/repaired.nii.gz" \
  --output-additions "$OUT/added_voxels.nii.gz" \
  --report "$OUT/decisions.json"
```

`radius-mm 0` означает **экспериментальный centerline-only repair**, без домысливания диаметра сосуда. Это не клинический выбор оптимального lumen radius. Размер добавляемого коридора и volume budget следует определить и зафиксировать по train, а не улучшать на validation/test.

Каждый отклонённый вариант логируется: `K10_abstain`, `globally_reused_endpoint`, `mask_gate:...`, `CT path solver:...`, ошибка CT sampling или модели. При отказе исходная маска остаётся неизменённой. Успешный запуск, вернувший 0 исправлений, является **валидным абстенированием**, а не доказательством провала/успеха клинической задачи.

Для просмотра в 3D Slicer загрузить:
1. `ct.nii.gz` как volume;
2. `broken.nii.gz` как segmentation;
3. `repaired.nii.gz` как вторую segmentation;
4. `added_voxels.nii.gz` как отдельный overlay с новыми вокселями.

Маски сохраняют исходную CT геометрию; код **не** делает скрытого resample.

## 4. Посчитать настоящие до/после метрики (GT только post hoc)

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_evaluate.py \
  --patient-id 953 --before "$OUT/broken.nii.gz" \
  --after "$OUT/repaired.nii.gz" --reference "$REFERENCE" \
  --output "$OUT/posthoc_metrics.json"
```

Здесь будут Dice, clDice, число компонент, добавленный объём в мм³ и доля добавленных вокселей вне GT. Это **контролируемое искусственное повреждение** исходной разметки, а не настоящий сегментационный pipeline, поэтому результаты **нельзя** переносить на естественные ошибки отдельной сегментационной модели.

## 5. Групповая оценка train-когорты с учётом отказов

Для последующей статьи нужен не один успешный пример, а сводка по пациентам, в которой ошибки запуска и отсутствие repair не исчезают из знаменателя. Добавлен единый train-only запуск для всех 17 CT28 пациентов:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_train_cohort.py --run
```

Результаты окажутся в `varvara/anza_lira_local_results/ccta_3d_runtime_pilot/train_cohort/`. Каждому пациенту выделяется независимая immutable папка `patient_<ID>`. Ошибки записываются в `pilot_failure.json` и попадают в агрегированный отчёт. Если хотя бы одна серия не завершена, команда завершается с ошибкой и статусом `partial_train_cohort`.

Для пробного контролируемого прогона только первых трёх train-пациентов:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_train_cohort.py \
  --run --patients 953 955 956 \
  --root varvara/anza_lira_local_results/ccta_3d_runtime_pilot/three_train_subjects
```

Сводка: `train_cohort_summary.json`, `train_cohort_patients.csv`, `train_cohort_results.md`. В каждой строке — число сцен/принятых ремонтов, Dice/clDice до и после, ложные добавленные воксели, причины abstain. Interval — patient-level descriptive bootstrap только по train; приёмка на реальных новых пациентах ещё требуется.

### Диагностика OOD относительно исходных контролируемых сцен

В `decisions.json` и `candidates.json` есть `candidate_count_shift`, сравнивающий количество PAIR/JUNCTION кандидатов каждого real-3D окна с **17 train-пациентами** старого CT28. Сцены с новыми парами количеств маркируются `out_of_train_count_support`. Это не доказывает ошибку модели, но обнаруживает один из документированных shortcut-факторов.

Опционально отключить repair для всех таких окон:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_cli.py \
  --ct ct.nii.gz --mask broken_mask.nii.gz \
  --patient-id external_001 --mode k10 --model-pack /path/to/relation_models.joblib \
  --strict-count-support --output-mask repaired.nii.gz --report decisions.json
```

Включение этого флага меняет **новый full-volume runtime**, а не старый K10/15 CT28 validation baseline. Строгость по cardinality не заменяет независимой оценки анатомии и надёжности на новых пациентах.

## 5. Контракт и готовность для исследования

| Проверка | Действие |
|---|---|
| Реальная CT + маска совпадают по affine и shape | Отказать при несовпадении; не делать auto-resample |
| Нет исходной маски, только volume | Нет обученного 3D сегментера в этом runtime; требуется сначала получить predicted mask |
| < 5 валидных концов | Вернуть unchanged mask и записать причины |
| K10 SHA / sklearn schema не совпадают | Отказ без использования модели |
| Relation/type confidence или perturbation consistency не проходят | Abstain |
| Путь пересекает посторонние сосудистые компоненты | Atomic reject |
| Конфликт по endpoint между окнами | Не использовать один endpoint дважды |
| Можно соединить два разных пропуска одного ствола | Разрешить, если endpoints различны, а актуальная mask topology остаётся корректной |
| Output уже есть | Не перезаписывать; новый эксперимент сохранять отдельно |
| CT28 test ID 954/958/972/973/980/984 | Запрет до заранее зафиксированного тестового протокола |

**После train:** проверить на patient-disjoint validation с фиксированными настройками; затем согласовать и заморозить всю систему для разового held-out теста. Отдельно нужны естественные segmentation failures, аудит бифуркаций, clinician-level false bridges, runtime и сравнения с baseline/no repair.

**Что остаётся для ANZA:** разработать и обучить 3D анизотропный локальный feature encoder, затем сравнить его с обычным compact 3D CNN и исходными radial CT features в **одном и том же** эксперименте. Сейчас доказательств выигрыша ANZA-оператора именно на этой задаче ещё нет.

## 6. Если что-то не работает

- `sha mismatch`: не тот `relation_models.joblib`; K10 нельзя заменять произвольной версией.
- `Too many endpoints` / `Too many candidate windows`: маска даёт слишком много коротких 3D-отростков. Нужно осмотреть сегментацию и train-only настройки skeleton/spur extraction; не увеличивать лимиты без оценки сложности.
- `scene_count=0`: в маске нет достаточно пяти концов либо нет близких разных исходных компонентов; никаких фиктивных endpoint не создаётся.
- `feature_or_model_error`: изучить `decisions.json`, версии sklearn и выход за CT FOV. Реальная 3D inference может оказаться out-of-distribution относительно контролируемых CT28 сцен.
- `mask_gate:unselected_component_contact`: новый маршрут касается другого сегмента; не обходить veto, а записать failure case.
- На CPU 3D skeletonization и path поиска могут быть ресурсоёмкими; измерить wall-clock и VRAM/RAM на train. GPU специально не требуется.

## 7. Какие результаты нам прислать после локального запуска

Файлы `ct_export.json`, `gaps.json`, `candidates.json`, `decisions.json`, `posthoc_metrics.json`; дополнительно визуализацию пары примеров (CT, broken, repaired, overlay). **Сам CT и исходную медицинскую маску в Git не добавлять.**

Это даст основание решать, что корректировать дальше: skeletonization/candidate recall, OOD K10 relation head, JUNCTION identity, global graph conflicts или path/mask safety — вместо неинформативного перебора новых моделей.