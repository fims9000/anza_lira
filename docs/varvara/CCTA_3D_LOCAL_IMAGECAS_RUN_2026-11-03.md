# CT28 / ANZA3D — локальный запуск исследования на ImageCAS

**Версия:** 2026-11-03. Цель — получить ПЕРВЫЕ РЕАЛЬНЫЕ масочные результаты для 3D CNN и AZConv3d, а не ещё один синтетический CI. Эта инструкция проверяет только 17 train + 5 development-val пациентов. **Шесть CT28 held-out test ID не открывать.**

Команды ниже описывают исходный CUDA-протокол. Локальный запуск завершён на Mac с обучением CNN и ANZA на MPS и оценкой development-validation на CPU. Фактические результаты и отличия от инструкции собраны в [README CCTA3D](../../artifacts/varvara/ct28_graph_lira/ccta3d/README.md).

## 0. Что потребуется

Ветка: `feature/ccta3d-local-experiment-handoff-2026-11-03`.

На машине с CT28 данными должны существовать:

- `varvara/anza_lira_local_data/imagecas_801_1000/801-1000.z04` и `801-1000.zip`;
- `varvara/anza_lira_local_data/Varvara_CT28_junction_source_pack/segmentations/<scan_id>.coronary.nii.gz` для train и validation;
- `varvara/anza_lira_local_results/ct28_copy_count_ablations/joint_check/fusion_10_relation_15/relation_models.joblib`;
- выбранный frozen path v2 JSON (путь в `ccta_3d_cli.DEFAULT_PATH`);
- замороженные репозиторные CT28 CSV и alignment metadata;
- Python 3.11, зависимые numpy/pandas/nibabel/sklearn/scipy, PyTorch с CUDA, совместимый с GPU.

Не загружать медицинские CT/веса в GitHub Actions или в репозиторий. Эксперимент пишется в игнорируемый Git локальный каталог.

## 1. Проверка входов, read-only

```bash
git fetch origin
git switch feature/ccta3d-local-experiment-handoff-2026-11-03
source .venv-ccta/bin/activate

python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_preflight.py \
  --require-validation --min-free-gb 10

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Preflight выводит конкретные отсутствующие пути и проверяет SHA выбранного HGB-пакета. Это только проверка присутствия, не раскрывает и не модифицирует медицинские тома. При ошибке **сначала устранить причину**, не запускать обучение.

## 2. Пациентский OOF + ANZA/CNN + обученный 15-copy relation head

Рекомендованный основной исследовательский режим: `augmented_15` + `neural_pair`. Сохраняется в уникальном experiment root. В отличие от старого default `clean/frozen_fusion`, этот запуск проверяет вклад 3D ANN в **выбор PAIR**, а также CT relation head, обученный на 15 контролируемых train-возмущениях.

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_an_study.py \
  --stage all \
  --patient-id 953 \
  --device cuda \
  --epochs 8 --folds 5 --batch-size 2 \
  --relation-training augmented_15 \
  --pair-identity-source neural_pair \
  --root varvara/anza_lira_local_results/ccta3d_an_aug15_neuralpair_20261103
```

Стадии `--stage patches`, `--stage oof`, `--stage relations`, `--stage pilot`, `--stage volume` можно использовать отдельно, с **тем же** `--root`, если полный проход прервётся. `--batch-size 2` — консервативный старт для 16 GB GPU, а не подтверждённый оптимум. Код сохраняет SHA и запрещает заменять существующие результаты; при изменении seed/patch/weight протокола использовать новый каталог.

После train pilot должны существовать:

- `oof_pair_conv3d/oof_protocol.json`, `oof_pair_anza3d/oof_protocol.json`, аналогично JUNCTION;
- `relation_conv3d/relation_protocol.json` и `relation_anza3d/relation_protocol.json` со статусом `experimental_patient_OOF_15copy_augmented_relation_head`;
- `train_953_three_model_comparison.json`, а также `conv3d_repaired.nii.gz` и `anza3d_repaired.nii.gz`.

Наличие файлов **не гарантирует успех исследования**: нужно проверить количество принятых решений, причины abstain и сегментационные метрики. При отсутствии улучшения ANZA вывод не менять задним числом.

## 3. Замороженная development-validation — только после анализа train

Первым делом зафиксировать K10, path и параметры: создаётся отдельный неизменяемый JSON.

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_val_freeze.py \
  --output varvara/anza_lira_local_results/ccta_3d_validation/frozen_dev_val_20261103.json
```

Запустить K10/15 на пяти development-val пациентах и получить GT mask statistics:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_val_cohort.py \
  --freeze varvara/anza_lira_local_results/ccta_3d_validation/frozen_dev_val_20261103.json \
  --run
```

Затем **тот же режим `augmented_15 / neural_pair`**, явно указав подготовленный каталог ANN training:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ccta_3d_val_an_compare.py \
  --freeze varvara/anza_lira_local_results/ccta_3d_validation/frozen_dev_val_20261103.json \
  --study-root varvara/anza_lira_local_results/ccta3d_an_aug15_neuralpair_20261103 \
  --output-root varvara/anza_lira_local_results/ccta3d_val_aug15_neuralpair_20261103 \
  --relation-training augmented_15 \
  --pair-identity-source neural_pair \
  --run --device cuda
```

Это **описательная development-validation**, поскольку K10/15 раньше выбирался на этих пяти пациентах. Успех на этих данных не означает независимого подтверждения. Реальный blind test шестёрки пациентов открывать только после фиксации окончательных моделей и протокола.

## 4. Что собрать из результатов

Статус дополнительных проверок после основного запуска:

- [x] PAIR/JUNCTION top-1 по пациентам в OOF-протоколах и метки отдельных кандидатов в CSV. Принятые и ложные решения по готовым сценам в структурном отчёте.
- [x] Dice/clDice до и после, GT-inconsistent/unknown bridge pairs, доля добавленных вокселей внутри эталона и причины отказов по маскам.
- [x] Отдельная оценка трёх случаев из исторического FOV-аудита по скору, рангу и решению на validation с замороженными моделями.
- [ ] Сравнение CNN и ANZA на нескольких train-only seeds. Полностью оценён seed `20261102`. Дополнительное обучение остановлено по просьбе пользователя.
- [x] Разбор трёх наблюдавшихся ошибок с косыми срезами CT и эталонным контуром. Независимая клиническая проверка остаётся открытой.

[Результаты остальных пунктов плана](../../artifacts/varvara/ct28_graph_lira/ccta3d/README.md#остальные-пункты-плана)

Не смешивать два протокола: `clean` relation head и `augmented_15` должны иметь отдельные каталоги и manifest SHA. Файл `ccta_3d_val_an_compare.py` теперь проверяет, что train relation protocol в каждом принятом результате соответствует запрошенному, и отвергает смешение.

## 5. Статус выполнения

- [x] Код и тесты для real CT28 scene tables, CT-signal ablations и 3D patch FOV;
- [x] Patient OOF и 15-copy neural relation head, полностью реализованы в коде;
- [x] Явная PAIR identity `neural_pair` и strict protocol checks в val comparison;
- [x] Локальное обучение CNN/ANZA на ImageCAS CT и проверка consistency на контролируемых validation-сценах;
- [x] Пилот полного объёма train-пациента 953;
- [x] Сравнение полных масок на пяти development-validation пациентах;
- [ ] Natural segmentation failures и экспертная проверка;
- [ ] Однократный blind held-out test.
