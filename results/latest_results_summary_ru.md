# Последняя сводка результатов

Дата среза: 2026-04-25

## Проверка окружения

- Создано локальное окружение `.venv` на Python 3.11.
- Установлены зависимости из `requirements.txt`.
- Добавлены недостающие зависимости `Pillow` и `requests`.
- Тесты после исправления AZConv: `59 passed, 1 skipped`.

## Найденная implementation-ошибка

Полный запуск `configs/drive_smoke.yaml` сейчас невозможен без данных DRIVE: ожидается структура `data/DRIVE/training/images`, `data/DRIVE/training/1st_manual`, `data/DRIVE/training/mask`.

При синтетической проверке `AZConv2d` была найдена ошибка в обработке padding:

- при `compatibility_floor > 0` floor добавлялся также к padded-соседям за пределами изображения;
- эти несуществующие соседи попадали в нормализацию совместимостей;
- на постоянном изображении слой искусственно занижал значения на границах.

Диагностика до исправления:

| Floor | Center | Corner | Edge |
|---:|---:|---:|---:|
| 0.0 | 1.000000 | 1.000000 | 1.000000 |
| 0.0005 | 1.000000 | 0.970248 | 0.984007 |

После исправления padded-соседи маскируются, а `compatibility_floor` добавляется только к реальным пикселям окна:

| Floor | Center | Corner | Edge |
|---:|---:|---:|---:|
| 0.0 | 1.000000 | 1.000000 | 1.000000 |
| 0.0005 | 1.000000 | 1.000000 | 1.000000 |

## Текущее состояние метрик

## Подготовленные датасеты

По состоянию на 2026-04-25 локально подготовлены:

| Dataset | Split | Images | Labels | FOV/Annotations | Источник |
|---|---|---:|---:|---:|---|
| DRIVE | training | 20 | 20 | 20 | Dropbox-копия официальных `training.zip`/`test.zip` |
| DRIVE | test | 20 | нет public labels | 20 | Dropbox-копия официального `test.zip` |
| CHASE-DB1 | training | 14 | 14 | 14 | Hugging Face mirror `Zomba/CHASE_DB1-retinal-dataset` |
| CHASE-DB1 | test | 14 | 14 | 14 | Hugging Face mirror `Zomba/CHASE_DB1-retinal-dataset` |
| FIVES | training | 600 | 600 | 600 | Figshare article `19688169` |
| FIVES | test | 200 | 200 | 200 | Figshare article `19688169` |
| ARCADE Syntax | train/val/test | 1000/200/300 | COCO json | COCO json | Zenodo record `10390295` |
| ARCADE Stenosis | train/val/test | 1000/200/300 | COCO json | COCO json | Zenodo record `10390295` |

Загрузчики проекта успешно видят `drive`, `chase_db1`, `fives`, `arcade_syntax`, `arcade_stenosis`. Для official DRIVE test нет публичной `1st_manual`, поэтому локальные DRIVE smoke/debug-запуски используют `validation_fallback_missing_test_labels`.

### DRIVE, quick_arch_fix_20260425/drive_implfix_best_ms_414243_e20

| Variant | Seeds | Dice mean+-std | IoU mean+-std | Precision | Recall | Balanced Acc | Params | GMACs | Fwd batch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 41,42,43 | 0.7456 +- 0.0061 | 0.5944 +- 0.0077 | 0.7374 | 0.7540 | 0.8638 | 1.54M | 19.97 | 0.074s |
| az_thesis | 41,42,43 | 0.7403 +- 0.0084 | 0.5878 +- 0.0106 | 0.8139 | 0.6816 | 0.8330 | 2.20M | 27.07 | 0.222s |

Вывод: `az_thesis` почти догнал baseline по Dice, но пока ниже на `0.0052`, тяжелее по параметрам и примерно в 3 раза медленнее на forward batch.

### Финальный пакет 2026-04-24, baseline vs az_thesis

| Dataset | Baseline Dice | AZ-Thesis Dice | Delta Dice |
|---|---:|---:|---:|
| DRIVE | 0.7432 | 0.5949 | -0.1483 |
| CHASE-DB1 | 0.6725 | 0.6479 | -0.0246 |
| FIVES | 0.7502 | 0.7199 | -0.0302 |
| ARCADE Syntax | 0.6522 | 0.5963 | -0.0560 |
| ARCADE Stenosis | 0.3034 | 0.2075 | -0.0959 |

Вывод: в этом пакете `az_thesis` не обгоняет baseline ни на одном датасете.

### Transfer FIVES -> CHASE, retinal_transfer_fives16_continue_ms

| Stage | Variant | Seeds | Dice mean+-std | IoU mean+-std | Precision | Recall | Balanced Acc |
|---|---|---|---:|---:|---:|---:|---:|
| FIVES pretrain | az_thesis | 41,42,43 | 0.7332 +- 0.0098 | 0.5789 +- 0.0122 | 0.7547 | 0.7142 | 0.8451 |
| CHASE finetune | az_thesis | 41,42,43 | 0.7283 +- 0.0066 | 0.5728 +- 0.0081 | 0.6888 | 0.7747 | 0.8688 |

## Рабочий вывод

Сейчас безопасная научная позиция: baseline остается главным прикладным эталоном, а `az_thesis` стоит подавать как исследовательскую ветку с интерпретируемой геометрией. Самый перспективный след - implementation-fix с fuzzy temperature, compatibility floor и input residual, но его нужно закрепить полным multiseed-прогоном перед сильными заявлениями.
