# Full-Dataset Article Runs

Дата: 2026-04-26.

## Почему раньше объектов было мало

Мало объектов было только в DRIVE: это классический маленький retinal benchmark. Локально есть 20 training images с manual vessel labels и 20 test images без manual vessel labels, поэтому свежие DRIVE-прогоны считаются на deterministic validation fallback.

Для финальной статьи DRIVE нельзя оставлять единственной опорой. Поэтому запускается full-dataset очередь по всем готовым датасетам.

## Реальные размеры локальных датасетов

| Dataset | Full local split/status |
|---|---|
| DRIVE | 20 train with labels; 20 test images without local manual labels |
| CHASE_DB1 | 14 train / 14 test with labels |
| FIVES | 600 train / 200 test |
| ARCADE syntax | 1000 train / 200 val / 300 test |
| ARCADE stenosis | 1000 train / 200 val / 300 test |
| GlobalScaleRoad | 3338 train / 339 val / 624 in-domain test |
| Roads_HF | 31 pairs; too small for main claim |
| SpaceNet3_Roads | downloaded raw/vector resources are not yet prepared as raster masks |

## Queue

Логи:

- `logs/article_full_dataset/01_chase_db1_full_azthesis_s42_e80.out.log`
- `logs/article_full_dataset/02_fives_full_azthesis_s42_e20.out.log`
- `logs/article_full_dataset/03_arcade_syntax_full_azthesis_s42_e40.out.log`
- `logs/article_full_dataset/04_arcade_stenosis_full_azthesis_s42_e40.out.log`
- `logs/article_full_dataset/05_global_roads_full256_azthesis_s42_e30.out.log`

Результаты:

- `results/article_full_dataset/chase_db1_full_azthesis_s42_e80`
- `results/article_full_dataset/fives_full_azthesis_s42_e20`
- `results/article_full_dataset/arcade_syntax_full_azthesis_s42_e40`
- `results/article_full_dataset/arcade_stenosis_full_azthesis_s42_e40`
- `results/article_full_dataset/global_roads_full256_azthesis_s42_e30`

## Итоговая таблица

| Dataset | Method | Train/eval scope | Status | Dice | IoU | clDice | Precision | Recall |
|---|---|---|---|---:|---:|---:|---:|---:|
| DRIVE | AZ-Thesis | 20 train, validation fallback | done | 0.7977 | 0.6635 | 0.7985 | 0.8010 | 0.7945 |
| CHASE_DB1 | AZ-Thesis | full 14/14 | done | 0.6691 | 0.5027 | 0.6785 | 0.5684 | 0.8130 |
| FIVES | AZ-Thesis | full 600/200 | running | | | | | |
| ARCADE syntax | AZ-Thesis | full 1000/200/300 | pending | | | | | |
| ARCADE stenosis | AZ-Thesis | full 1000/200/300 | pending | | | | | |
| GlobalScaleRoad | AZ-Thesis | full 3338/339/624 | pending | | | | | |

Обновлять таблицу надо после завершения каждого шага очереди.

## Текущий анализ

- DRIVE малый по природе датасета, поэтому в статье он должен быть classic benchmark, а не единственная основа.
- CHASE_DB1 завершен на полном локальном split: 14 training / 14 test. Test Dice ниже validation Dice, значит validation subset внутри training images слишком оптимистичен.
- CHASE_DB1 full-from-scratch результат `0.6691 Dice` лучше текущего latest-only CHASE `0.6248 Dice`, но хуже старого transfer-result `0.7377 Dice`. Для статьи можно отдельно указать, что transfer/pretraining на FIVES повышает переносимость.
- FIVES сейчас обучается на полном 600/200 split. Один epoch занимает примерно 135 секунд, поэтому 20 эпох займут ориентировочно 45-50 минут.

## Small Medical Queue

Параллельно поставлена отдельная очередь для малых медицинских retinal-наборов. Она не стартует второй CUDA-тренинг, пока GPU занят большим FIVES/ARCADE/GIS прогоном, чтобы не получить OOM на RTX 5070 Ti 16 GB.

Лог очереди:

- `logs/article_small_medical/small_medical_launcher.out.log`

План очереди:

| Dataset | Method | Run | Status |
|---|---|---|---|
| HRF-Seg+ | AZ-Thesis | `hrf_segplus_azthesis_s42_e60` | waiting for GPU |
| CHASE_DB1 | AZ-Thesis | `chase_db1_azthesis_s42_e80` | pending |
| DRIVE | AZ-Thesis | `drive_azthesis_s42_e120` | pending |
