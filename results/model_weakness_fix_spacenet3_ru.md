# Аудит модели и практический фикс (перед запуском SpaceNet3)

Дата: `2026-04-27`

## Что нашли как слабое место

При просмотре `metrics.json` по последним AZ-прогонам видно, что в части запусков анизотропия частично схлопывается:

- `az_anisotropy_gap_min` падает до очень малых значений (`~1e-6 ... 1e-4`) в отдельных слоях;
- это особенно заметно в некоторых CHASE/FIVES transfer-прогонах и части GIS-прогонов;
- при слабом штрафе `reg_hyperbolicity` слой может уходить к почти изотропному поведению.

Ключевой симптом:
- геометрическая ветка формально включена, но вклад направленной геометрии в глубине сети нестабилен.

## Что применили как фикс

Для нового большого прогона (SpaceNet3 Paris) добавлен усиленный anti-collapse профиль:

1. `az_geometry_mode: local_hyperbolic`
2. `az_learn_directions: true`
3. `az_min_hyperbolicity: 0.20` (было мягче)
4. `reg_hyperbolicity: 0.01` (усилен штраф за деградацию геометрии)
5. `reg_anisotropy_gap: 0.001`
6. `az_normalize_mode: per_rule` (чтобы не размывать rule-специфику)
7. `hybrid_mix_init: 0.12`, `az_residual_init: 0.05` (стабильный residual-контур)

Конфиг:
- `configs/spacenet3_paris_azthesis_large.yaml`

## Новый большой датасет

Выбран новый датасет, который не использовался в финальной таблице:

- `SpaceNet3 AOI_3_Paris` (из tarball + geojson)
- подготовка в формат train/val/in-domain-test:
  - `data/SpaceNet3_prepared/GlobalScaleRoad/...`

Скрипт подготовки:
- `scripts/prepare_spacenet3_paris.py`

## Текущий запуск

Запущена очередь:
1. подготовка SpaceNet3 Paris;
2. затем обучение AZ-Thesis на подготовленном наборе.

Логи:
- `logs/spacenet3_paris_large/spacenet3_prepare_train_20260427_014430.out.log`
- `logs/spacenet3_paris_large/spacenet3_prepare_train_20260427_014430.err.log`
