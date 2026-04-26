# Расширение медицинских retinal-датасетов

Дата: 2026-04-26.

## Что сделано

DRIVE и CHASE_DB1 нельзя честно "докачать" внутри самих датасетов: это фиксированные публичные наборы с малым числом размеченных изображений. Поэтому расширение делаем как внешний retinal-domain pretrain/fine-tune.

Добавлен и подготовлен HRF-Seg+:

| Датасет | Локальная папка | Train | Test | Назначение |
|---|---:|---:|---:|---|
| HRF-Seg+ | `data/HRF_SegPlus` | 30 | 15 | дополнительный retinal pretrain / sanity benchmark |

Источник: Zenodo, DOI `10.5281/zenodo.16744782`, лицензия CC BY 4.0.

## Текущая медицинская база

| Датасет | Train images | Test images | Комментарий |
|---|---:|---:|---|
| DRIVE | 20 | 20 без локальных manual labels | для локальных метрик используется validation fallback |
| CHASE_DB1 | 14 | 14 | маленький, метрики сильно зависят от split/seed |
| FIVES | 600 | 200 | основной большой retinal vessel датасет |
| HRF-Seg+ | 30 | 15 | внешний дополнительный retinal-domain набор |
| ARCADE syntax | 1000 | 300 | сосудистая сегментация в ангиографии |
| ARCADE stenosis | 1000 | 300 | сосудистая/стенозная ангиография |

## Практический вывод

Для статьи DRIVE и CHASE не надо искусственно раздувать. Корректная схема:

1. Основной большой retinal pretrain: FIVES.
2. Дополнительный domain pretrain/sanity: HRF-Seg+.
3. Fine-tune на CHASE_DB1 и DRIVE.
4. В таблице явно писать, что DRIVE local test labels отсутствуют, поэтому DRIVE-оценка является validation fallback, а CHASE/FIVES/HRF имеют локальные masks.

## Реализация

Добавлены:

- `scripts/fetch_hrf_segplus_from_zenodo.py`
- `configs/hrf_segplus_benchmark.yaml`
- поддержка `dataset: hrf_seg_plus` в `utils.py`
- тест даталоадера HRF-Seg+

Проверка:

```powershell
python -m pytest tests\test_drive_pipeline.py::test_hrf_segplus_dataloaders_build_and_batch -q
```

Результат: `1 passed`.
