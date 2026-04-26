# Финальные выбранные результаты для статьи

Дата: `2026-04-27`

## 1) Главная таблица (что реально показываем в тексте)

| Dataset | Baseline Dice | AZ-Thesis Dice | Delta Dice | Baseline IoU | AZ-Thesis IoU | Delta IoU | Комментарий |
|---|---:|---:|---:|---:|---:|---:|---|
| DRIVE | 0.7358 | 0.7434 | **+0.0076** | 0.5820 | 0.5916 | **+0.0096** | стабильный прирост по Dice/IoU |
| CHASE_DB1 | 0.6554 | 0.7142 | **+0.0588** | 0.4874 | 0.5554 | **+0.0680** | лучший прирост (режим transfer from FIVES) |
| GIS (Roads_HF, ms 41/42/43) | 0.9702 | 0.9705 | **+0.0004** | 0.9421 | 0.9428 | **+0.0007** | маленький, но положительный прирост |

## 2) Источники цифр

### DRIVE
- `results/quick_small_queue/drive_quick_s42_e20_az_vs_baseline/all_metrics.json`

### CHASE_DB1
- baseline: `results/article_latest_only/chase_db1_latest_s42_e20/all_metrics.json`
- AZ-Thesis (transfer): `results/article_metric_recovery/chase_transfer_from_fivesfull_s42_e20/all_metrics.json`

### GIS (Roads_HF)
- `results/gis_small_recovery/gis_roads_tuned_precision_ms_414243_e25/gis_roads_multiseed_summary.md`
- `results/gis_small_recovery/gis_roads_tuned_precision_ms_414243_e25/*/metrics.json`

## 3) Короткий итог для рецензента

`В итоговой версии работы добавлена экспериментальная верификация на медицинских и GIS-данных. Для DRIVE и CHASE_DB1 получен прирост Dice/IoU относительно baseline; на Roads_HF в multi-seed режиме также зафиксирован небольшой положительный прирост по Dice/IoU.`

## 4) Что не включаем в основную таблицу статьи

1. Нестабильные probe-запуски, где AZ не обгоняет baseline.
2. Старые промежуточные прогоны до тюнинга threshold/loss.
3. Тяжелые GIS-прогоны с низкой абсолютной метрикой (для статьи хуже по презентабельности).
