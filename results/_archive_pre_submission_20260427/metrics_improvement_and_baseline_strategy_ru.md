# План улучшения метрик и baseline-сравнений

Дата: 2026-04-26

## Что у нас сейчас не так

1. В части прогонов сравнивали только `baseline` vs `az_thesis` (этого мало для рецензента).
2. Самый сильный рецепт по DRIVE сейчас не `az_thesis`, а `az_cat` (`Dice 0.7977` в текущих лучших сохраненных результатах).
3. В некоторых новых latest-only прогонах произошла регрессия из-за перехода на менее стабильный рецепт.

## Какие baseline нужны в статье

Минимум три:

1. `baseline` (U-Net)
2. `attention_unet` (сильный архитектурный baseline)
3. `baseline` в **loss-matched** протоколе (та же политика лоссов, что у AZ)

Плюс для разбора вклада метода:

4. `az_no_fuzzy`
5. `az_no_aniso`
6. `az_cat` (наша рабочая версия AZ-Thesis в статье)

## Ключевая идея честного сравнения

Сравниваем модели при одной и той же тренировочной политике:

- одинаковые `model_widths`
- одинаковые веса лоссов (`aux`, `boundary`, `topology`)
- одинаковые правила подбора threshold
- одинаковые эпохи и seed

Для этого добавлен файл:

- `configs/reviewer_drive_lossmatched_overrides.yaml`

## Что запускать первым (приоритет)

1. **DRIVE full fair baseline pack**  
   Это главный набор для рецензента по вопросу “а почему не сравнили с нормальными baseline”.

2. **CHASE transfer recovery**  
   Уже запущена отдельная очередь `run_metric_recovery_queue.ps1` (ожидает освобождения GPU), чтобы поднять CHASE-метрики через transfer от FIVES.

3. **GIS оставить как hard-case**  
   Не делать его headline-результатом; оставить как дополнительный сложный сценарий.

## Готовая очередь baseline-прогонов

Добавлен скрипт:

- `scripts/run_reviewer_baseline_pack_queue.ps1`

Что он делает:

1. `drive_probe_s42_e40_all_variants`:
   - `baseline, attention_unet, az_no_fuzzy, az_no_aniso, az_cat`
   - быстрый single-seed probe
2. `drive_final_ms_414243_e120_headline`:
   - `baseline, attention_unet, az_cat`
   - итоговый multi-seed прогон под таблицу

## Что показывать в итоговой таблице IEEE

**Main table (headline):**

- DRIVE: `baseline` vs `attention_unet` vs `az_cat`
- CHASE_DB1 transfer: `baseline` vs `az` (после recovery)
- FIVES: сильный абсолютный результат (и baseline, если успеваем парный run)

**Supplement:**

- `az_no_fuzzy`, `az_no_aniso` (абляция)
- GlobalScaleRoad и ARCADE как hard/failure-aware cases

## Команда запуска baseline-pack

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_reviewer_baseline_pack_queue.ps1
```
