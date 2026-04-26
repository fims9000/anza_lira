# Финальный пакет для отправки статьи

Дата: `2026-04-27`

## 1) Основные файлы (использовать в camera-ready)

1. Основной текст (обновленный):
   - `results/imsc2026_paper32_revised_camera_ready_ru.md`
2. Финальная таблица метрик:
   - `results/article_final_selected_results_ru.md`
3. Финальный чеклист перед загрузкой:
   - `results/camera_ready_final_checklist_ru.md`
4. GIS-блок (описание + multi-seed):
   - `results/gis_small_recovery/gis_article_ready_pack_ru.md`

## 2) Ключевые итоговые метрики (для быстрой вставки)

| Dataset | Baseline Dice | AZ-Thesis Dice | Delta Dice | Baseline IoU | AZ-Thesis IoU | Delta IoU |
|---|---:|---:|---:|---:|---:|---:|
| DRIVE | 0.7358 | 0.7434 | +0.0076 | 0.5820 | 0.5916 | +0.0096 |
| CHASE_DB1 | 0.6554 | 0.7142 | +0.0588 | 0.4874 | 0.5554 | +0.0680 |
| GIS (Roads_HF, seeds 41/42/43) | 0.9702 | 0.9705 | +0.0004 | 0.9421 | 0.9428 | +0.0007 |

## 3) Рекомендуемые изображения

- GIS (простой и чистый формат):
  - `results/article_visual_assets/gis_roads_advantage_v1/simple_compare_best1.png`
  - `results/article_visual_assets/gis_roads_advantage_v1/simple_compare_best2.png`
- Медицинский блок:
  - `results/article_visual_assets/chase_article_figures/figure2_chase_examples.png`
  - `results/article_visual_assets/chase_article_figures/figure3_chase_xai.png`

## 4) Что очищено

Промежуточные/устаревшие отчеты перенесены в архив:

- `results/_archive_pre_submission_20260427/`

Это сделано, чтобы в корне `results` остались только актуальные файлы для сборки camera-ready.
