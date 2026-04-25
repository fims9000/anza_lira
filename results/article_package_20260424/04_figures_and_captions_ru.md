# Фигуры, Фото И Подписи

## 1) Основные фигуры (готовые для статьи)

| Figure | Файл | Назначение | Готовая подпись |
|---|---|---|---|
| Fig.1 | `article_assets/final_figures/figure1_pipeline.svg` | Общая схема пайплайна | `Overall segmentation pipeline with geometry-aware local aggregation and binary mask prediction.` |
| Fig.2a | `article_assets/final_figures/figure2_drive_examples.png` | Сравнение на DRIVE | `Qualitative DRIVE examples: baseline vs proposed method on thin elongated structures.` |
| Fig.2b | `article_assets/final_figures/figure2_chase_examples.png` | Сравнение на CHASE | `Qualitative CHASE-DB1 examples with local error corrections in vessel branches.` |
| Fig.3a | `article_assets/final_figures/figure3_xai.png` | XAI/геометрия (DRIVE) | `Layer-wise geometry attention maps on DRIVE: direction, anisotropic contribution, and regime confidence.` |
| Fig.3b | `article_assets/final_figures/figure3_chase_xai.png` | XAI/геометрия (CHASE) | `Layer-wise geometry attention maps on CHASE-DB1 with boundary-level behavior.` |

## 2) Финальные геометрические “story” изображения

| Кейс | PNG | JSON (метаданные) | Комментарий |
|---|---|---|---|
| DRIVE final recall | `article_assets/final_figures/drive_final_recall_geometry_story/figure_geometry_attention_story_drive_03_test.png` | `article_assets/final_figures/drive_final_recall_geometry_story/figure_geometry_attention_story_drive_03_test.json` | лучший финальный DRIVE-кандидат, sample `03_test` |
| DRIVE quick probe | `article_assets/final_figures/quick_drive_localgeom_story/figure_geometry_attention_story_drive_15_test.png` | `article_assets/final_figures/quick_drive_localgeom_story/figure_geometry_attention_story_drive_15_test.json` | ранний этап фикса архитектуры |
| ARCADE syntax | `article_assets/final_figures/figure_geometry_attention_story_arcade_syntax_212.png` | `article_assets/final_figures/figure_geometry_attention_story_arcade_syntax_212.json` | sample `212` |
| ARCADE stenosis | `article_assets/final_figures/figure_geometry_attention_story_arcade_stenosis_289.png` | `article_assets/final_figures/figure_geometry_attention_story_arcade_stenosis_289.json` | sample `289` |

## 3) Дополнительные “фото-сетки” сравнений

ARCADE syntax:
- `article_assets/exports_arcade_finalpack/baseline_seed42_vs_az_thesis_seed42/013_110/article_grid.png`
- `article_assets/exports_arcade_finalpack/baseline_seed42_vs_az_thesis_seed42/110_199/article_grid.png`
- `article_assets/exports_arcade_finalpack/baseline_seed42_vs_az_thesis_seed42/205_284/article_grid.png`

ARCADE stenosis:
- `article_assets/exports_arcade_finalpack_stenosis/baseline_seed42_vs_az_thesis_seed42/013_110/article_grid.png`
- `article_assets/exports_arcade_finalpack_stenosis/baseline_seed42_vs_az_thesis_seed42/108_197/article_grid.png`
- `article_assets/exports_arcade_finalpack_stenosis/baseline_seed42_vs_az_thesis_seed42/205_284/article_grid.png`

CHASE transfer-пример:
- `article_assets/cache_exports/chase_az_thesis_from_fives16_continue_dice_ft20/012_Image_11L/article_grid.png`

## 4) Краткие русские подписи (если нужен RU-вариант)

- `Рис. 1. Полный пайплайн сегментации с геометрически-ориентированной локальной агрегацией и формированием бинарной маски.`
- `Рис. 2. Визуальное сравнение baseline и Proposed AZ-based method на сложных локальных участках (тонкие ветви, развилки).`
- `Рис. 3. Послойная геометрическая интерпретация: направление локальной агрегации, карта анизотропного вклада и уверенность выбранного режима.`

## 5) Как подписывать легенду геометрии

- `u-axis`: направление вытяжения локального ядра (`sigma_u`);
- `s-axis`: поперечное сжатие (`sigma_s`);
- `Contribution`: вклад анизотропной геометрии под маской вероятности сегментации;
- `Regime confidence`: уверенность доминирующего fuzzy-режима.

