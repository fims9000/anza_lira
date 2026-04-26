# Submission 32: закрытие замечаний по визуализации (дополнение)

Дата: 2026-04-26

## Что именно было исправлено

1. Старый road-рисунок заменен на интерпретируемый формат `2 x 4` с явной легендой цветов прямо на изображении.
2. Добавлен не только положительный пример, но и failure case, чтобы показать ограничения метода честно.
3. Для выбранных тайлов добавлены численные расшифровки (Dice/IoU/Precision/Recall + связность), чтобы визуализация не выглядела субъективной.
4. Проведен аудит по всем 30 тайлам из отбора для figure-пайплайна.

## Новые файлы визуализаций

- Positive case (full + zoom): `results/article_visual_assets/global_roads_figures_v2/positive_gain_zoom_grid.png`
- Failure case (full + zoom): `results/article_visual_assets/global_roads_figures_v2/failure_loss_zoom_grid.png`
- Advantage-focused case A (auto-focus): `results/article_visual_assets/global_roads_advantage_v3/top1_advantage_case_idx_004.png`
- Advantage-focused case B (auto-focus): `results/article_visual_assets/global_roads_advantage_v3/top2_advantage_case_idx_006.png`

## Легенда (вшита в новые PNG)

- `Error map`: TP=green, FP=red, FN=blue, TN=black.
- `Difference map`: AZ-better=green, Base-better=red, both-road-correct=white, both-wrong=black, both-background-correct=dark-gray.
- `Skeleton improvement map`: AZ-recovers=green, Base-recovers=red, both-hit=white, both-miss=blue.

## Численная проверка по текущему выбранному тайлу (idx=1)

- Dice: baseline `0.3112` -> AZ `0.3873` (Δ `+0.0762`)
- IoU: baseline `0.1842` -> AZ `0.2402` (Δ `+0.0560`)
- Precision: baseline `0.2007` -> AZ `0.2804`
- Recall: baseline `0.6921` -> AZ `0.6262`
- Components: baseline `294` vs AZ `383`
- Largest CC ratio: baseline `0.8359` vs AZ `0.5454`

## Аудит по всем 30 тайлам

- AZ лучше baseline по Dice: `20 / 30`
- baseline лучше AZ по Dice: `10 / 30`
- Средний Dice: baseline `0.3666` -> AZ `0.3742` (Δ `+0.0077`)

Полный детальный аудит:

- `results/global_roads_visualization_audit_v2_ru.md`
- `results/global_roads_advantage_visual_report_ru.md`
- `results/article_visual_assets/global_roads_figures_v2/global_roads_tile_metrics_full.csv`
- `results/article_visual_assets/global_roads_figures_v2/global_roads_tile_metrics_full.json`

## Что вставлять в статью

1. В основной текст: positive case + failure case рядом в одинаковом формате.
2. В подписи: явно указать легенду цветов и конкретные цифры по выбранным тайлам.
3. В обсуждении ограничений: сослаться на failure case и аудит по 30 тайлам.
