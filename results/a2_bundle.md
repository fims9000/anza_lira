# Статья 2: единый master-пакет

Дата: `2026-04-27`

## 1) Основные файлы

- Ревизия под рецензентов: `results/edm_article2_revision_materials_ru.md`
- Финальная таблица метрик: `results/article2_final_metrics_pack_ru.md`
- Фигуры и подписи: `results/article2_figures_pack_ru.md`

## 2) Таблица, которую вставляем в текст

| Датасет | Протокол | Baseline Dice | Proposed Dice | Delta Dice | Baseline IoU | Proposed IoU | Delta IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| DRIVE | internal protocol, `seed=42` | 0.7905 | 0.7977 | +0.0072 | 0.6536 | 0.6635 | +0.0099 |
| CHASE_DB1 | multi-seed `41/42/43`, flips-TTA | 0.7469 | 0.7596 | +0.0127 | n/a | n/a | n/a |
| FIVES | `seed=42`, `e20` | 0.7677 | 0.8001 | +0.0324 | 0.6230 | 0.6668 | +0.0438 |

## 3) Фигуры, которые используем

- `results/article_visual_assets/fives_article2_v1/simple_compare_1_000_100_D.png`
- `results/article_visual_assets/fives_article2_v1/simple_compare_2_001_101_G.png`
- `results/article_visual_assets/fives_article2_v1/legend_diff.png`

## 4) Что обязательно указать в тексте статьи

1. Baseline — это `standard U-Net` в том же протоколе.
2. Параметры обучения: epochs, optimizer, lr, batch size, loss weights, threshold policy, augmentation.
3. Ссылка на публичный код: `https://github.com/fims9000/anza_lira`.
4. Отдельно отметить, что comparison с крупными Transformer-моделями — future work (если не успеваем полный benchmark).

## 5) Контроль перед отправкой

- Для каждой цифры есть путь к исходному `metrics.json`/summary.
- Для каждого рисунка есть короткая подпись и легенда.
- В письме рецензентам есть ответ пункт-в-пункт.

