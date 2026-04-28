# GlobalScaleRoad: визуализация преимуществ (с авто-фокусом)

Этот файл сделан специально для проблемы «не видно преимущества на общей картинке».

## Что изменено

- Для каждого тайла автоматически ищется локальное окно, где AZ выигрывает у baseline.
- В figure показывается полный тайл + zoom именно этого окна.
- В zoom даны отдельные карты ошибок baseline/AZ и advantage-map с цветовой легендой и счётчиками.

## Лучшие кейсы для вставки в статью

| idx | full Dice base | full Dice AZ | zoom Dice base | zoom Dice AZ | Δzoom Dice | figure |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 0.0000 | 0.4213 | 0.0000 | 0.5698 | +0.5698 | `C:/Users/Comp1/SASHA/anza_lira/results/a3_final_package/final_article3/figures/spacenet_v3_advantage/case_003_advantage.png` |
| 5 | 0.2959 | 0.4336 | 0.3069 | 0.4824 | +0.1755 | `C:/Users/Comp1/SASHA/anza_lira/results/a3_final_package/final_article3/figures/spacenet_v3_advantage/case_005_advantage.png` |
| 7 | 0.6758 | 0.6972 | 0.6820 | 0.7472 | +0.0652 | `C:/Users/Comp1/SASHA/anza_lira/results/a3_final_package/final_article3/figures/spacenet_v3_advantage/case_007_advantage.png` |

Рекомендация: в paper оставить 1-2 такие advantage-focused фигуры вместо «общего» road-тайла.

Дополнительно доступны упрощённые figure формата `Input | Baseline | AZ | AZ vs Baseline`:
- `results/article_visual_assets/global_roads_advantage_v3/simple_compare_best1.png`
- `results/article_visual_assets/global_roads_advantage_v3/simple_compare_best2.png`

