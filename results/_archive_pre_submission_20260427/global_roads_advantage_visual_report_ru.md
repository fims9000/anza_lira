# GlobalScaleRoad: визуализация преимуществ (с авто-фокусом)

Этот файл сделан специально для проблемы «не видно преимущества на общей картинке».

## Что изменено

- Для каждого тайла автоматически ищется локальное окно, где AZ выигрывает у baseline.
- В figure показывается полный тайл + zoom именно этого окна.
- В zoom даны отдельные карты ошибок baseline/AZ и advantage-map с цветовой легендой и счётчиками.

## Лучшие кейсы для вставки в статью

| idx | full Dice base | full Dice AZ | zoom Dice base | zoom Dice AZ | Δzoom Dice | figure |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.3524 | 0.4009 | 0.3277 | 0.5703 | +0.2426 | `C:/Users/Comp1/SASHA/anza_lira/results/article_visual_assets/global_roads_advantage_v3/case_004_advantage.png` |
| 6 | 0.3303 | 0.3646 | 0.4100 | 0.5364 | +0.1264 | `C:/Users/Comp1/SASHA/anza_lira/results/article_visual_assets/global_roads_advantage_v3/case_006_advantage.png` |
| 1 | 0.3112 | 0.3873 | 0.3074 | 0.4414 | +0.1340 | `C:/Users/Comp1/SASHA/anza_lira/results/article_visual_assets/global_roads_advantage_v3/case_001_advantage.png` |

Рекомендация: в paper оставить 1-2 такие advantage-focused фигуры вместо «общего» road-тайла.

Дополнительно доступны упрощённые figure формата `Input | Baseline | AZ | AZ vs Baseline`:
- `results/article_visual_assets/global_roads_advantage_v3/simple_compare_best1.png`
- `results/article_visual_assets/global_roads_advantage_v3/simple_compare_best2.png`

