# Статья 3: финальный пакет

Дата: `2026-04-27`

## 1) Принятое правило

В статье 3 не используем `GlobalScaleRoad` как основную метрику.

## 2) Основной датасет (GIS)

### Roads_HF

Источник:
- `results/gis_small_recovery/gis_roads_tuned_precision_ms_414243_e25/gis_roads_multiseed_summary.md`

| Датасет | Протокол | Baseline Dice | AZ-Thesis Dice | Delta Dice | Baseline IoU | AZ-Thesis IoU | Delta IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| Roads_HF | seeds `41/42/43`, `e25` | 0.9702 | 0.9705 | **+0.0004** | 0.9421 | 0.9428 | **+0.0007** |

Дополнительно:
- `clDice`: `+0.0077`.

## 3) Второй датасет для статьи 3

### Рекомендуемый вариант: FIVES

Источник итоговых чисел:
- `results/article2_final_metrics_pack_ru.md`

| Датасет | Протокол | Baseline Dice | AZ-Thesis Dice | Delta Dice | Baseline IoU | AZ-Thesis IoU | Delta IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| FIVES | `seed=42`, `e20` | 0.7677 | 0.8001 | **+0.0324** | 0.6230 | 0.6668 | **+0.0438** |

## 4) Фигуры

### Roads_HF
- `results/article_visual_assets/gis_roads_advantage_v1/simple_compare_best1.png`
- `results/article_visual_assets/gis_roads_advantage_v1/simple_compare_best2.png`

### FIVES
- `results/article_visual_assets/fives_article2_v1/simple_compare_1_000_100_D.png`
- `results/article_visual_assets/fives_article2_v1/simple_compare_2_001_101_G.png`

## 5) Готовый абзац для текста статьи

`В статье 3 в качестве основного GIS-кейса используется Roads_HF (multi-seed 41/42/43), где AZ-Thesis показывает сопоставимое с baseline качество по Dice/IoU (0.9702 -> 0.9705 и 0.9421 -> 0.9428) и улучшение структурной согласованности (clDice +0.0077). В качестве второго датасета используется FIVES, где зафиксирован более выраженный прирост относительно U-Net baseline (Dice 0.7677 -> 0.8001, IoU 0.6230 -> 0.6668).`

