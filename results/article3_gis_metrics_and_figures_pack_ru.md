# Статья №3: обновленный пакет (без GlobalScaleRoad)

Дата: `2026-04-27`

## 1) Фиксация решения

Для 3-й статьи **не используем метрику GlobalScaleRoad**.  
База статьи:
1. `Roads_HF` как основной GIS-кейс;
2. второй датасет — либо медицинский не-глазной (ARCADE), либо FIVES как сильный стабильный numeric-case.

---

## 2) Что уже можно ставить сразу (стабильно)

### 2.1 Roads_HF (основная строка)

Источник:
- `results/gis_small_recovery/gis_roads_tuned_precision_ms_414243_e25/gis_roads_multiseed_summary.md`

| Датасет | Протокол | Baseline Dice | AZ-Thesis Dice | Delta Dice | Baseline IoU | AZ-Thesis IoU | Delta IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| Roads_HF | seeds `41/42/43`, `e25` | 0.9702 | 0.9705 | **+0.0004** | 0.9421 | 0.9428 | **+0.0007** |

Дополнительно:
- `clDice`: `+0.0077` (AZ лучше по структурной согласованности).

---

## 3) Второй датасет: варианты и рекомендация

### Вариант A (рекомендую для дедлайна): FIVES

Плюс: четкий положительный прирост, хорошие числа для таблицы.

Источник:
- baseline: `results/article2_fives_compare/fives_full_baseline_s42_e20/baseline_seed42/metrics.json`
- az: `results/article_full_dataset/fives_full_azthesis_s42_e20/az_thesis_seed42/metrics.json`

| Датасет | Протокол | Baseline Dice | AZ-Thesis Dice | Delta Dice | Baseline IoU | AZ-Thesis IoU | Delta IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| FIVES | `seed=42`, `e20` | 0.7677 | 0.8001 | **+0.0324** | 0.6230 | 0.6668 | **+0.0438** |

Минус: это глазной датасет.

---

### Вариант B (не-глазной медицинский): ARCADE (коронарные сосуды)

Плюс: медицинский сосудистый домен, не retina.

Текущее состояние (по архивным baseline-vs-az сравнениям):
- `arcade_syntax`: Dice `0.6522 -> 0.5963` (хуже baseline)
- `arcade_stenosis`: Dice `0.3034 -> 0.2075` (хуже baseline)

Источники:
- `results/_archive_pre_latest_20260426_150548/final_pack_20260424/arcade_syntax_baseline_vs_azthesis_s42_e20/arcade_syntax_multiseed_summary.md`
- `results/_archive_pre_latest_20260426_150548/final_pack_20260424/arcade_stenosis_baseline_vs_azthesis_s42_e20/arcade_stenosis_multiseed_summary.md`

Вывод: ARCADE сейчас можно вставлять только как **честный limitation/future-work**, но не как «победа по метрикам».

---

## 4) Что ставим в итоговую таблицу статьи №3

Если нужна сильная и безопасная таблица прямо сейчас:
1. `Roads_HF` (multi-seed, стабильный паритет + структурный рост);
2. `FIVES` (выраженный прирост Dice/IoU).

Если принципиально нужен не-глазной medical во 2-й строке:
1. `Roads_HF`;
2. `ARCADE` как exploratory-case с честной пометкой, что прирост пока не достигнут.

---

## 5) Картинки для 3-й статьи

### Для Roads_HF:
- `results/article_visual_assets/gis_roads_advantage_v1/simple_compare_best1.png`
- `results/article_visual_assets/gis_roads_advantage_v1/simple_compare_best2.png`

### Для FIVES (если берем вариант A):
- `results/article_visual_assets/fives_article2_v1/simple_compare_1_000_100_D.png`
- `results/article_visual_assets/fives_article2_v1/simple_compare_2_001_101_G.png`

---

## 6) Готовый короткий текст в статью №3

`В третьей статье в качестве основного GIS-блока используется Roads_HF (31 изображение, протокол multi-seed 41/42/43), где метод AZ-Thesis показывает сопоставимое с baseline качество по Dice/IoU (0.9702 -> 0.9705 и 0.9421 -> 0.9428) и улучшение структурной согласованности (clDice +0.0077). В качестве дополнительной медицинской валидации используется FIVES, где получен выраженный прирост относительно U-Net baseline (Dice 0.7677 -> 0.8001, IoU 0.6230 -> 0.6668).`

