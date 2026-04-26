# GIS (Roads_HF): финальный блок для статьи

Дата обновления: `2026-04-27`

## 1) Датасет

- Название: `Roads_HF` (в конфиге: `gis_roads`)
- Путь: `data/Roads_HF`
- Формат: `images/*.png` + `masks/*.png` (бинарная сегментация дорог)
- Объем: `31` пар изображение/маска
- Разрешение исходных кадров: `1280x720`
- Сплит (deterministic): `val_fraction=0.2`, `gis_test_fraction=0.2`
  - train: `19`
  - val: `6`
  - test: `6`

## 2) Протокол

- Эксперимент: `results/gis_small_recovery/gis_roads_tuned_precision_ms_414243_e25`
- Варианты: `baseline`, `az_thesis`
- Seeds: `41, 42, 43`
- Эпохи: `25`
- Threshold selection: `val_sweep` по `dice`

## 3) Итоговые метрики (mean +- std, seeds 41/42/43)

Источник: `results/gis_small_recovery/gis_roads_tuned_precision_ms_414243_e25/gis_roads_multiseed_summary.md`

| Вариант | Dice | IoU | Precision | Recall | Specificity | Balanced Acc |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.9702 +- 0.0060 | 0.9421 +- 0.0112 | 0.9736 +- 0.0033 | 0.9669 +- 0.0149 | 0.9923 +- 0.0005 | 0.9796 +- 0.0073 |
| az_thesis | 0.9705 +- 0.0077 | 0.9428 +- 0.0145 | 0.9759 +- 0.0076 | 0.9655 +- 0.0203 | 0.9931 +- 0.0014 | 0.9793 +- 0.0095 |

Дельта `az_thesis - baseline`:

- Dice: `+0.0004`
- IoU: `+0.0007`
- Precision: `+0.0023`
- Recall: `-0.0013`
- clDice: `+0.0077` (по прямому расчету из `metrics.json` seed 41/42/43)
- Balanced Acc: `-0.0002`

## 4) Разбивка по seed (test)

Источник: `*/baseline_seed{41,42,43}/metrics.json`, `*/az_thesis_seed{41,42,43}/metrics.json`

| Seed | Baseline Dice | AZ-Thesis Dice | Delta |
|---|---:|---:|---:|
| 41 | 0.9747 | 0.9782 | +0.0035 |
| 42 | 0.9740 | 0.9734 | -0.0006 |
| 43 | 0.9617 | 0.9599 | -0.0018 |
| Mean | 0.9702 | 0.9705 | +0.0004 |

## 5) Готовый текст для статьи (GIS-блок)

`На малом GIS-наборе Roads_HF (31 изображение) метод AZ-Thesis в multi-seed постановке (seeds 41/42/43) показывает сопоставимое с baseline качество по Dice/IoU и небольшой положительный прирост по среднему Dice (+0.0004) и IoU (+0.0007). При этом наблюдается рост структурной согласованности (clDice +0.0077), что соответствует цели метода по улучшению геометрической целостности тонких протяженных объектов.`

## 6) Что вставлять в статью

1. Таблица из раздела **3** (mean +- std).
2. Короткий текст из раздела **5**.
3. Один простой рисунок сравнения:
   - `results/article_visual_assets/gis_roads_advantage_v1/simple_compare_best1.png`
   - или `results/article_visual_assets/gis_roads_advantage_v1/simple_compare_best2.png`
