# Q1-upgrade checklist для Article 3 (практический план)

Дата: `2026-04-28`

## 1) Scientific claim (фиксируем корректно)

Главный claim:

`AZ-механизм лучше сохраняет связность тонких протяжённых структур в низкоконтрастных и локально неоднозначных зонах, чем matched baseline.`

Не использовать формулировку:

`лучше везде по всем метрикам`.

Обязательные метрики для claim:
- `clDice`
- `skeleton recall`
- `fragmentation` / число компонент
- `Dice`, `IoU` (как контроль overlap)

## 2) Сильные сравнения (минимум для Q1)

Сравнивать при одном и том же протоколе:
- U-Net (обязательно),
- U-Net++,
- Attention U-Net,
- nnU-Net (или эквивалент strong baseline),
- минимум один transformer (SegFormer/UNETR).

Единый протокол:
- одинаковые split,
- одинаковые augment,
- одинаковая threshold policy (val sweep),
- `3-5` seeds,
- таблица `mean +- std`,
- статистическая проверка (например paired bootstrap / Wilcoxon по test images).

## 3) Геометрия должна быть доказуема (и уже частично реализована)

В коде сейчас:
- `anisotropy_gap`,
- `rule_usage_entropy_norm`,
- `metric_condition`,
- `direction_diversity` (добавлено),
- `AZ/Conv ratio` через `mix_alpha` (добавлено),
- model-native direction (`raw theta_map`) в figure.

Файлы:
- `train.py` -> `collect_architecture_state`
- `scripts/eval_direction_diversity.py`
- `scripts/export_geometry_clean_article_figure.py`

## 4) Ablation, которую сложно пробить рецензией

Нужный минимальный набор:
1. baseline
2. baseline + topology loss
3. AZ w/o fuzzy
4. AZ w/o anisotropy
5. AZ full (learnable geometry)

Итоговая таблица:
- overlap + structure metrics,
- direction-diversity diagnostics,
- runtime/memory.

## 5) Reproducibility (чтобы “все юзали”)

Нужно довести до:
- одна команда train,
- одна команда eval/table,
- одна команда figures,
- фиксированные split-списки,
- release checkpoint + configs.

## 6) Product-level блок

Показать:
- latency / throughput,
- VRAM usage,
- model size,
- стабильность при смене датасета.

Плюс:
- Docker/Colab,
- краткий demo-pipeline.

## 7) Где AZ-механизм сильнее всего

Лучшие классы задач:
- тонкие, протяжённые, ветвящиеся структуры,
- где важнее связность, чем только overlap.

Топ-домены:
- ретинальные сосуды,
- коронарные/церебральные сосуды (CTA/MRA),
- дороги на спутнике,
- трещины/дефекты линий,
- речные/канальные сети,
- нейриты/филаменты.

Обычно слабее:
- широкие blob-объекты без явной направленности.

## 8) Что уже исправлено сейчас

1. Direction panel в статье строится из raw модели (`theta_map`) без ручной подгонки направления.
2. Добавлен отдельный run с `local_hyperbolic + learn_directions=true` для проверки неколлапсной геометрии.
3. Добавлен автоматический отчёт direction-diversity:
   - `results/article3_spacenet_sprint_v3_recover/direction_diversity_summary.json`
   - `results/article3_spacenet_v3_dirlearn_probe_s42_e10/direction_diversity_summary.json`
