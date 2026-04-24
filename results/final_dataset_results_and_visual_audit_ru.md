# Финальный аудит датасетов и визуализаций (2026-04-24)

## 1) Сводка по датасетам (что уже получено)

### CHASE DB1 (3 seeds, baseline vs transfer AZ)
- Источник: `results/chase_transfer_vs_baseline_summary.md`
- Raw Dice:
  - Transfer AZ-Thesis: `0.7283 ± 0.0066`
  - Baseline U-Net: `0.7228 ± 0.0224`
  - Дельта: `+0.0055`
- Flips-TTA Dice:
  - Transfer AZ-Thesis: `0.7596 ± 0.0055`
  - Baseline U-Net: `0.7469 ± 0.0260`
  - Дельта: `+0.0127`
- Вывод: на CHASE transfer-вариант стабильно лучше baseline по Dice и более стабилен по seed-разбросу.

### ARCADE Syntax (3 seeds, baseline vs az_thesis)
- Источник: `results/arcade_syntax_full_ms_414243_baseline_vs_azthesis/arcade_syntax_multiseed_summary.md`
- Baseline:
  - Dice `0.6711 ± 0.0062`
  - IoU `0.5050 ± 0.0071`
  - Balanced Acc `0.8815 ± 0.0081`
- AZ-Thesis:
  - Dice `0.6069 ± 0.0094`
  - IoU `0.4357 ± 0.0097`
  - Balanced Acc `0.8568 ± 0.0073`
- Дельта Dice (az_thesis - baseline): `-0.0642`
- Вывод: baseline явно сильнее.

### ARCADE Syntax: geometry/connectivity метрики
- Источник: `results/arcade_syntax_full_ms_414243_baseline_vs_azthesis/geometry_connectivity_summary.md`
- Baseline:
  - clDice `0.6143 ± 0.0174`
  - Geometry Connectivity `0.6271 ± 0.0149`
- AZ-Thesis:
  - clDice `0.5470 ± 0.0190`
  - Geometry Connectivity `0.5604 ± 0.0185`
- Вывод: по текущей реализации геометрическая связность тоже не в пользу AZ-Thesis.

### ARCADE Stenosis (seed42, 10 epochs, smoke-протокол)
- Baseline: `results/arcade_stenosis_e10_baseline_seed42/metrics.json`
  - Dice `0.1123`, IoU `0.0595`, Recall `0.8275`, Balanced Acc `0.8372`, threshold `0.4`
- AZ-Thesis: `results/arcade_stenosis_e10_az_thesis_seed42/metrics.json`
  - Dice `~0`, IoU `~0`, Precision `1.0`, Recall `~0`, Balanced Acc `0.5`, threshold `0.7`
- Вывод: вырожденный режим у AZ-Thesis (почти пустая маска).

### DRIVE (исторические бенчмарки в репозитории)
- Источник: `results/drive_real_comparison.md`
- На текущем наборе логов `az_cat` и baseline выше `az_thesis`.
- Вывод: для практической статьи центральным кандидатом сейчас выглядит не `az_thesis`.

---

## 2) Визуальный аудит (очень внимательно, по запросу рецензента)

## 2.1 Что хорошо уже работает
- Сравнение prediction-vs-baseline читается хорошо:
  - `article_assets/final_figures/figure2_drive_examples.png`
  - `article_assets/final_figures/figure2_chase_examples.png`
- Карта ошибок/улучшений понятна:
  - green = исправлено относительно baseline
  - red = ухудшено
  - cyan = снят baseline false positive
  - orange = добавлен новый false positive

## 2.2 Что было проблемой и что исправлено
- Проблема: в Geometry Contribution карты часто были почти «серыми», вклад геометрии визуально слабый.
- Что сделано:
  - обновлен `scripts/export_geometry_attention_story.py`
  - обновлен `scripts/export_arcade_article_assets.py`
  - добавлена робастная нормализация вклада (percentile scaling), чтобы оранжево/синяя структура реально проявлялась.
- Обновленные артефакты:
  - `article_assets/final_figures/figure_geometry_attention_story_arcade_syntax_110.png`
  - `article_assets/final_figures/figure_geometry_attention_story_arcade_stenosis_108.png`
  - `article_assets/exports_arcade_cpu_check/baseline_seed42_vs_az_thesis_seed42/...`

## 2.3 Что все еще не дожато
- Direction field по-прежнему местами слишком «равномерный» и не всегда показывает убедимую локальную смену направления.
- На `arcade_stenosis` улучшение-карта может вводить в заблуждение:
  - там AZ даёт почти пустую маску, и часть «улучшений» объясняется не лучшей геометрией, а коллапсом предсказания.
- Поэтому для статьи нельзя опираться только на красивую XAI-карту: она должна идти вместе с количественными метриками.

---

## 3) Что это значит для статьи прямо сейчас

### Честный статус
- Утверждение «AZ-архитектура в целом лучше baseline на всех задачах» сейчас данными не подтверждается.
- Подтверждается более узкий и честный тезис:
  - на CHASE transfer-сценарии есть устойчивый выигрыш;
  - на ARCADE текущая версия `az_thesis` требует доработки.

### Как безопасно формулировать в тексте
- Не писать «state-of-the-art improvement across datasets».
- Писать:
  - «метод демонстрирует конкурентность и локальные улучшения качества/интерпретируемости; эффект зависит от домена и требует настройки под задачу».

---

## 4) Следующий практический шаг (приоритет)

1. Зафиксировать финальный основной результат статьи на CHASE (где выигрыш уже есть).
2. ARCADE оставить как честную внешнюю проверку переносимости + limitation/ongoing optimization.
3. Для геометрической части добавить 1 количественный XAI-метрик:
   - среднее `|log(sigma_u/sigma_s)|` на vessel vs background
   - и/или along-vessel vs cross-vessel response ratio.
4. В финальные фигуры брать только те кейсы, где improvement map согласуется с Dice/IoU.

---

## 5) Где лежат финальные артефакты для сборки статьи

- Сводки метрик:
  - `results/chase_transfer_vs_baseline_summary.md`
  - `results/arcade_syntax_full_ms_414243_baseline_vs_azthesis/arcade_syntax_multiseed_summary.md`
  - `results/arcade_syntax_full_ms_414243_baseline_vs_azthesis/geometry_connectivity_summary.md`
  - `results/arcade_stenosis_e10_baseline_seed42/metrics.json`
  - `results/arcade_stenosis_e10_az_thesis_seed42/metrics.json`
- Финальные фигуры:
  - `article_assets/final_figures/figure1_pipeline.svg`
  - `article_assets/final_figures/figure2_drive_examples.png`
  - `article_assets/final_figures/figure2_chase_examples.png`
  - `article_assets/final_figures/figure_geometry_attention_story_arcade_syntax_110.png`
  - `article_assets/final_figures/figure_geometry_attention_story_arcade_stenosis_108.png`
