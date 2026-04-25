# Пакет Для Написания Статьи (2026-04-24)

Этот каталог собран как единый набор материалов для текста статьи:

- итоговые численные результаты и честные выводы;
- описание датасетов и протокола;
- описание архитектуры (Baseline / Attention U-Net / Proposed AZ-based method);
- список готовых рисунков с подписями;
- структура статьи по разделам;
- приложение по воспроизводимости.

## Состав

- `01_results_and_conclusions_ru.md`
- `02_datasets_description_ru.md`
- `03_architecture_description_ru.md`
- `04_figures_and_captions_ru.md`
- `05_article_structure_ru.md`
- `06_reproducibility_appendix_ru.md`

## Базовые источники внутри репозитория

- `results/final_pack_20260424/*`
- `results/quick_arch_fix_20260424/*`
- `article_assets/final_figures/*`
- `article_assets/exports_arcade_finalpack/*`
- `article_assets/exports_arcade_finalpack_stenosis/*`
- `utils.py`, `models/azconv.py`, `models/segmentation.py`


## UPDATE (2026-04-25)

Добавлена верификация `implementation-corrected AZ variant` на полном протоколе DRIVE (20 epochs, seeds 41/42/43).

Ключевой итог: этот вариант не подтвердил преимущество над baseline на полном multi-seed прогоне (`Dice vs baseline = -0.0052`), поэтому в пакет статьи он включен как отрицательный/нестабильный результат, а не как финальный лучший кандидат.

## UPDATE (2026-04-25, recalibration)

Для impl-fix пакета DRIVE добавлена post-hoc переоценка тех же чекпойнтов при `threshold=0.80` (без переобучения):
- `results/quick_arch_fix_20260425/drive_implfix_best_ms_414243_e20/rethreshold_thr080_summary.md`

Ключевой итог: при стабилизированном пороге impl-fix возвращает конкурентный multi-seed результат (`AZ Dice mean 0.7498` vs `baseline 0.7456`).
