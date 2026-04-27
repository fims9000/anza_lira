# ANZA-LIRA

Единый рабочий репозиторий по сегментации протяжённых структур (сосуды/дороги) с акцентом на `AZ-Thesis` и сравнение с `U-Net baseline`.

## Что это за система

Проект состоит из:
- обучаемой модели сегментации (`train.py`);
- локального AZ-оператора (`models/azconv.py`);
- набора конфигов под датасеты и протоколы (`configs/`);
- скриптов запуска и сборки отчётов (`scripts/`);
- итоговых материалов для статей (`results/`).

## Быстрый запуск

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --config configs/fives_benchmark.yaml --variants baseline,az_thesis
```

## Навигация по статьям (актуально)

### Статья 2 (EDM) — основной фокус

- Статус сбора: [results/a2_status.md](results/a2_status.md)
- Единый мастер-пакет: [results/a2_bundle.md](results/a2_bundle.md)
- Материалы ревизии: [results/edm_article2_revision_materials_ru.md](results/edm_article2_revision_materials_ru.md)
- Таблица метрик: [results/article2_final_metrics_pack_ru.md](results/article2_final_metrics_pack_ru.md)
- Пакет фигур: [results/article2_figures_pack_ru.md](results/article2_figures_pack_ru.md)

### Статья 3

- Единый пакет метрик/фигур: [results/a3_bundle.md](results/a3_bundle.md)

## Структура проекта

- `models/` — архитектура и слои;
- `configs/` — протоколы запусков;
- `scripts/` — очереди и утилиты;
- `results/` — только актуальные материалы и выбранные результаты;
- `logs/` — логи долгих запусков;
- `tests/` — проверки корректности.

## Правило по метрикам

В таблицы статей попадают только те значения, у которых есть:
1. проверяемый источник (`metrics.json` или summary-файл),
2. явно указанный протокол (`seed`, `epochs`, split, threshold policy).

## Публичный репозиторий

`https://github.com/fims9000/anza_lira`

