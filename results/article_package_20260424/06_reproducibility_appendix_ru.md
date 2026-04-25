# Приложение: Воспроизводимость

## 1) Ключевые конфиги

- `configs/drive_benchmark.yaml`
- `configs/chase_db1_benchmark.yaml`
- `configs/fives_benchmark.yaml`
- `configs/arcade_syntax_benchmark.yaml`
- `configs/arcade_stenosis_benchmark.yaml`
- `configs/drive_az_thesis_final_candidate_recall.yaml` (final DRIVE candidate)

## 2) Основные скрипты

- `scripts/run_drive_multiseed.py`
- `scripts/run_az_thesis_sweep.py`
- `scripts/export_drive_article_assets.py`
- `scripts/export_arcade_article_assets.py`
- `scripts/export_geometry_attention_story.py`

## 3) Артефакты результатов для цитирования

Мультидатасетная таблица:
- `results/final_pack_20260424/*_multiseed_summary.md`

DRIVE final candidate:
- `results/quick_arch_fix_20260424/drive_final_candidate_recall_hm010_pos9_ms_414243_e20/drive_multiseed_summary.md`

Картинки:
- `article_assets/final_figures/*`
- `article_assets/exports_arcade_finalpack/*`
- `article_assets/exports_arcade_finalpack_stenosis/*`

## 4) Минимальный набор метрик для отчета

- Dice
- IoU
- Precision
- Recall
- Specificity
- Balanced Accuracy
- Selected threshold
- (опционально) latency / params / GMACs

## 5) Что фиксировать в подписи к эксперименту

- dataset + split protocol;
- seeds;
- variant/model name (`Baseline U-Net`, `Attention U-Net`, `Proposed AZ-based method`);
- threshold selection policy (`eval_threshold_sweep`, target metric);
- важные архитектурные override-параметры (`encoder_az_stages`, `encoder_block_mode`, `hybrid_mix_init`, `az_geometry_mode`).

## 6) Рекомендованная naming-конвенция в тексте

- не использовать внутренние имена типа `az_thesis` как основное название;
- использовать:
  - `Baseline U-Net`
  - `Attention U-Net`
  - `Proposed AZ-based method`

В техническом приложении можно добавить соответствие:
- `Proposed AZ-based method` = `variant=az_thesis` + конкретные overrides из конфигов.

