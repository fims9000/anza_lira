# Результаты И Выводы Для Статьи

Дата фиксации пакета: **2026-04-24**.

## 1) Финальный мультидатасетный пакет (baseline vs az_thesis, seed 42)

Источник:
- `results/final_pack_20260424/drive_baseline_vs_azthesis_s42_e20/drive_multiseed_summary.md`
- `results/final_pack_20260424/chase_baseline_vs_azthesis_s42_e20/chase_db1_multiseed_summary.md`
- `results/final_pack_20260424/fives_baseline_vs_azthesis_s42_e10/fives_multiseed_summary.md`
- `results/final_pack_20260424/arcade_syntax_baseline_vs_azthesis_s42_e20/arcade_syntax_multiseed_summary.md`
- `results/final_pack_20260424/arcade_stenosis_baseline_vs_azthesis_s42_e20/arcade_stenosis_multiseed_summary.md`

| Dataset | Baseline Dice | AZ-Thesis Dice | Delta Dice | Baseline IoU | AZ-Thesis IoU | Delta IoU |
|---|---:|---:|---:|---:|---:|---:|
| DRIVE | 0.7432 | 0.5949 | -0.1483 | 0.5913 | 0.4234 | -0.1680 |
| CHASE-DB1 | 0.6725 | 0.6479 | -0.0246 | 0.5066 | 0.4792 | -0.0274 |
| FIVES (e10) | 0.7502 | 0.7199 | -0.0302 | 0.6002 | 0.5624 | -0.0378 |
| ARCADE Syntax | 0.6522 | 0.5963 | -0.0560 | 0.4839 | 0.4248 | -0.0592 |
| ARCADE Stenosis | 0.3034 | 0.2075 | -0.0959 | 0.1788 | 0.1158 | -0.0631 |

Промежуточный честный вывод по этому пакету: в этой серии `az_thesis` уступает baseline по всем пяти датасетам.

## 2) Контроль на DRIVE с дополнительным architectural fix (multi-seed 41/42/43)

Источник:
- `results/quick_arch_fix_20260424/drive_final_candidate_recall_hm010_pos9_ms_414243_e20/drive_multiseed_summary.md`

| Model | Dice mean +- std | IoU mean +- std | Precision mean +- std | Recall mean +- std | Balanced Acc mean +- std | Dice vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| Baseline U-Net | 0.7442 +- 0.0037 | 0.5926 +- 0.0047 | 0.7868 +- 0.0303 | 0.7083 +- 0.0313 | 0.8446 +- 0.0135 | +0.0000 |
| Proposed AZ-based method | 0.7489 +- 0.0007 | 0.5985 +- 0.0009 | 0.7959 +- 0.0085 | 0.7072 +- 0.0060 | 0.8447 +- 0.0025 | +0.0046 |

Интерпретация:
- после донастройки архитектуры AZ-вариант стал конкурентным и немного лучше на DRIVE по Dice/IoU;
- выигрыш небольшой, поэтому формулировка в статье должна быть аккуратной (не “radical improvement”).

## 3) Сравнение с Attention U-Net на DRIVE

Источник:
- `results/final_pack_20260424/drive_attention_unet_s42_e20/drive_multiseed_summary.md`

| Model | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Baseline U-Net | 0.7432 | 0.5913 | 0.7711 | 0.7172 |
| Attention U-Net | 0.7407 | 0.5882 | 0.8061 | 0.6851 |

Практический вывод: baseline и attention близки по Dice; attention дает более высокую precision ценой recall.

## 4) Что выносить в основной текст статьи

1. Основная мультидатасетная таблица (DRIVE/CHASE/FIVES/ARCADE) должна показывать baseline как сильный и стабильный ориентир.
2. Proposed AZ-based method лучше позиционировать как:
   - интерпретируемый геометрический метод;
   - метод с конкурентной точностью на DRIVE после targeted architectural fix.
3. Для сильного общего claims по переносимости нужно повторить final AZ-кандидат на CHASE/FIVES/ARCADE в том же multi-seed протоколе.

## 5) Готовые аккуратные формулировки для раздела “Выводы”

- `Baseline U-Net remains the strongest robust reference across the full multi-dataset benchmark in our current pack.`
- `The proposed AZ-based model provides interpretable local geometry and reaches competitive (slightly higher) DRIVE multi-seed accuracy after architecture-level balancing.`
- `The current gain is modest; therefore, we position the contribution as a combination of competitive segmentation quality and explicit geometry-aware interpretability.`

