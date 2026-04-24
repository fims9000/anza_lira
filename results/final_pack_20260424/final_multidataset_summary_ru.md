# Финальная сводка по датасетам (baseline vs az_thesis)

Дата сборки: 2026-04-24
Режим: GPU (`cuda`), seed `42`

## Что включено в этот пакет

- DRIVE: `drive_baseline_vs_azthesis_s42_e20`
- CHASE-DB1: `chase_baseline_vs_azthesis_s42_e20`
- FIVES: `fives_baseline_vs_azthesis_s42_e10` (стабильный завершенный прогон)
- ARCADE Syntax: `arcade_syntax_baseline_vs_azthesis_s42_e20`
- ARCADE Stenosis: `arcade_stenosis_baseline_vs_azthesis_s42_e20`

## Ключевые метрики (test)

| Dataset | Baseline Dice | AZ-Thesis Dice | Delta Dice | Baseline IoU | AZ-Thesis IoU | Delta IoU |
|---|---:|---:|---:|---:|---:|---:|
| drive | 0.7432 | 0.5949 | -0.1483 | 0.5913 | 0.4234 | -0.1680 |
| chase_db1 | 0.6725 | 0.6479 | -0.0246 | 0.5066 | 0.4792 | -0.0274 |
| fives (e10) | 0.7502 | 0.7199 | -0.0302 | 0.6002 | 0.5624 | -0.0378 |
| arcade_syntax | 0.6522 | 0.5963 | -0.0560 | 0.4839 | 0.4248 | -0.0592 |
| arcade_stenosis | 0.3034 | 0.2075 | -0.0959 | 0.1788 | 0.1158 | -0.0631 |

## Вывод

В этой серии запусков `az_thesis` не обогнал `baseline` ни на одном из собранных датасетов.
Для прикладной статьи сейчас безопаснее держать `baseline` как основной эталон, а `az_thesis` показывать как исследовательскую ветку с упором на интерпретируемость/геометрию и дальнейший тюнинг.
