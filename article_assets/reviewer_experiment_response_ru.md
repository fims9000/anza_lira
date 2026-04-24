# Блок для ответа рецензенту: сравнение, данные, обучение, применимость

## Что у нас уже есть в реализации

В коде реализованы:

- `baseline` = standard U-Net;
- `attention_unet` = Attention U-Net;
- `az_thesis` = proposed geometry-aware AZ-based method;
- `az_no_fuzzy`, `az_no_aniso`, `az_cat` = ablation variants.

Не реализованы и не обучались в текущей версии:

- U-Net++;
- TransUNet / transformer-based segmentation model.

Поэтому в статье корректная формулировка такая: сравнение выполнено со standard U-Net и частично с Attention U-Net; сравнение с U-Net++ и transformer architectures оставлено как направление дальнейшей работы.

## Датасеты и split

| Dataset | Task | Train | Val | Test | Source config |
|---|---:|---:|---:|---:|---|
| DRIVE | retinal vessel segmentation | 16 | 4 | 20 | `configs/drive_benchmark.yaml` |
| CHASE-DB1 | retinal vessel segmentation | 6 | 2 | 20 | `configs/chase_db1_benchmark.yaml` |
| FIVES | retinal vessel segmentation | 510 | 90 | 200 | `configs/fives_benchmark.yaml` |
| ARCADE Syntax | coronary artery segmentation | 1000 | 200 | 300 | `configs/arcade_syntax_benchmark.yaml` |
| ARCADE Stenosis | coronary stenosis/vessel objective | 1000 | 200 | 300 | `configs/arcade_stenosis_benchmark.yaml` |

Для второй статьи про сегментационный алгоритм лучше использовать retinal datasets как основные, а ARCADE оставить как дополнительный пример применимости к сосудистым структурам не из глазного дна.

## Параметры обучения

Общие параметры:

- optimizer: Adam;
- learning rate: `2e-4`;
- weight decay: `1e-4`;
- seed: `42`;
- device: CUDA/GPU;
- loss: BCE + Dice/Tversky overlap loss;
- threshold selection: validation threshold sweep.

По датасетам:

| Dataset | Epochs in final run | Batch size | Crop/patch | Threshold metric | Main threshold range |
|---|---:|---:|---:|---|---|
| DRIVE | 20 | 2 | 256 | core_mean | 0.55-0.80 |
| CHASE-DB1 | 20 | 1 | 512 | dice | 0.30-0.80 |
| FIVES | 10 | 1 | 640 | dice | 0.30-0.80 |
| ARCADE Syntax | 20 | 2 | 256 | core_mean | 0.30-0.70 |
| ARCADE Stenosis | 20 | 2 | 256 | core_mean | 0.30-0.70 |

Augmentation:

- retinal datasets: random patch sampling with foreground-biased crop selection;
- ARCADE: `arcade_augment=true`, random cropped training patches.

## Метрики: standard U-Net vs proposed method

| Dataset | U-Net Dice | Proposed Dice | Delta Dice | U-Net IoU | Proposed IoU | Delta IoU |
|---|---:|---:|---:|---:|---:|---:|
| DRIVE | 0.7432 | 0.5949 | -0.1483 | 0.5913 | 0.4234 | -0.1680 |
| CHASE-DB1 | 0.6725 | 0.6479 | -0.0246 | 0.5066 | 0.4792 | -0.0274 |
| FIVES | 0.7502 | 0.7199 | -0.0302 | 0.6002 | 0.5624 | -0.0378 |
| ARCADE Syntax | 0.6522 | 0.5963 | -0.0560 | 0.4839 | 0.4248 | -0.0592 |
| ARCADE Stenosis | 0.3034 | 0.2075 | -0.0959 | 0.1788 | 0.1158 | -0.0631 |

Честный вывод: в текущей серии запусков proposed `az_thesis` не превосходит standard U-Net по Dice/IoU. Поэтому нельзя писать, что метод улучшает U-Net по численным метрикам. Можно писать, что метод вводит интерпретируемый geometry-aware local aggregation mechanism и демонстрирует применимость на нескольких сегментационных датасетах, но требует дальнейшей оптимизации для конкурентного качества.

## Дополнительное сравнение с Attention U-Net на DRIVE

| Method | Dice | IoU | Precision | Recall | Params | GMACs |
|---|---:|---:|---:|---:|---:|---:|
| Standard U-Net | 0.7432 | 0.5913 | 0.7711 | 0.7172 | 1.54M | 20.17 |
| Attention U-Net | 0.7407 | 0.5882 | 0.8061 | 0.6851 | 1.57M | 20.55 |
| Proposed AZ-based method | 0.5949 | 0.4234 | 0.6054 | 0.5847 | 2.01M | 26.57 |

Источник Attention U-Net:

- `results/final_pack_20260424/drive_attention_unet_s42_e20/attention_unet_seed42/metrics.json`

## Что писать про сравнение с U-Net++ / TransUNet

Без запусков нельзя заявлять численное сравнение с U-Net++ или TransUNet. Безопасная формулировка:

> In the present implementation we compare the proposed geometry-aware local aggregation model with a standard U-Net baseline and an Attention U-Net baseline. Large transformer-based segmentation architectures, such as TransUNet, and extended encoder-decoder variants, such as U-Net++, are not included in the current experimental protocol and will be considered in future work.

## Применимость метода

Метод имеет смысл применять там, где целевой объект имеет локально вытянутую или направленную структуру:

- сосуды глазного дна;
- коронарные сосуды;
- тонкие линейные структуры в медицинских изображениях;
- дорожные/речные/линейные объекты в GIS/remote sensing;
- трещины, волокна, контуры, трубчатые структуры.

Главный аргумент применимости: оператор не просто агрегирует пиксели в квадратном окне, а учит локальную направленную геометрию: усиливает соседей вдоль оси `u` и подавляет поперечные несогласованные отклики вдоль `s`.

## Что нужно исправить в тексте статьи

1. Добавить таблицу comparison with existing methods: U-Net, Attention U-Net, proposed AZ-based method.
2. Убрать фразу, что proposed method лучше baseline по метрикам, если она где-то есть.
3. Подчеркнуть новизну как interpretability + local anisotropic fuzzy aggregation, а не как SOTA по Dice.
4. Добавить описание датасета, split, epochs, optimizer, learning rate, batch size, patch size, loss, threshold selection.
5. Добавить ссылку на публичный GitHub после публикации репозитория.

## Где лежат источники

- Multi-dataset summary: `results/final_pack_20260424/final_multidataset_summary_ru.md`
- DRIVE U-Net/AZ metrics: `results/final_pack_20260424/drive_baseline_vs_azthesis_s42_e20/all_metrics.json`
- DRIVE Attention U-Net metrics: `results/final_pack_20260424/drive_attention_unet_s42_e20/all_metrics.json`
- CHASE metrics: `results/final_pack_20260424/chase_baseline_vs_azthesis_s42_e20/all_metrics.json`
- FIVES metrics: `results/final_pack_20260424/fives_baseline_vs_azthesis_s42_e10/all_metrics.json`
- ARCADE Syntax metrics: `results/final_pack_20260424/arcade_syntax_baseline_vs_azthesis_s42_e20/all_metrics.json`
- ARCADE Stenosis metrics: `results/final_pack_20260424/arcade_stenosis_baseline_vs_azthesis_s42_e20/all_metrics.json`
