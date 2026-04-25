# Geometry-Aware AZ-Based Vessel Segmentation: Черновик Статьи

Дата версии: 2026-04-24  
Версия: draft v1 (на основе `results/article_package_20260424/*`)

## Аннотация

Сегментация тонких сосудистых структур в офтальмологических и коронарных изображениях остается сложной задачей из-за сильного дисбаланса классов, локального шума и высокой чувствительности к топологии ветвления. В этой работе мы рассматриваем geometry-aware AZ-based подход, в котором локальная анизотропная агрегация признаков организована через rule-wise fuzzy режимы и интерпретируемые направленные оси. В сравнительных экспериментах используются пять датасетов: DRIVE, CHASE-DB1, FIVES, ARCADE Syntax и ARCADE Stenosis. Базовым ориентиром выступает Baseline U-Net; дополнительно приводится сравнение с Attention U-Net. На полном текущем мультидатасетном пакете baseline остается наиболее устойчивым референсом, тогда как предложенный метод после targeted architectural balancing демонстрирует конкурентный и немного более высокий multi-seed результат на DRIVE (Dice 0.7489 vs 0.7442). Полученные результаты показывают, что основной вклад подхода состоит в сочетании конкурентной точности и явной геометрической интерпретируемости.

Ключевые слова: vessel segmentation, U-Net, attention U-Net, geometry-aware learning, interpretability, retinal imaging, coronary angiography.

## 1. Введение

Автоматическая сегментация сосудистых структур применяется в широком спектре медицинских задач, включая скрининг офтальмологических патологий и анализ коронарных ангиограмм. При этом тонкие ветви, узкие просветы и локальные артефакты остаются наиболее проблемными областями даже для современных CNN-подходов. Практический вызов состоит не только в достижении высокой метрики Dice/IoU, но и в том, чтобы модель принимала объяснимые решения в сложных локальных участках.

Классические U-Net-подобные архитектуры эффективно агрегируют локальный контекст, но не задают явной геометрической структуры агрегации. В результате сложно интерпретировать, где модель усиливает вытянутые сосудистые паттерны, а где подавляет поперечный шум. Для устранения этого ограничения в работе рассматривается AZ-based геометрический блок, в котором локальная агрегация параметризуется анизотропными направлениями и fuzzy-режимами.

Цель работы: оценить, может ли geometry-aware AZ-based метод одновременно:
1. сохранить конкурентное качество сегментации;
2. предоставить интерпретируемую локальную геометрию решений;
3. быть устойчивым в multi-seed оценке.

## 2. Датасеты И Протокол

### 2.1 Набор датасетов

В работе используются пять датасетов:

- DRIVE (`data/DRIVE`): 20 training + 20 test.
- CHASE-DB1 (`data/CHASE_DB1`): 8 training + 20 test.
- FIVES (`data/FIVES`): 600 training + 200 test.
- ARCADE Syntax (`data/ARCADE/arcade/syntax`): train/val/test = 1000/200/300 изображений.
- ARCADE Stenosis (`data/ARCADE/arcade/stenosis`): train/val/test = 1000/200/300 изображений.

Для ARCADE маски строятся из polygon annotation (json). Для ретинальных датасетов используются RGB-изображения, ручные сосудистые маски и FOV-маски валидной области.

### 2.2 Preprocessing и аугментации

Общий preprocessing:
- RGB вход;
- нормализация ImageNet mean/std;
- бинаризация масок.

Ретинальные датасеты:
- patch-crop;
- foreground/thin-vessel/hard-mining bias sampling;
- train-time flips/rotations и фотометрические джиттеры.

ARCADE:
- опциональный resize;
- train patch-crop;
- flips/rotations;
- valid mask без FOV-ограничения.

### 2.3 Модели сравнения

Сравниваются:
- `Baseline U-Net`;
- `Attention U-Net`;
- `Proposed AZ-based method`.

### 2.4 Метрики

Основные метрики:
- Dice;
- IoU;
- Precision;
- Recall;
- Specificity;
- Balanced Accuracy.

Для части экспериментов применяется threshold sweep с выбором порога по целевой функции (`core_mean` или `dice` в зависимости от конфига).

## 3. Метод

### 3.1 Baseline и Attention U-Net

`Baseline U-Net` построен как классический encoder-decoder с skip-connections и bilinear upsampling.  
`Attention U-Net` добавляет attention gates на skip-путях для селективной фильтрации перед слиянием в декодере.

### 3.2 Proposed AZ-based method

Базовый исследовательский каркас реализован в `AZSOTAUNet` и включает:
- residual encoder;
- ASPP bottleneck;
- attention-gated decoder;
- deep supervision;
- boundary head.

Гибкость обеспечивается параметрами:
- `encoder_az_stages`;
- `encoder_block_mode` (`az/hybrid/hybrid_shallow`);
- `hybrid_mix_init`;
- режимы bottleneck/decoder/boundary.

### 3.3 AZConv: геометрически-ориентированная локальная агрегация

Локальный AZ-оператор использует rule-wise агрегацию в окне `k x k`:

`K_r(center, neighbor) = mu_r(center) * mu_r(neighbor) * kappa_r(center, neighbor, offset)`

где `mu_r` задает fuzzy membership, а `kappa_r` параметризует анизотропную геометрию с осями `(u, s)` и масштабами `sigma_u`, `sigma_s`.

Поддерживаются режимы геометрии:
- `fixed_cat_map`,
- `learned_angle`,
- `learned_hyperbolic`,
- `local_hyperbolic`.

Это позволяет визуализировать не только “где модель смотрит”, но и “как именно” организована локальная геометрическая агрегация.

### 3.4 Финальная balanced-конфигурация для DRIVE

Лучший multi-seed DRIVE-кандидат использует:
- `encoder_az_stages: 2`,
- `encoder_block_mode: hybrid`,
- `hybrid_mix_init: 0.10`,
- `bottleneck_mode: aspp`,
- `decoder_mode: residual`,
- `boundary_mode: conv`,
- `az_geometry_mode: local_hyperbolic`,
- `az_learn_directions: true`,
- `bce_pos_weight: 9.0`.

## 4. Эксперименты И Результаты

### 4.1 Основной мультидатасетный срез (seed 42)

| Dataset | Baseline Dice | Proposed Dice | Delta Dice | Baseline IoU | Proposed IoU | Delta IoU |
|---|---:|---:|---:|---:|---:|---:|
| DRIVE | 0.7432 | 0.5949 | -0.1483 | 0.5913 | 0.4234 | -0.1680 |
| CHASE-DB1 | 0.6725 | 0.6479 | -0.0246 | 0.5066 | 0.4792 | -0.0274 |
| FIVES (e10) | 0.7502 | 0.7199 | -0.0302 | 0.6002 | 0.5624 | -0.0378 |
| ARCADE Syntax | 0.6522 | 0.5963 | -0.0560 | 0.4839 | 0.4248 | -0.0592 |
| ARCADE Stenosis | 0.3034 | 0.2075 | -0.0959 | 0.1788 | 0.1158 | -0.0631 |

В этой серии baseline стабильно выше по Dice/IoU на всех датасетах.

### 4.2 DRIVE multi-seed после архитектурного балансирования (41/42/43)

| Model | Dice mean +- std | IoU mean +- std | Precision mean +- std | Recall mean +- std | Balanced Acc mean +- std | Dice vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| Baseline U-Net | 0.7442 +- 0.0037 | 0.5926 +- 0.0047 | 0.7868 +- 0.0303 | 0.7083 +- 0.0313 | 0.8446 +- 0.0135 | +0.0000 |
| Proposed AZ-based method | 0.7489 +- 0.0007 | 0.5985 +- 0.0009 | 0.7959 +- 0.0085 | 0.7072 +- 0.0060 | 0.8447 +- 0.0025 | +0.0046 |

На DRIVE предложенный метод становится конкурентным и показывает небольшой выигрыш по Dice/IoU.

### 4.3 Сравнение с Attention U-Net на DRIVE

| Model | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Baseline U-Net | 0.7432 | 0.5913 | 0.7711 | 0.7172 |
| Attention U-Net | 0.7407 | 0.5882 | 0.8061 | 0.6851 |

Attention U-Net смещает профиль в сторону большей precision, но теряет recall относительно baseline.

## 5. Интерпретируемость И Визуальный Анализ

### 5.1 Ключевые фигуры

- Fig.1: `article_assets/final_figures/figure1_pipeline.svg`
- Fig.2a: `article_assets/final_figures/figure2_drive_examples.png`
- Fig.2b: `article_assets/final_figures/figure2_chase_examples.png`
- Fig.3a: `article_assets/final_figures/figure3_xai.png`
- Fig.3b: `article_assets/final_figures/figure3_chase_xai.png`

### 5.2 Геометрические story-визуализации

- DRIVE final: `article_assets/final_figures/drive_final_recall_geometry_story/figure_geometry_attention_story_drive_03_test.png`
- ARCADE syntax: `article_assets/final_figures/figure_geometry_attention_story_arcade_syntax_212.png`
- ARCADE stenosis: `article_assets/final_figures/figure_geometry_attention_story_arcade_stenosis_289.png`

Интерпретация легенды:
- `u-axis`: направление вытяжения локального ядра;
- `s-axis`: поперечное сжатие;
- `Contribution`: локальный вклад анизотропной геометрии;
- `Regime confidence`: уверенность выбранного fuzzy-режима.

## 6. Обсуждение

Текущий набор экспериментов показывает двойственную, но методологически важную картину:

1. `Baseline U-Net` остается самым надежным общим ориентиром на полном мультидатасетном срезе.
2. После targeted architectural balancing предложенный AZ-подход дает конкурентный и немного более высокий multi-seed результат на DRIVE.
3. Главный вклад метода следует формулировать не как “крупный скачок метрик”, а как сочетание:
   - конкурентной сегментации,
   - явной геометрической интерпретируемости.

Для сильного cross-dataset claim требуется повтор финальной balanced-конфигурации на CHASE/FIVES/ARCADE в аналогичном multi-seed протоколе.

## 7. Заключение

В работе исследован geometry-aware AZ-based подход к сегментации сосудистых структур. На текущем полном benchmark-пакете baseline остается наиболее сильным референсом, однако предложенный метод после архитектурного балансирования показывает конкурентный и немного более высокий результат на DRIVE при сохранении интерпретируемой локальной геометрии. Таким образом, практический вклад подхода заключается в объединении качества и объяснимости. Следующий этап — подтверждение переносимости balanced AZ-конфигурации на остальных датасетах.

## 8. Ограничения И Дальнейшая Работа

- Ограничение: финальный лучший AZ-кандидат пока подтвержден multi-seed только на DRIVE.
- План:
  1. прогнать final balanced AZ-кандидат на CHASE/FIVES/ARCADE;
  2. собрать единый multi-seed свод по всем датасетам;
  3. подготовить финальную camera-ready таблицу.

## Приложение A: Пути К Артефактам

- Главный пакет: `results/article_package_20260424/`
- Итоговые таблицы: `results/final_pack_20260424/`
- DRIVE fix: `results/quick_arch_fix_20260424/drive_final_candidate_recall_hm010_pos9_ms_414243_e20/`
- Фигуры: `article_assets/final_figures/`

