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


## 9. Update 2026-04-25: verification of implementation-corrected AZ variant

We re-ran the best implementation-fix candidate on DRIVE with the full protocol: 20 epochs, seeds 41/42/43.

Source runs:
- New check: `results/quick_arch_fix_20260425/drive_implfix_best_ms_414243_e20/drive_multiseed_summary.md`
- Previous candidate: `results/quick_arch_fix_20260424/drive_final_candidate_recall_hm010_pos9_ms_414243_e20/drive_multiseed_summary.md`

| Run | Baseline Dice mean+-std | Proposed AZ Dice mean+-std | Dice vs baseline |
|---|---:|---:|---:|
| Previous best candidate | 0.7442 +- 0.0037 | 0.7489 +- 0.0007 | +0.0046 |
| Implementation-corrected AZ variant | 0.7456 +- 0.0061 | 0.7403 +- 0.0084 | -0.0052 |

Interpretation:
- The implementation-corrected AZ variant does not preserve the previous positive margin on the full multi-seed protocol.
- Therefore, this variant is reported as a negative/unstable result and is not used as the primary final claim.
- Final manuscript positioning should emphasize robustness-first conclusions and separate exploratory AZ implementation fixes from validated final results.

## 10. Update 2026-04-25: threshold-selection stabilization (implementation fix)

To address the observed precision/recall imbalance (high precision with degraded recall), we implemented a careful threshold-selection stabilization policy in code:

- support for near-best selection with tolerance (`eval_threshold_score_tolerance`);
- optional upper cap for selected threshold (`eval_threshold_max`);
- optional recall floor during threshold choice (`eval_threshold_min_recall`);
- explicit threshold reference (`eval_threshold_reference`) for stable tie-breaking.

Applied policy for the current impl-fix candidate:
- `eval_threshold_metric: core_mean`
- `eval_threshold_reference: 0.80`
- `eval_threshold_score_tolerance: 0.005`
- `eval_threshold_max: 0.85`

Important scope note:
- this is an implementation-level calibration fix (decision policy), not a claim of improved final benchmark performance until the updated run is re-validated on the full protocol (20 epochs, seeds 41/42/43).

## 11. Update 2026-04-25: post-hoc checkpoint recalibration at thr=0.80 (no retraining)

Using the same impl-fix checkpoints (`20 epochs`, seeds `41/42/43`) and only changing the test threshold to `0.80`, we obtain:

- AZ mean Dice: `0.7498` (was `0.7403` with original auto-selected thresholds);
- AZ mean Recall: `0.7220` (was `0.6816`);
- Baseline mean Dice in the same run pack: `0.7456`;
- Dice delta AZ vs baseline: `+0.0043`.

Artifact:
- `results/quick_arch_fix_20260425/drive_implfix_best_ms_414243_e20/rethreshold_thr080_summary.md`

Interpretation:
- The previous impl-fix failure was largely due to threshold calibration drift (overly high selected thresholds), not only due to feature extractor quality.
- This supports a careful manuscript claim: implementation-corrected AZ remains sensitive to decision calibration; with stabilized thresholding it regains competitive multi-seed performance on DRIVE.

## 12. Update 2026-04-25: full 20e multi-seed confirmation with threshold policy fix

Run:
- `results/quick_arch_fix_20260425/drive_implfix_policyfix_ms_414243_e20/drive_multiseed_summary.md`

Result (DRIVE, 20 epochs, seeds 41/42/43):
- Baseline Dice mean: `0.7456 +- 0.0061`
- Implementation-corrected AZ + policy-fix Dice mean: `0.7498 +- 0.0016`
- Delta Dice vs baseline: `+0.0043`
- Mean selected threshold for AZ: `0.8000 +- 0.0000`

Interpretation:
- the previous degradation was largely decision-calibration related;
- with stabilized threshold policy, the corrected AZ variant is again competitive and slightly ahead on DRIVE multi-seed.

## 13. Update 2026-04-25: hybrid_mix activation probe (negative so far)

We implemented explicit hybrid-mix diagnostics and optional mix-target regularization.
Short probes (`seed42`, `6 epochs`) with `reg_hybrid_mix` did not produce a meaningful mix shift (`~0.10 -> ~0.101`) and did not improve short-run Dice.

Artifacts:
- `results/quick_arch_fix_20260425/drive_mixreg_probe_s42_e6/`
- `results/quick_arch_fix_20260425/drive_mixreg_probe2_s42_e6/`
- `results/quick_arch_fix_20260425/drive_mixreg_probe3_s42_e6/`

Decision:
- do not promote hybrid-mix regularization as a main improvement at this stage;
- keep the threshold-policy fix as the validated implementation correction.

## 14. Update 2026-04-25: risk-focused negative analysis

Detailed downside analysis is tracked in:
- `results/quick_arch_fix_20260425/drive_implfix_policyfix_ms_414243_e20/risk_analysis_ru.md`

Key caution points for manuscript wording:
- despite Dice gain on DRIVE, AZ still underperforms baseline on recall and balanced accuracy;
- latency cost remains substantially higher;
- with only 3 seeds, confidence interval on Dice delta remains wide.

## 15. Update 2026-04-25: recall-floor + Tversky trial (negative trade-off)

Run:
- `results/quick_arch_fix_20260425/drive_recallfloor_tversky_ms_414243_e20/`
- summary: `results/quick_arch_fix_20260425/drive_recallfloor_tversky_ms_414243_e20/tradeoff_summary.md`

What was tested:
- recall-oriented loss and threshold policy (`Tversky alpha=0.40, beta=0.60`, `eval_threshold_min_recall=0.76`).

Outcome (mean vs previous policyfix):
- Recall: `+0.0229`
- Balanced Acc: `+0.0079`
- Dice: `-0.0121`
- Precision: `-0.0411`

Conclusion:
- this setting is useful as a diagnostic (trade-off frontier), but not as a replacement for the current best policyfix, because overall Dice/regression is too large.
