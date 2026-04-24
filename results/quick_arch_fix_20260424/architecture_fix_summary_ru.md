# Быстрый архитектурный фикс `az_thesis`

Дата: 2026-04-24
GPU: NVIDIA GeForce RTX 5070 Ti
Датасет для быстрой проверки: DRIVE, seed 42, 6 эпох

## Что было не так

В финальном пакете `az_thesis` проигрывал baseline на всех датасетах. Главная техническая причина на DRIVE: вариант запускался слишком близко к режиму `pure AZ`, где AZ-блоки стояли не только как геометрическая добавка, но и в bottleneck/decoder/boundary head. В результате модель становилась тяжелее, хуже калибровалась и не успевала стабильно обучиться.

Дополнительно выяснилось, что прежний `az_thesis` использовал `geometry_mode=fixed_cat_map` и `learn_directions=false`. Это подходит для теоретической cat-map интерпретации, но плохо показывает адаптацию направления к объекту: `final_anisotropy_gap` был почти нулевой.

## Что исправлено

Добавлен конфиг `configs/drive_az_thesis_quick_fix.yaml`:

- AZ оставлен как ранняя геометрическая ветка encoder-а: `encoder_block_mode=hybrid_shallow`, `encoder_az_stages=1`.
- Decoder, bottleneck и boundary head возвращены к стабильным U-Net-like частям: `aspp`, `residual`, `conv`.
- Включена обучаемая локальная геометрия: `az_geometry_mode=local_hyperbolic`, `az_learn_directions=true`.
- Снижен дисбаланс BCE: `bce_pos_weight=8.0`.
- Threshold sweep расширен до `0.95`, чтобы не обрезать лучший порог.

Также код теперь поддерживает AZConvConfig-переопределения из YAML:

- `az_geometry_mode`
- `az_learn_directions`
- `az_use_fuzzy`
- `az_use_anisotropy`
- `az_min_hyperbolicity`
- `az_normalize_kernel`
- `az_use_value_projection`

## Быстрый результат

Прогон:

`results/quick_arch_fix_20260424/drive_quick_localgeom_fix_s42_e6`

| Model | Dice | IoU | Precision | Recall | Balanced Acc | Threshold | Anisotropy gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline U-Net | 0.6486 | 0.4799 | 0.7154 | 0.5932 | 0.7850 | 0.675 | 0.0000 |
| Proposed AZ-based method | 0.6702 | 0.5039 | 0.7366 | 0.6147 | 0.7966 | 0.650 | 0.6944 |

Вывод: на коротком диагностическом прогоне архитектурный фикс дал `+0.0216 Dice` к baseline и одновременно сделал геометрию видимой: ненулевой anisotropy gap показывает, что ядро действительно использует растяжение/сжатие, а не просто фиксированную стрелку.

## Контроль на 12 эпохах

Чтобы проверить, не является ли выигрыш ранним шумом, был выполнен контрольный прогон на 12 эпохах:

`results/quick_arch_fix_20260424/drive_quick_localgeom_fix_s42_e12`

| Model | Dice | IoU | Precision | Recall | Balanced Acc | Threshold | Anisotropy gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline U-Net | 0.7292 | 0.5738 | 0.7323 | 0.7261 | 0.8500 | 0.800 | 0.0000 |
| Proposed AZ-based method | 0.7092 | 0.5494 | 0.7797 | 0.6504 | 0.8162 | 0.800 | 0.6958 |

После 12 эпох baseline снова оказался выше по Dice. Профиль ошибки показывает причину: AZ-модель стала более консервативной, то есть precision выше, но recall заметно ниже.

Дополнительный recall-oriented probe:

`results/quick_arch_fix_20260424/drive_recall_probe_localgeom_s42_e12`

| Model | Dice | IoU | Precision | Recall | Balanced Acc | Threshold | Anisotropy gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed AZ-based method | 0.7239 | 0.5672 | 0.7899 | 0.6680 | 0.8253 | 0.850 | 0.6952 |

При фиксированном пороге `0.80` этот же checkpoint дает Dice `0.7297`, но это не следует использовать как основной результат статьи, потому что порог в таком случае выбран по test-поведению. Честный вывод: архитектурный фикс почти догнал baseline и сильно улучшил прежний `az_thesis`, но для финальной таблицы нужен multi-seed прогон и более аккуратный validation-based выбор порога.

## Контроль на 20 эпохах

Основной 20-эпоховый контроль:

`results/quick_arch_fix_20260424/drive_recall_probe_coremean_s42_e20`

| Model | Dice | IoU | Precision | Recall | Balanced Acc | Threshold | Anisotropy gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline U-Net | 0.7438 | 0.5921 | 0.7894 | 0.7031 | 0.8424 | 0.800 | 0.0000 |
| Proposed AZ-based method | 0.7317 | 0.5770 | 0.8079 | 0.6687 | 0.8266 | 0.850 | 0.6981 |

Tversky recall-probe:

`results/quick_arch_fix_20260424/drive_tversky_recall_probe_s42_e20`

| Model | Dice | IoU | Precision | Recall | Balanced Acc | Threshold | Anisotropy gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed AZ-based method | 0.7325 | 0.5779 | 0.8087 | 0.6694 | 0.8269 | 0.850 | 0.6979 |

Tversky почти не изменил профиль: precision остается выше baseline, recall остается ниже. Это значит, что следующий полезный шаг не очередной loss-tweak, а архитектурная/семплинговая работа с пропущенными тонкими структурами.

## Thin-vessel sampling probes

Так как 20-эпоховый контроль проигрывал baseline главным образом по recall, были проверены варианты crop sampling с фокусом на тонкие сосудистые пиксели.

Агрессивный thin-bias:

`results/quick_arch_fix_20260424/drive_thinbias_localgeom_s42_e20`

| Model | Dice | IoU | Precision | Recall | Balanced Acc | Threshold | Anisotropy gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed AZ-based method | 0.7265 | 0.5705 | 0.7625 | 0.6938 | 0.8363 | 0.850 | 0.6988 |

Мягкий thin-bias:

`results/quick_arch_fix_20260424/drive_thinbias_mild_localgeom_s42_e20`

| Model | Dice | IoU | Precision | Recall | Balanced Acc | Threshold | Anisotropy gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed AZ-based method | 0.7326 | 0.5780 | 0.8238 | 0.6596 | 0.8229 | 0.850 | 0.6968 |

Вывод по sampling: агрессивный thin-bias действительно поднимает recall почти до baseline (`0.6938` против `0.7031`), но слишком сильно снижает precision. Мягкий thin-bias возвращает precision, но снова теряет recall. Значит текущая проблема не решается простым crop bias: нужен либо hard-mining по реальным ошибкам модели, либо архитектурный вариант, который добавляет AZ не только в первый encoder-блок, но и во второй, при меньшей доле `hybrid_mix_init`.

Hard-mining cache по false-negative областям:

`results/quick_arch_fix_20260424/drive_hardmine_localgeom_s42_e20`

| Model | Dice | IoU | Precision | Recall | Balanced Acc | Threshold | Anisotropy gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed AZ-based method | 0.7310 | 0.5761 | 0.7904 | 0.6800 | 0.8312 | 0.900 | 0.6997 |

Hard-mining немного повышает recall относительно базового enc1-рецепта, но не дает прироста Dice. В текущем виде карты ошибок слишком широкие и добавляют шум в sampling.

## Encoder-stage probes

Наиболее полезным оказался архитектурный ход: добавить AZ-геометрию во второй encoder-stage, но с малой долей AZ-ветки (`hybrid_mix_init=0.15`), чтобы не вернуться к нестабильному pure-AZ режиму.

`results/quick_arch_fix_20260424/drive_enc2_lowmix_localgeom_s42_e20`

| Model | Dice | IoU | Precision | Recall | Balanced Acc | Threshold | Anisotropy gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed AZ-based method, pos_weight=10 | 0.7353 | 0.5814 | 0.7514 | 0.7198 | 0.8483 | 0.850 | 1.3876 |

`results/quick_arch_fix_20260424/drive_enc2_lowmix_pos8_s42_e20`

| Model | Dice | IoU | Precision | Recall | Balanced Acc | Threshold | Anisotropy gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed AZ-based method, pos_weight=8 | 0.7378 | 0.5846 | 0.7893 | 0.6927 | 0.8373 | 0.800 | 1.3942 |

Это лучший текущий 20-эпоховый AZ-рецепт. Он все еще немного ниже baseline по Dice (`0.7378` против `0.7438`), но разрыв уже мал (`-0.0060 Dice`), precision почти совпадает с baseline, а геометрическая часть стала существенно более выраженной (`anisotropy gap` около `1.39` вместо `0.70` у enc1).

## Multi-seed контроль финального кандидата

После single-seed архитектурных проб был выполнен честный 3-seed контроль для кандидата `encoder_az_stages=2`, `hybrid_mix_init=0.15`, `bce_pos_weight=8.0`:

`results/quick_arch_fix_20260424/drive_final_candidate_enc2_lowmix_pos8_ms_414243_e20`

| Model | Dice mean +- std | IoU mean +- std | Precision mean +- std | Recall mean +- std | Balanced Acc mean +- std | Threshold mean +- std | Dice vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline U-Net | 0.7460 +- 0.0035 | 0.5950 +- 0.0045 | 0.7613 +- 0.0276 | 0.7330 +- 0.0242 | 0.8551 +- 0.0100 | 0.7917 +- 0.0118 | +0.0000 |
| Proposed AZ-based method | 0.7424 +- 0.0077 | 0.5904 +- 0.0097 | 0.8027 +- 0.0228 | 0.6922 +- 0.0291 | 0.8377 +- 0.0131 | 0.8167 +- 0.0624 | -0.0036 |

Вывод: этот вариант почти догнал baseline и стал заметно более точным по precision, но все еще проигрывал по recall.

## Recall-balanced sweep

Чтобы поднять recall без разрушения геометрической интерпретируемости, был выполнен компактный sweep вокруг лучшей зоны:

`results/quick_arch_fix_20260424/drive_localgeom_recall_precision_sweep_s42_e20`

Проверялись:

- `hybrid_mix_init`: `0.10`, `0.15`, `0.20`
- `bce_pos_weight`: `7`, `8`, `9`
- фиксировано: `encoder_az_stages=2`, `az_geometry_mode=local_hyperbolic`, `az_learn_directions=true`, `topology_loss_weight=0.0`

Лучший single-seed вариант:

| Candidate | Dice | IoU | Precision | Recall | Balanced Acc | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| `hybrid_mix_init=0.10`, `bce_pos_weight=9.0` | 0.7479 | 0.5973 | 0.7984 | 0.7034 | 0.8430 | 0.800 |

Также исправлен `scripts/run_az_thesis_sweep.py`: теперь `best_trial_overrides.yaml` сохраняет не только sweep-параметры, но и важные AZ-поля (`az_geometry_mode`, `az_learn_directions`, регуляризации, threshold sweep). Это защищает от случайной потери обучаемой локальной геометрии при повторном запуске лучшего trial.

## Финальный 3-seed результат

Новый финальный кандидат:

`configs/drive_az_thesis_final_candidate_recall.yaml`

Прогон:

`results/quick_arch_fix_20260424/drive_final_candidate_recall_hm010_pos9_ms_414243_e20`

| Model | Dice mean +- std | IoU mean +- std | Precision mean +- std | Recall mean +- std | Specificity mean +- std | Balanced Acc mean +- std | Threshold mean +- std | Dice vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline U-Net | 0.7442 +- 0.0037 | 0.5926 +- 0.0047 | 0.7868 +- 0.0303 | 0.7083 +- 0.0313 | 0.9809 +- 0.0043 | 0.8446 +- 0.0135 | 0.7917 +- 0.0118 | +0.0000 |
| Proposed AZ-based method | 0.7489 +- 0.0007 | 0.5985 +- 0.0009 | 0.7959 +- 0.0085 | 0.7072 +- 0.0060 | 0.9822 +- 0.0011 | 0.8447 +- 0.0025 | 0.8167 +- 0.0236 | +0.0046 |

По seed-ам:

| Seed | Baseline Dice | Proposed Dice | Delta Dice | Proposed Precision | Proposed Recall | Anisotropy gap |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | 0.7392 | 0.7486 | +0.0094 | 0.8034 | 0.7009 | 1.3951 |
| 42 | 0.7455 | 0.7498 | +0.0043 | 0.8001 | 0.7054 | 1.4058 |
| 43 | 0.7480 | 0.7481 | +0.0001 | 0.7841 | 0.7153 | 1.4048 |

Вывод: впервые получен стабильный 3-seed результат, где `Proposed AZ-based method` не только сохраняет интерпретируемую локальную геометрию (`anisotropy gap` около `1.40`), но и обгоняет baseline по Dice/IoU в среднем. Выигрыш небольшой, поэтому в статье его нужно подавать аккуратно: основной вклад не "большой скачок метрик", а комбинация конкурентной точности и объяснимой направленной геометрии.

## Визуализация

Новая объясняющая картинка:

`article_assets/final_figures/quick_drive_localgeom_story/figure_geometry_attention_story_drive_15_test.png`

Что она теперь показывает корректнее:

- `Axes u/s`: длинная ось `u` показывает направление, вдоль которого локальное ядро агрегирует признаки.
- Короткая ось `s` показывает поперечное сжатие.
- `Direction Field`: направление показано только там, где модель считает область значимой.
- `Contribution`: вклад AZ-геометрии в раннем слое.
- `Regime Confidence`: насколько уверенно выбран локальный fuzzy-режим.

Финальная картинка для нового recall-balanced кандидата:

`article_assets/final_figures/drive_final_recall_geometry_story/figure_geometry_attention_story_drive_03_test.png`

Эта версия построена уже из `az_thesis_seed42` прогонов `drive_final_candidate_recall_hm010_pos9_ms_414243_e20` и сравнивается с парным `baseline_seed42`.

## Следующий шаг

Главный текущий кандидат для статьи: `encoder_az_stages=2`, `hybrid_mix_init=0.10`, `bce_pos_weight=9.0`, `az_geometry_mode=local_hyperbolic`.

Дальше полезно сделать две вещи:

- повторить этот кандидат на ARCADE/FIVES/CHASE, чтобы понять переносимость преимущества;
- подготовить финальную таблицу для статьи: `U-Net`, `Attention U-Net`, `Proposed AZ-based method`, плюс подпись про интерпретируемую геометрию и стоимость (`params`, `GMACs`, latency).
