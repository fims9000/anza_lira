# Архитектура Для Раздела Methods

Источники:
- `models/segmentation.py`
- `models/azconv.py`
- `configs/drive_az_thesis_final_candidate_recall.yaml`

## 1) Baseline U-Net

`BaselineUNet`:
- 3-уровневый encoder-decoder;
- блоки `Conv-BN-ReLU` x2;
- skip-connections;
- bilinear upsampling + fusion;
- final `1x1` head.

Типовая ширина (в финальных прогонах DRIVE): `[32, 64, 128, 192]`.

## 2) Attention U-Net

`AttentionUNet`:
- та же базовая пирамидальная структура;
- attention gate на skip-соединениях;
- цель: улучшить фильтрацию skip-фичей в декодере.

## 3) Proposed AZ-based method (AZ-SOTA family)

Основной исследовательский класс: `AZSOTAUNet`.

Ключевые части:
- residual encoder с AZ-блоками;
- ASPP bottleneck;
- attention-gated decoder;
- deep supervision (`aux_logits`);
- boundary head (`boundary_logits`).

Гибкая конфигурация:
- `encoder_az_stages` (сколько encoder-стадий с AZ);
- `encoder_block_mode` = `az | hybrid | hybrid_shallow`;
- `hybrid_mix_init` (доля AZ-ветки в hybrid-блоке);
- режимы bottleneck/decoder/boundary (`aspp/residual/conv` и AZ-варианты).

## 4) Локальное геометрическое ядро AZConv2d

Базовая идея AZ-слоя:
- для каждого локального окна `k x k` используется rule-wise анизотропная агрегация;
- смешение правил задается fuzzy membership `mu_r`;
- геометрия задается режимами:
  - `fixed_cat_map`
  - `learned_angle`
  - `learned_hyperbolic`
  - `local_hyperbolic`

Интуитивная форма локального вклада:

`K_r(center, neighbor) = mu_r(center) * mu_r(neighbor) * kappa_r(center, neighbor, offset)`

где `kappa_r` включает анизотропию по осям `(u, s)` через `sigma_u`, `sigma_s`.

## 5) Финальный DRIVE-кандидат (rebalanced)

Из `configs/drive_az_thesis_final_candidate_recall.yaml`:

- `encoder_az_stages: 2`
- `encoder_block_mode: hybrid`
- `hybrid_mix_init: 0.10`
- `bottleneck_mode: aspp`
- `decoder_mode: residual`
- `boundary_mode: conv`
- `az_geometry_mode: local_hyperbolic`
- `az_learn_directions: true`
- `az_min_hyperbolicity: 0.15`
- `bce_pos_weight: 9.0`
- `eval_threshold_metric: core_mean`, sweep `0.25..0.95`

Это и есть текущая конфигурация `Proposed AZ-based method`, которая дала лучший стабильный DRIVE multi-seed результат.

## 6) Текст для статьи (готовый абзац)

`Our proposed model builds on an AZ-enhanced residual U-Net with an ASPP bottleneck, attention-gated decoding, deep supervision, and an auxiliary boundary head. The core AZConv operator performs rule-wise local anisotropic aggregation with fuzzy memberships, enabling explicit geometry-aware filtering via local (u,s) directional scales. In the final DRIVE candidate, AZ geometry is injected in early encoder stages through a hybrid residual design, while bottleneck/decoder remain stability-oriented (ASPP/residual).`

