# Impl-Fix Ablation (AZConv) - 2026-04-25

Цель: закрыть незавершенные implementation-проблемы в AZ-ядре, а не только тюнить loss.

## Что изменено в коде

Файл `models/azconv.py`:

- добавлены новые параметры `AZConvConfig`:
  - `fuzzy_temperature` (мягкость fuzzy-gate),
  - `normalize_mode` (`global|per_rule|none`),
  - `compatibility_floor` (малый стабилизирующий floor на compat).
- добавлен `softmax(logits / temperature)`.
- добавлены режимы нормализации compat:
  - `global` (как раньше),
  - `per_rule` (нормализация по соседям внутри правила с сохранением массы `mu_center`),
  - `none` (без нормализации).

Файл `utils.py`:

- добавлен парсинг новых ключей конфига:
  - `az_fuzzy_temperature`,
  - `az_normalize_mode`,
  - `az_compatibility_floor`.

Файл `train.py`:

- новые поля сохраняются в `metrics.json` для трассировки экспериментов.

## Мини-абляция (DRIVE, seed42, 6 epochs)

Базовый reference для этого же короткого режима:

- Baseline: Dice `0.6586` (`results/quick_arch_fix_20260425/drive_implfix_probe_s42_e6/drive_multiseed_summary.md`)

Варианты AZ impl-fix:

| Session | Temp | Normalize | Floor | Dice | IoU | Precision | Recall | BalAcc |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `implfix_gl_t135_f5e4` | 1.35 | global | 0.0005 | 0.6482 | 0.4795 | 0.7378 | 0.5780 | 0.7789 |
| `implfix_pr_t135_f5e4` | 1.35 | per_rule | 0.0005 | 0.6440 | 0.4749 | 0.7118 | 0.5880 | 0.7823 |
| `implfix_pr_t115_f0` | 1.15 | per_rule | 0.0 | 0.6358 | 0.4661 | 0.7162 | 0.5717 | 0.7748 |
| `implfix_pr_t100_f0` | 1.00 | per_rule | 0.0 | 0.6199 | 0.4492 | 0.7412 | 0.5327 | 0.7573 |

Дополнительный архитектурный стабилизатор:

- `implfix_gl_t135_f5e4_resid15`:
  - `use_input_residual=true`,
  - `residual_init=0.15`.

Результат:

| Session | Temp | Normalize | Floor | Input residual | Dice | IoU | Precision | Recall | BalAcc |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|
| `baseline (same e6 protocol)` | - | - | - | - | 0.6586 | 0.4910 | 0.7611 | 0.5804 | 0.7813 |
| `implfix_gl_t135_f5e4` | 1.35 | global | 0.0005 | no | 0.6482 | 0.4795 | 0.7378 | 0.5780 | 0.7789 |
| `implfix_gl_t135_f5e4_resid15` | 1.35 | global | 0.0005 | yes (`0.15`) | **0.6683** | **0.5018** | 0.7451 | **0.6058** | **0.7928** |

## Вывод по текущему шагу

1. Реализационные ручки добавлены корректно и воспроизводимо.
2. Вариант `per_rule` в e6-режиме не дал выигрыша и пока не подтвержден.
3. Вариант `global + temp=1.35 + floor=5e-4` + `input residual bypass` дал лучший результат и уже на e6 обошел baseline.

## Что делать дальше (архитектура, не только loss)

1. Повторить лучшую impl-fix конфигурацию (`global + temp=1.35 + floor=5e-4 + input residual`) на `20 epochs`, seed `42`, затем `41/42/43`.
2. Добавить в `HybridAZResidualBlock` ограничение диапазона mix (например, `mix in [0.05, 0.95]`) для стабилизации вклада AZ-ветки.
3. Запустить сравнительный run-pack:
   - `baseline`,
   - `az_thesis` (текущий best без impl-fix),
   - `az_thesis` + impl-fix (лучший из п.1),
   и уже из него собирать окончательный блок статьи.
