# AZ-Thesis: геометрия, диагностика и визуализации для статьи

Дата: 2026-04-26.

Этот файл фиксирует, что надо доказать в статье помимо финальных Dice/IoU. Текущие обучения не трогались.

## 1. Как честно формулировать метод

В статье безопаснее позиционировать AZ-Thesis так:

> AZ-Thesis is a topology-aware segmentation architecture that augments a convolutional backbone with a soft rule-weighted anisotropic geometric correction branch. The geometric branch is motivated by hyperbolic directional systems, but is implemented as a stable finite feature-space operator for thin-structure segmentation.

Не делать главный claim как "Anosov neural network". Внутренние термины лучше переводить так:

| Internal term | Article term |
|---|---|
| Anosov / cat map | hyperbolic-inspired directional metric |
| fuzzy rules | soft rule weighting |
| AZ cat | AZ-Thesis final configuration |
| local hyperbolic | learnable/local directional metric variant |

## 2. Реальная формула реализации

В текущем коде метод ближе всего описывать так:

```text
Y = Res(X) + SE((1 - lambda) Conv(X) + lambda AZ(X))
```

где `lambda = sigmoid(mix_logit)` для hybrid encoder blocks.

Внутри AZConv:

```text
mu_r(x) = softmax(g_r(X) / tau)
K_r(x, y) = mu_r(x) mu_r(y) exp(-d_r(x, y))
AZ(X)(x) = PointwiseMix_r,c sum_y K_r(x, y) V(X)(y)
```

Для статьи:

- `mu_r` = soft rule membership;
- `d_r` = anisotropic directional distance;
- `K_r` = geometric compatibility kernel;
- `lambda` = strength of geometric correction.

Loss:

```text
L = L_BCE + L_Dice/Tversky + beta_boundary L_boundary + beta_topo L_clDice + regularizers
```

## 3. Что код уже логирует

Уже есть:

- `architecture_state` в `metrics.json`;
- `az_layer_count`;
- `hybrid_mix_alpha`;
- `metric_min_eig`, `metric_condition`;
- `anisotropy_gap`;
- `rule_usage_entropy_norm`;
- `compat_mass`;
- `final_anisotropy_gap`;
- `test_cldice`, `skeleton_precision`, `skeleton_recall`.

Это хорошо, но есть один критичный вывод по текущему CHASE full run.

## 4. Найденная проблема геометрии

CHASE full result:

| Diagnostic | Value |
|---|---:|
| Dice | 0.6691 |
| clDice | 0.6785 |
| hybrid_mix_alpha_mean | 0.4904 |
| az_metric_condition_mean | 1.00007 |
| az_anisotropy_gap_mean | 0.0000106 |
| az_rule_usage_entropy_norm_mean | 0.8832 |
| compat_mass | 1.0 |

Интерпретация:

- SPD есть: метрика положительная, отрицательных eigenvalues нет.
- Soft rules используются: entropy не 0 и не max-collapse.
- Но anisotropy фактически почти нулевая: condition number около 1.
- Значит fixed directional vectors есть, но kernel почти изотропный из-за почти равных `sigma_u` и `sigma_s`.
- В такой конфигурации reviewer может справедливо спросить: где геометрический вклад, если метрика почти identity?

Это не ломает текущий прогон, но объясняет, почему метод иногда не дает сильного выигрыша.

## 5. Что нужно исправить после текущего обучения

Не трогать уже запущенную очередь. Для следующей версии:

1. Сделать `lambda` мягким: `hybrid_mix_init = 0.10-0.15` для `az_thesis`.
2. Зафиксировать или явно инициализировать anisotropy:
   - целевой `metric_condition` хотя бы `1.2-3.0`, не `1.000`.
   - либо добавить fixed anisotropy ratio для `fixed_cat_map`;
   - либо усилить регуляризацию/инициализацию `sigma_u != sigma_s`.
3. Перейти с `normalize_mode: global` на `per_rule` в диагностическом прогоне, чтобы cross-rule normalization не вымывала геометрию.
4. Добавить в статью ablation:
   - Baseline;
   - Baseline + topology loss;
   - AZ-Thesis w/o topology;
   - AZ-Thesis w/o directional metric;
   - AZ-Thesis full.
5. В финальной таблице не показывать внутренние имена `az_cat`/`az_thesis` как разные методы.

## 6. Что нужно добавить в диагностику

Обязательно для claim:

| Diagnostic | Expected behavior | Why |
|---|---|---|
| metric min eigenvalue | > 0 | SPD / корректная метрика |
| metric condition | > 1, finite | реальная анизотропия |
| rule entropy | between collapse and uniform | rules не декоративные |
| per-rule mass | not all in one rule | нет схлопывания |
| AZ/Conv norm ratio | small but nonzero | branch реально вносит correction |
| compat mass | finite/stable | численная стабильность |
| clDice/skeleton recall | stable or higher | claim про тонкие структуры |

Сейчас нет явного `AZ/Conv norm ratio`. Его надо добавить в posthoc hooks или в следующий training diagnostics.

## 7. Визуализации для статьи

Минимальный набор фигур:

### Figure 1: Method

Схема:

```text
Input -> CNN encoder
      -> Conv branch
      -> AZ geometric correction branch
      -> hybrid mixing lambda
      -> decoder + topology-aware supervision
```

Текст на фигуре:

- soft rule membership;
- anisotropic directional metric;
- geometric compatibility kernel;
- topology-aware loss.

### Figure 2: Prediction comparison

Панели:

```text
image | ground truth | baseline prediction | AZ-Thesis prediction | error map | improvement map
```

Нужно брать не только лучший sample, но и один limitation case.

### Figure 3: Geometry evidence

Панели:

```text
input | rule partition | rule entropy | direction field | anisotropy/gain map | compatibility kernel
```

Цель: показать, что branch действительно смотрит вдоль локальной структуры.

### Figure 4: Skeleton/topology

Панели:

```text
GT skeleton | baseline skeleton | AZ skeleton | false breaks / recovered links
```

Цель: объяснить clDice/skeleton recall, а не только Dice.

### Figure 5: Ablation table/plot

Таблица:

```text
Baseline
Baseline + topology
AZ-Thesis w/o topology
AZ-Thesis w/o directional metric
AZ-Thesis full
```

## 8. Уже существующие scripts для визуализаций

Есть заготовки:

- `scripts/export_drive_article_assets.py`
- `scripts/export_geometry_attention_story.py`
- `scripts/export_arcade_article_assets.py`
- `scripts/build_drawio_article_figures.py`

После завершения checkpoint-ов использовать примерно так:

```powershell
python scripts\export_geometry_attention_story.py --results-dir results\article_full_dataset --run fives_full_azthesis_s42_e20\az_thesis_seed42 --output-dir article_assets\geometry_story --device cuda
```

Для DRIVE/retinal assets:

```powershell
python scripts\export_drive_article_assets.py --results-dir results\article_small_medical\drive_azthesis_s42_e120 --run az_thesis_seed42 --samples 0,1,2 --output-dir article_assets\retinal_examples --device cuda
```

Команды не запускались сейчас, чтобы не занимать GPU.

## 9. Синтетические тесты, которые стоит добавить

Нужны не как main metric, а как geometry sanity proof:

| Test | Input | Expected |
|---|---|---|
| straight line | thin oriented line | direction response aligns with line |
| crossing lines | X-shaped vessels | different rules activate on different branches |
| broken vessel | line with small gap | topology-aware model reduces break |
| blob | round object | no artificial elongation |

Это лучше делать posthoc на trained checkpoints и/или через direct AZConv response, без долгого обучения.

## 10. Главный вывод

Сейчас архитектурная идея правильная, но для статьи нельзя ограничиться таблицей метрик.

Самая важная техническая доработка после текущих прогонов:

```text
сделать анизотропию реально ненулевой и показать это в diagnostics/figures
```

Иначе у нас получится "AZ-Thesis" в названии, но почти isotropic kernel в фактической метрике. Это рецензент может легко атаковать.

## 11. Исправление, внесенное после аудита

Добавлена явная инициализация анизотропии для fixed directional metric:

```text
az_init_anisotropy_gap: 0.35
```

Что это меняет:

| gap | metric condition |
|---:|---:|
| 0.00 | 1.0000 |
| 0.20 | 1.4918 |
| 0.35 | 2.0138 |
| 0.50 | 2.7183 |

Теперь стартовая метрика не является почти identity: при `0.35` она SPD и имеет умеренную анизотропию. Это лучше соответствует claim про directional anisotropic correction.

Изменения:

- `models/azconv.py`: добавлен `init_anisotropy_gap` и asymmetric initialization для `sigma_u/sigma_s`;
- `utils.py`: добавлен config override `az_init_anisotropy_gap`;
- `train.py`: параметр пишется в `metrics.json`;
- benchmark-конфиги получили явное `az_init_anisotropy_gap: 0.35`;
- тесты проверяют, что fixed-cat metric теперь имеет `metric_condition > 1.5`, а ablation может отключить стартовую анизотропию через `init_anisotropy_gap=0.0`.

Проверка:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python -m pytest tests\test_azconv_shapes.py::test_fixed_cat_map_metric_tensor_is_positive_definite_and_logged tests\test_azconv_shapes.py::test_fixed_cat_map_can_disable_initial_anisotropy_for_ablation tests\test_architecture_flow.py::test_architecture_state_reports_active_az_math_after_forward -q
```

Результат: `3 passed`.

Важно: уже запущенный FIVES-процесс был стартован до этой правки, поэтому его `anis_gap=0.0000` в логе ожидаем. Новые процессы, которые стартуют после правки, будут использовать исправленную инициализацию.
