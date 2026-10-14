# CT28 Graph-LIRA ablations

Статус: development train/validation. Held-out test не открывался, path stage
не выполнялся. Основной baseline — clean local Logistic.

Для всех сравнений зафиксированы patient-level OOF, 368 validation-сцен,
candidate pools, relation HGB, endpoint-disjoint compatibility, `tau=0.85`,
consistency gate `0.60` и 15 repeats с `0.8 мм + 20°`.

## Clean local HGB

В `local_hgb/` только шесть local LogisticRegression-моделей заменены на
HistGradientBoostingClassifier.

### Local validation

| Candidate | Evidence | Logistic AUROC | HGB AUROC | Logistic AUPRC | HGB AUPRC |
|---|---|---:|---:|---:|---:|
| PAIR | geometry | 0.6740 | 0.9770 | 0.1088 | 0.7174 |
| PAIR | CT | 0.8051 | 0.9111 | 0.2051 | 0.4524 |
| PAIR | geometry+CT | 0.8510 | 0.9910 | 0.2404 | 0.9012 |
| JUNCTION | geometry | 0.9492 | 0.9909 | 0.2939 | 0.7182 |
| JUNCTION | CT | 0.9069 | 0.8954 | 0.3127 | 0.3086 |
| JUNCTION | geometry+CT | 0.9536 | 0.9937 | 0.4176 | 0.8462 |

### Relation validation

| Local family | Representation | Raw accuracy | Accuracy at tau | Coverage |
|---|---|---:|---:|---:|
| Logistic | geometry-only | 88.86% | 84.51% | 84.51% |
| Logistic | geometry+CT | 83.97% | 80.71% | 76.90% |
| Logistic | CT-conditioned | 89.40% | 83.15% | 83.42% |
| HGB | geometry-only | 92.12% | 90.22% | 91.58% |
| HGB | geometry+CT | **92.93%** | **92.39%** | **92.12%** |
| HGB | CT-conditioned | 92.66% | 91.58% | 91.85% |

### Structural validation

| Local family | Representation | Exact before | False before | Exact after | False after | Coverage |
|---|---|---:|---:|---:|---:|---:|
| Logistic | geometry-only | 29.61% | 28.53% | 1.46% | 0.54% | 1.36% |
| Logistic | CT-conditioned | 29.13% | 27.99% | 0.97% | 0.00% | 0.54% |
| HGB | geometry-only | **77.67%** | 6.79% | 0.00% | 0.00% | 0.00% |
| HGB | CT-conditioned | **79.13%** | 7.07% | 0.00% | 0.00% | 0.00% |

HGB улучшает candidate identity до consistency, но максимальная consistency
остаётся ниже gate: `0.5333` для geometry-only и `0.4667` для
CT-conditioned. После gate coverage равен нулю. Поэтому дальнейшая работа
использует LogisticRegression; HGB сохраняется как абляция.

## Single-stress 30° + 1 мм

В `stress30_jitter1/` clean scene заменена одной детерминированной реализацией
с position noise `1 мм` и tangent noise `30°`. Geometry и CT features,
local models и relation heads пересчитаны. Consistency проверена 15
дополнительными perturbations `0.8 мм + 20°`.

### Relation validation на stressed scenes

| Local family | Representation | Raw accuracy | Accuracy at tau | Coverage |
|---|---|---:|---:|---:|
| Logistic | geometry-only | 79.62% | 69.57% | 64.13% |
| Logistic | geometry+CT | 76.63% | 72.28% | 63.59% |
| Logistic | CT-conditioned | 76.90% | 71.47% | 65.49% |
| HGB | geometry-only | 75.27% | 69.29% | 63.86% |
| HGB | geometry+CT | 75.82% | 67.12% | 61.68% |
| HGB | CT-conditioned | 74.18% | 69.29% | 62.77% |

### Structural validation

| Base/local | Representation | Exact before | False before | Exact after | False after | Coverage | Accepted | False accepted |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clean Logistic | geometry-only | 29.61% | 28.53% | 1.46% | 0.54% | 1.36% | 5 | 40.00% |
| clean Logistic | CT-conditioned | 29.13% | 27.99% | 0.97% | 0.00% | 0.54% | 2 | 0.00% |
| stressed Logistic | geometry-only | 22.33% | 19.02% | 6.80% | 3.26% | 7.07% | 26 | 46.15% |
| stressed Logistic | CT-conditioned | 23.30% | 19.57% | 7.28% | 3.26% | 7.34% | 27 | 44.44% |
| stressed HGB | geometry-only | 26.70% | 16.58% | 4.85% | 1.90% | 4.62% | 17 | 41.18% |
| stressed HGB | CT-conditioned | 24.76% | 17.93% | 5.83% | 1.63% | 4.89% | 18 | 33.33% |

Single-stress повышает consistency coverage, но текущий вариант одновременно
ухудшает relation accuracy и safety. Результат показывает, что perturbations
влияют на устойчивость, и служит отправной точкой для следующих экспериментов
с perturbation training.

