# Результаты, которые имеет смысл использовать

Этот файл — короткая выборка результатов для статьи/доклада. Здесь не перечислена вся история экспериментов.

Технические отрицательные и промежуточные эксперименты остаются в репозитории для воспроизводимости, но их не нужно автоматически переносить в основной текст статьи.

## 1. Проверка данных

Расширенный matched-CCTA cohort:

- 28 пациентов;
- train: 17;
- validation: 5;
- test: 6.

Все 28 оригинальных ImageCAS CT совпали со своими ImageCAS-X annotations по:

- shape;
- spacing;
- affine.

Итог:

`28/28 PASS`

Источник:

`results/ccta_graph_lira_safe_repair/2026-09-21/real_ct28_alignment.csv`

## 2. Главный локальный эксперимент

Controlled geometry-matched relation task:

- train: 379 positive + 379 negative;
- validation: 134 + 134;
- test: 167 + 167.

Threshold выбирается только на validation при FPR <= 5%.

### Held-out test

| метод | AUROC | Recall | FPR | Precision |
|---|---:|---:|---:|---:|
| geometry HGB | 0.9685 | 37.72% | 1.20% | 96.92% |
| radial 2.5-D CT | 0.9440 | 68.86% | 4.19% | 94.26% |
| **geometry + radial CT** | **0.9847** | **82.63%** | **1.80%** | **97.87%** |

Для combined модели:

- TP = 138;
- FP = 3;
- FN = 29;
- TN = 164.

Источник:

`results/ccta_graph_lira_safe_repair/2026-09-21/real_ct28_headline_test.csv`

## 3. Patient-cluster uncertainty

Для geometry + radial CT:

- AUROC median 0.9847;
- 95% interval 0.9676–0.9957;
- recall median 82.98%;
- 95% interval 69.74–92.17%;
- FPR median 1.72%;
- 95% interval 0–4.43%;
- precision median 97.96%;
- 95% interval 94.90–100%.

Источник:

`results/ccta_graph_lira_safe_repair/2026-09-21/real_ct28_patient_cluster_bootstrap_ci.csv`

## 4. Paired improvement over strong geometry baseline

Geometry + radial CT minus geometry HGB:

- recall delta median: +44.71 percentage points;
- 95% interval: +32.62 to +60.00 pp;
- FPR delta median: +0.61 pp;
- 95% interval: -1.29 to +2.56 pp.

Это один из самых важных результатов текущего этапа.

Источник:

`results/ccta_graph_lira_safe_repair/2026-09-21/real_ct28_paired_patient_cluster_bootstrap_ci.csv`

## 5. Risk / coverage

Combined geometry+CT на held-out test:

- validation budget 0% -> recall 23.95%, observed test FPR 0%;
- 1% -> recall 57.49%, FPR 0.60%;
- 2% -> recall 58.08%, FPR 0.60%;
- 5% -> recall 82.63%, FPR 1.80%;
- 10% -> recall 97.60%, FPR 8.38%.

Это хороший материал для risk-coverage figure.

Источник:

`results/ccta_graph_lira_safe_repair/2026-09-21/real_ct28_multibudget_risk_coverage.csv`

## 6. Сложная анатомия

Для geometry + radial CT:

| branch | accepted / n | rate |
|---|---:|---:|
| D2 | 8/8 | 100% |
| OM1 | 9/9 | 100% |
| OM2 | 13/13 | 100% |
| D1 | 21/22 | 95.5% |
| LCX | 18/21 | 85.7% |
| IM | 9/11 | 81.8% |
| R-PLA | 29/36 | 80.6% |
| LAD | 7/11 | 63.6% |

Нужно указывать n рядом с процентами.

Источник:

`results/ccta_graph_lira_safe_repair/2026-09-21/real_ct28_anatomy_subgroups.csv`

## 7. Что лучше не тащить в основной Results сейчас

Не надо перегружать первую версию статьи:

- ранними 4-case BDMAP image-only экспериментами;
- большим количеством слабых вариантов feature engineering;
- sequence Transformer над сильно сжатыми статистиками;
- промежуточными mask-veto вариантами;
- всеми неудачными threshold grids.

Их роль была исследовательская: они помогли понять, где находится bottleneck.

Если понадобится раздел ablation / supplementary material, часть этих экспериментов можно вернуть.

## 8. Что обязательно добавить до финальной статьи

Сейчас отсутствует последний ключевой результат:

**CT-conditioned Graph-LIRA end-to-end structural evaluation.**

До него нельзя писать в Conclusion, что вся graph-repair система уже улучшена.

Следующий Results-блок должен сравнить:

- canonical geometry Graph-LIRA;
- CT-conditioned Graph-LIRA;

при неизменных:

- `tau = 0.85`;
- consistency `= 0.60`.

И уже там оценить:

- repair-needed exact;
- false structural repair;
- incomplete / abstain;
- coverage;
- false among accepted;
- exact among accepted.
