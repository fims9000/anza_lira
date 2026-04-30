# Финальный пакет визуализации для 3 статьи (SpaceNet3 / Global Roads)

Дата обновления: `2026-04-29`  
Статус: `готово для вставки в текст статьи`

---

## 1) Что зафиксировано

Мы зафиксировали **единый финальный формат фигуры 2x2** для интерпретации геометрии AZ-модели:

1. `Input + Ground Truth`
2. `Baseline vs AZ error difference vs GT`
3. `AZ Orientation Axis (model theta map)`
4. `Anisotropy Strength Map`

Важно: оси в панели 3 строятся из **модельного направления** (`theta_map`, ось без полярности), а не рисуются вручную.

---

## 2) Финальные картинки (разные кейсы)

Папка с итоговыми PNG:

`results/a3_final_package/final_article3/figures`

### Основные positive-кейсы (рекомендуются в статью)

- `geometry_clean_global_roads_spacenet3_paris_img0417.png`  
- `geometry_clean_global_roads_spacenet3_paris_img0396.png`  
- `geometry_clean_global_roads_spacenet3_paris_img0027.png`  
- `geometry_clean_global_roads_spacenet3_paris_img0212.png`  

### Нейтральный кейс (можно в приложение)

- `geometry_clean_global_roads_spacenet3_paris_img0175.png`

### Failure-case (обязательно иметь 1 пример в тексте/appendix)

- `geometry_clean_global_roads_spacenet3_paris_img0453.png`

---

## 3) Таблица по выбранным кейсам (tile-level)

Пороги:
- AZ threshold: `0.80`
- Baseline threshold: `0.70`

| sample_index | sample_id | Dice(AZ) | Dice(Base) | Delta Dice | Delta IoU | Delta Precision | Delta Recall | Роль в статье |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 30 | spacenet3_paris_img0417 | 0.6240 | 0.3359 | +0.2881 | +0.2517 | +0.2682 | +0.3113 | main positive |
| 28 | spacenet3_paris_img0396 | 0.6922 | 0.5590 | +0.1332 | +0.1413 | +0.1034 | +0.1758 | main positive |
| 3  | spacenet3_paris_img0027 | 0.4211 | 0.0000 | +0.4211 | +0.2667 | +0.2752 | +0.8953 | strong recovery case |
| 16 | spacenet3_paris_img0212 | 0.3003 | 0.0000 | +0.3003 | +0.1767 | +0.6963 | +0.1914 | strong recovery case |
| 13 | spacenet3_paris_img0175 | 0.6054 | 0.6124 | -0.0070 | -0.0072 | -0.0099 | -0.0047 | neutral |
| 35 | spacenet3_paris_img0453 | 0.3746 | 0.6761 | -0.3015 | -0.2803 | -0.0685 | -0.3737 | failure case |

Примечание: это **tile-level** сравнение для визуальных примеров; оно нужно для иллюстрации механики локальных эффектов, а не как единственная итоговая метрика по датасету.

---

## 4) Что именно означает каждая панель

### Панель A: Input + GT
- Фон: исходный спутниковый тайл.
- Полупрозрачная маска: GT-дороги.

### Панель B: Baseline vs AZ error difference vs GT
Цвета:
- `green`: AZ исправил FN baseline (нашёл пропущенную дорогу).
- `blue`: AZ убрал FP baseline.
- `orange`: AZ добавил новый FP.
- `red`: AZ дал новый FN.
- `light gray`: оба метода предсказали дорогу.

Это **error-centric** карта: она читается рецензентом сразу как “где стало лучше/хуже относительно GT”.

### Панель C: AZ Orientation Axis
- Белые короткие оси = локальная ориентация, извлечённая из модели.
- Это **ось направления** (mod pi), а не стрелка “вперёд-назад”.
- Оси показываются по объектной support-области (предсказание/GT), чтобы убрать шум фона.

### Панель D: Anisotropy Strength Map
- Сине-оранжевая шкала: от слабой к сильной анизотропии.
- Отображает, где геометрический механизм AZ реально активен сильнее.

---

## 5) Готовые подписи для статьи

### Подпись для основного рисунка (короткая)
`Model-native geometry interpretation on SpaceNet3 tile: input+GT, baseline-vs-AZ error difference, AZ orientation axis map, and anisotropy strength map.`

### Подпись расширенная (для 1-й большой фигуры)
`Error-centric comparison between baseline and AZ on SpaceNet3. Green marks baseline false negatives corrected by AZ, blue marks baseline false positives removed by AZ, orange denotes new AZ false positives, and red denotes new AZ false negatives. Orientation axes are derived from the model theta map (axis, not directional flow), while anisotropy strength indicates where geometry-aware aggregation is most active.`

---

## 6) Что вставлять в основной текст (минимум)

Рекомендуемый комплект в paper body:

1. `geometry_clean_global_roads_spacenet3_paris_img0417.png` (главный positive-case)
2. `geometry_clean_global_roads_spacenet3_paris_img0396.png` (второй positive-case)
3. `geometry_clean_global_roads_spacenet3_paris_img0453.png` (один failure-case)

Остальные (img0027, img0212, img0175) — в appendix/supplementary.

---

## 7) Как воспроизвести картинки (команда)

Шаблон:

`python scripts/export_geometry_clean_article_figure.py --results-dir results --run article3_spacenet_sprint_v3_recover --baseline-run article3_spacenet_sprint_v3_baseline --sample-index <IDX> --output-dir results/a3_final_package/final_article3/figures --device cpu`

Где `<IDX>`: `3, 13, 16, 28, 30, 35`.

---

## 8) Что важно не перепутать в тексте статьи

1. Это визуализация **оси ориентации**, а не векторного потока.
2. Панель B — это сравнение ошибок относительно GT, а не просто разность масок.
3. В основном тексте показать не только лучший случай, но и один failure-case (это усиливает доверие рецензента).
4. Отдельно указать, что глобальные датасетные метрики приводятся в основной таблице результатов, а этот блок — интерпретационный.

---

## 9) Короткий готовый абзац (можно вставить почти без правок)

`To make the geometric behavior of AZ interpretable, we use a four-panel visualization protocol: (i) input with ground truth overlay, (ii) error-centric baseline-vs-AZ difference map, (iii) model-native orientation axis map, and (iv) anisotropy strength map. The orientation panel is computed from the model theta field (axis modulo pi), while anisotropy strength is estimated from the local sigma_u/sigma_s geometry term. This allows us to connect local error corrections to explicit internal geometric states of the model rather than relying on qualitative mask overlays only.`

---

## 10) Расширенный матблок (готово для Methods / Appendix)

Ниже формулы, которые можно вставлять в раздел “Geometry State Extraction”.

### 10.1 Rule-aggregated axis estimator

Для каждого пикселя \(p\):

\[
C(p)=\frac{\sum_{r=1}^R \mu_r(p)\cos\bigl(2\theta_r(p)\bigr)}{\sum_{r=1}^R \mu_r(p)+\varepsilon},
\]
\[
S(p)=\frac{\sum_{r=1}^R \mu_r(p)\sin\bigl(2\theta_r(p)\bigr)}{\sum_{r=1}^R \mu_r(p)+\varepsilon},
\]
\[
\theta_{\text{axis}}(p)=\frac{1}{2}\operatorname{atan2}\bigl(S(p),C(p)\bigr).
\]

Комментарий для текста: удвоенный угол убирает двусмысленность \(\theta\sim\theta+\pi\), поэтому это корректная оценка **оси**.

### 10.2 Axis consistency

\[
\rho(p)=\sqrt{C(p)^2+S(p)^2}, \quad 0\le \rho(p)\le 1.
\]

- \(\rho\to 1\): режимы согласованы по оси;
- \(\rho\to 0\): локально конфликтные/шумные направления.

### 10.3 Confidence gating

\[
c_{\max}(p)=\max_r \mu_r(p), \quad c(p)=c_{\max}(p)\cdot \rho(p).
\]

Именно \(c(p)\) применяется как “доверие” для осевых глифов и для стабилизации strength-map.

### 10.4 Anisotropy strength

\[
g_r(p)=\tanh\!\left(\log\frac{\sigma_{u,r}(p)}{\sigma_{s,r}(p)+\varepsilon}\right),
\]
\[
g(p)=\frac{\sum_r \mu_r(p)g_r(p)}{\sum_r \mu_r(p)+\varepsilon}\cdot c(p).
\]

\(|g(p)|\) используется как финальная карта интенсивности анизотропии.

### 10.5 Robust normalization for visualization

Для объекта \(M_{\text{obj}}\) берём квантили \(q_{10}, q_{95}\):
\[
\tilde g(p)=\operatorname{clip}\!\left(\frac{|g(p)|-q_{10}}{q_{95}-q_{10}+\varepsilon},0,1\right).
\]

Это убирает доминирование выбросов и делает карты сопоставимыми между кейсами.

---

## 11) Готовый текст на английском (копипаст в IEEE draft)

### 11.1 Methods paragraph (short)

`We extract model-native geometric states from AZ layers by aggregating rule-wise orientation and anisotropy parameters. Because the local kernel is symmetric with respect to direction sign, we estimate undirected orientation axes via doubled-angle averaging. Specifically, we compute weighted cos(2theta) and sin(2theta) moments over fuzzy memberships and recover axis orientation with a half-angle atan2 mapping. We further compute an axis-consistency score and use it to confidence-gate local anisotropy strength.`

### 11.2 Visualization protocol paragraph

`Our interpretation protocol uses four synchronized panels: input with ground truth, error-centric baseline-vs-AZ map, model-axis overlay, and anisotropy-strength map. Error-centric decomposition explicitly separates corrected false negatives, removed false positives, newly introduced false positives, and newly introduced false negatives. This links qualitative improvements to measurable geometric states instead of relying on mask overlays only.`

### 11.3 Discussion paragraph (balanced claim)

`The contribution of this study is not an accuracy-only benchmark, but a reproducible geometric interpretation pipeline for anisotropic fuzzy aggregation. The proposed diagnostics reveal where geometry-aware aggregation is active, where rule agreement is low, and how these states relate to local correction or degradation patterns.`

### 11.4 Limitation paragraph

`The current formulation estimates orientation axes (modulo pi) rather than directed flow polarity. Therefore, the method explains alignment and anisotropy activity but does not model forward/backward transport semantics. Extending the operator to polarity-aware directional fields is left for future work.`

---

## 12) Q/A блок для рецензента (готовые ответы)

### Q1: “Почему стрелки не всегда в одну физическую сторону?”
**Ответ:** потому что модель в текущей формулировке оценивает ось (\(\theta\sim\theta+\pi\)), а не ориентированный поток; это математическое свойство квадратичной геометрической формы ядра.

### Q2: “Это не ручная отрисовка?”
**Ответ:** нет, ось извлекается из внутренних тензоров модели (`mu_map`, `theta_map`, `sigma_u_map`, `sigma_s_map`) и выводится детерминированным скриптом.

### Q3: “Почему не показываете только лучшие примеры?”
**Ответ:** в пакет включён и positive, и neutral, и failure-case; это специально сделано для честной интерпретации режимов работы.

### Q4: “Как понять, что карта силы не декоративная?”
**Ответ:** используется формализованный показатель \(g(p)\), confidence-gating и робастная нормировка внутри объектной области; значения связаны с параметрами \(\sigma_u,\sigma_s\) модели.

### Q5: “Чем 3-я статья отличается от 1-й и 2-й?”
**Ответ:** 1-я = теория оператора, 2-я = алгоритм и метрики, 3-я = интерпретируемость и диагностика внутренней геометрии модели.

---

## 13) Шаблоны таблиц (вставить и заполнить)

### 13.1 Таблица diagnostics

| Run | axis_consistency_mean | rule_entropy_mean | anisotropy_gap | direction_diversity | note |
|---|---:|---:|---:|---:|---|
| baseline (`article3_spacenet_sprint_v3_baseline`) | n/a | n/a | n/a | n/a | no AZ states in baseline model |
| az_thesis (`article3_spacenet_sprint_v3_recover`) | 0.9999999939 (orientation_resultant_r) | 0.956845 | 0.176166 | 0.0000000138 | primary run |
| az_thesis probe (`article3_spacenet_v3_dirlearn_probe_s42_e10`) | 0.6376703156 (orientation_resultant_r) | 0.957746 | 0.684039 | 0.1613856432 | shorter probe run, higher directional spread |

Источник:  
`results/article3_spacenet_sprint_v3_recover/metrics.json` + `direction_diversity_summary_recheck_after_fallback.json`,  
`results/article3_spacenet_v3_dirlearn_probe_s42_e10/metrics.json` + `direction_diversity_summary.json`.

### 13.2 Таблица case-study

| sample_id | Delta Dice | Delta IoU | fixFN px | removeFP px | addFP px | newFN px | case type |
|---|---:|---:|---:|---:|---:|---:|---|
| spacenet3_paris_img0417 | +0.2881 | +0.2517 | 231 | 417 | 195 | 5 | positive |
| spacenet3_paris_img0396 | +0.1332 | +0.1413 | 295 | 190 | 65 | 61 | positive |
| spacenet3_paris_img0027 | +0.4211 | +0.2667 | 804 | 0 | 2108 | 0 | positive |
| spacenet3_paris_img0212 | +0.3003 | +0.1767 | 94 | 0 | 41 | 0 | positive |
| spacenet3_paris_img0175 | -0.0070 | -0.0072 | 1357 | 1526 | 1644 | 1408 | neutral |
| spacenet3_paris_img0453 | -0.3015 | -0.2803 | 243 | 1053 | 382 | 2603 | failure |

Источник: вычислено скриптом из предсказаний `article3_spacenet_sprint_v3_recover` vs `article3_spacenet_sprint_v3_baseline` на selected sample indices `30, 28, 3, 16, 13, 35`.

### 13.3 Таблица “interpretation vs behavior”

| Region | high anisotropy? | high axis consistency? | observed effect |
|---|---|---|---|
| main road trunk (img0417/img0396) | high | high | continuity recovery, long connected segments restored |
| dense crossings / urban junctions (img0175) | medium-high | high globally, but mixed locally | simultaneous FP cleanup and new local FP/FN around intersections |
| weak-contrast side roads (img0027/img0212) | medium | medium-high | strong FN correction, but possible FP growth on texture-like background |

---

## 14) Безопасные формулировки claims (чтобы не оверклеймить)

### Можно писать
- “improves interpretability of model behavior”
- “links local corrections to internal geometric states”
- “shows robust geometry-aware behavior in selected regimes”
- “provides a reproducible diagnostics protocol”

### Нежелательно писать
- “outperforms all methods in all settings”
- “universal superiority”
- “directional flow is recovered” (если полярность не моделируется)

---

## 15) Готовый skeleton разделов (тезисно)

### Introduction (4-6 предложений)
1. Проблема thin-structure сегментации.
2. Почему одного Dice мало.
3. Нужна интерпретация внутренней геометрии.
4. Наш вклад: model-native diagnostics.

### Methods
1. Извлечение состояний AZ.
2. Axis-estimator через doubled-angle.
3. Anisotropy strength и confidence gating.
4. Error-centric decomposition.

### Results
1. 2 positive + 1 neutral + 1 failure-case.
2. Короткая таблица case-level delta.
3. Таблица diagnostics.

### Discussion
1. Где геометрия реально помогает.
2. Где появляются ошибки и почему.
3. Ограничения (ось vs поток).

### Conclusion
1. Что именно добавляет 3-я статья к пакету работ.
2. Как это использовать в практике и в будущих моделях.

---

## 16) Что можно сразу копировать в “Contributions” (bullet-ready)

`Our contributions are threefold:`
1. `A reproducible model-native protocol to extract orientation axis, anisotropy strength, and confidence from AZ layers.`
2. `An error-centric decomposition that links baseline-vs-AZ corrections to internal geometric states.`
3. `A cross-case analysis (positive, neutral, and failure examples) demonstrating when geometry-aware fuzzy aggregation is beneficial and when it degrades.`

---

## 17) Практический мини-план написания (быстрый)

1. Взять из этого файла секции 10-12 и вставить в черновик.
2. Подставить 3-4 ключевые картинки из секции 2.
3. Заполнить таблицы из секции 13 реальными числами.
4. Проверить claims по секции 14.
5. Сжать до лимита страниц, убрав дубли.

---

## 18) Финальный “one-page summary” для себя перед отправкой

- Что нового: не метрики, а геометрическая интерпретация модели.
- Что главное на рисунках: error map + axis + anisotropy strength.
- Что честно говорим: есть и failure-cases.
- Что защищаем перед рецензентом: воспроизводимость, математика, не-overclaim.

---

## 19) Major Revision hardening: что обязательно учесть в 3-й статье

Это блок именно под жёсткого рецензента. Его задача — заранее закрыть самые очевидные претензии, не обещая того, чего у нас ещё нет.

### 19.1 Слабый baseline

Что написать честно:

`The present study uses a matched U-Net baseline to isolate the effect of the AZ local aggregation block. We acknowledge that comparison with Attention U-Net, U-Net++, nnU-Net, and transformer-based segmentation models is required for a complete state-of-the-art evaluation and leave it for the extended version.`

Почему так нормально:
- мы не притворяемся, что победили все архитектуры;
- мы объясняем, что текущая цель — изолировать локальный блок.

### 19.2 Roads_HF не ставить как главный quantitative win

Roads_HF лучше оставить как qualitative/sanity-check:

`Roads_HF is not used as primary quantitative evidence because the baseline is already near saturation. We retain it only as an auxiliary qualitative case for geometry visualization.`

Это защищает от вопроса: “зачем показывать датасет, где нет прироста”.

### 19.3 clDice на HRF упал — не скрывать

Готовый абзац:

`On HRF_SegPlus, the proposed model improves Dice, IoU, and Precision, but clDice decreases. This indicates that the current configuration may over-suppress faint vessel continuations while reducing false positives. We therefore interpret the HRF result as a precision-oriented operating regime rather than a topology-optimal one. Incorporating clDice or skeleton-recall loss is a natural next step.`

### 19.4 Ablation, если нет времени переобучать

В основной текст вставить как analytical ablation:

- `without fuzzy`: остаётся только геометрическая анизотропия, но нет подавления weak-agreement соседей;
- `without anisotropy`: \(\sigma_u=\sigma_s\), то есть fuzzy isotropic aggregation;
- `full AZ`: совместное действие membership + directional metric.

Важно: не писать “we experimentally prove”, если нет полного retraining. Писать “analytically, the components correspond to...”.

### 19.5 Статистика

Минимально честная формулировка:

`Because the current test sets are modest, we report per-image examples and explicitly mark the study as proof-of-concept. Full confidence intervals and multi-seed significance analysis are planned for the extended version.`

Если успеем посчитать per-image std, добавить:

`Per-image standard deviations are provided in supplementary material.`

### 19.6 Reproducibility

В статье обязательно вставить:

`The implementation and figure-generation scripts are available at https://github.com/fims9000/anza_lira.`

И отдельно:

`All geometry figures are generated from model checkpoints using deterministic scripts, not manual drawing.`

### 19.7 Ссылка на Аносова

В 3-й статье не тянуть Аносова в основной текст. Лучше опираться на:

- steerable filters;
- anisotropic diffusion;
- U-Net;
- retinal vessel segmentation reviews.

Формулировка:

`The directional part of the layer follows the general intuition of steerable and anisotropic filtering, but the proposed mechanism combines it with fuzzy rule memberships inside a trainable segmentation model.`

---

## 20) Что реально успеть за короткое время

Приоритеты:

1. Убрать Roads_HF из главной quantitative table.
2. Добавить честный Limitations paragraph.
3. Объяснить clDice drop на HRF.
4. Добавить analytical ablation paragraph.
5. Добавить GitHub/reproducibility sentence.
6. Поставить одну strong positive figure + одну failure figure.

Этого достаточно, чтобы статья выглядела не как “сырой proof-of-concept”, а как честная короткая conference paper с понятным scope.
