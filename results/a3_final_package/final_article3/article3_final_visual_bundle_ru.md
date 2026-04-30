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

---

## 21) Таблица “замечание рецензента -> что исправлено”

| Замечание | Что сделано в материалах | Статус |
|---|---|---|
| Слабый baseline: только U-Net | В draft добавлен честный scope: U-Net используется как matched baseline для изоляции AZ-блока; Attention U-Net, U-Net++, nnU-Net и transformer-модель вынесены в Future work | закрыто текстом / future work |
| Недостаточно медицинских датасетов | HRF оставлен как medical case, но явно указано, что DRIVE/CHASE/STARE нужны для extended version | закрыто как limitation |
| Нет ablation study | Добавлен analytical ablation: without fuzzy, without anisotropy, full AZ; полный retraining grid заявлен как required extension | частично закрыто |
| clDice на HRF упал | Добавлен отдельный абзац: AZ улучшает Precision/Dice, но может подавлять faint vessel continuations; решение — clDice/skeleton loss | закрыто объяснением |
| Почему такое число режимов R | Добавлен пункт про regime-count scalability и необходимость sweep \(R=1,2,4,8,16\) | закрыто как future work |
| Нет статистической значимости | Для SpaceNet3 посчитана per-image статистика mean ± std по 39 test tiles; добавлены CSV/JSON supplementary artifacts | частично закрыто |
| Аносов выглядит чужеродно | В 3-й статье рекомендовано не использовать Аносова; вместо этого опираться на steerable filters / anisotropic diffusion | закрыто рекомендацией |
| Нет reproducibility | Добавлена ссылка на GitHub и команда генерации фигур; README приведён в публичный вид | закрыто |
| Roads_HF бесполезен в таблице | Roads_HF убран из main quantitative table, оставлен как qualitative/sanity-check | закрыто |
| Precision вырос, Recall упал | В main table есть Precision/Recall; в Discussion объяснён trade-off | закрыто |

---

## 22) Per-image statistics для SpaceNet3 (готово в supplementary)

Файлы:

- `results/a3_final_package/final_article3/spacenet_v3_per_image_metrics.csv`
- `results/a3_final_package/final_article3/spacenet_v3_per_image_summary.json`

Порог:
- baseline: `0.70`
- AZ: `0.80`

Количество test tiles: `39`.

| Metric | Baseline mean ± std | AZ mean ± std | Delta mean ± std |
|---|---:|---:|---:|
| Dice | 0.4931 ± 0.2246 | 0.5248 ± 0.2001 | +0.0318 ± 0.1238 |
| IoU | 0.3538 ± 0.1854 | 0.3790 ± 0.1776 | +0.0252 ± 0.0987 |
| Precision | 0.4958 ± 0.2112 | 0.5071 ± 0.2055 | +0.0113 ± 0.1618 |
| Recall | 0.5112 ± 0.2598 | 0.5956 ± 0.2317 | +0.0844 ± 0.1833 |

Готовый абзац:

`For the reproducible SpaceNet3 split, we additionally computed per-image statistics over 39 test tiles. The AZ variant improves mean Dice, IoU, Precision, and Recall, but the standard deviation of the delta remains substantial. Therefore, we interpret these results as a promising trend rather than a statistically final claim.`

---

## 23) Минимальный “response to reviewer” для письма/формы

`We thank the reviewer for the constructive comments. We revised the paper to clarify that the current work is a geometry-interpretability and proof-of-concept study rather than a complete state-of-the-art benchmark. We removed Roads_HF from the main quantitative table and retained it only as a qualitative saturated-baseline sanity check. We added a complete metric table with Dice, IoU, Precision, Recall, and clDice; discussed the HRF clDice decrease and its likely reason; added an analytical component ablation; provided per-image SpaceNet3 statistics; added an explicit limitations paragraph and future work plan covering stronger baselines, additional medical datasets, regime-count sensitivity, and topology-aware losses. The implementation and deterministic figure-generation scripts are available at https://github.com/fims9000/anza_lira.`

---

## 24) Что осталось не закрыто экспериментально

Честно:

1. Attention U-Net и U-Net++ уже добавлены/подключены, но их надо реально дообучить в reviewer-pack.
2. Нет nnU-Net / transformer baseline.
3. Нет специализированного vessel baseline уровня CS-Net/IterNet.
4. Нет retrained ablation grid по fuzzy-only / anisotropy-only / full для всех medical datasets.
5. Нет завершённого sweep по \(R=1,2,4,8,16\).
6. Нет multi-seed confidence intervals для всех датасетов.
7. HRF/Roads reproducibility artifacts в public repo надо либо добавить, либо оставить эти числа как local/supplementary.

Если статья короткая conference paper, это можно оставить как limitations. Если хотим journal/Q1 — это всё придётся реально дообучать.

---

## 25) Что теперь реально добавлено в код для закрытия рецензента

### 25.1 Stronger baselines

В репозитории теперь доступны:

- `baseline` — vanilla U-Net;
- `attention_unet` — Attention U-Net;
- `unet_plus_plus` — lightweight U-Net++;
- `az_no_fuzzy` — ablation без fuzzy-factor;
- `az_no_aniso` — ablation без anisotropy;
- `az_thesis` — полный предлагаемый вариант.

Проверка сборки выполнена: все варианты возвращают segmentation logits размера `(B, 1, H, W)`.
Для `unet_plus_plus` дополнительно проверен synthetic forward/backward pass.

### 25.2 Medical reviewer pack

Команда:

`powershell -ExecutionPolicy Bypass -File scripts/run_article3_reviewer_medical_pack.ps1 -Device cuda -Seeds "41,42,43" -Epochs 60`

Что запускает:

- DRIVE;
- CHASE_DB1;
- HRF_SegPlus;
- `baseline, attention_unet, unet_plus_plus, az_no_fuzzy, az_no_aniso, az_thesis`;
- seeds `41,42,43`;
- одинаковую loss/threshold policy через `configs/reviewer_drive_lossmatched_overrides.yaml`.

Это закрывает:

- слабый baseline;
- medical datasets;
- ablation;
- multi-seed statistics.

### 25.3 Regime-count sweep

Команда:

`powershell -ExecutionPolicy Bypass -File scripts/run_article3_regime_count_sweep.ps1 -Config configs/drive_benchmark.yaml -Device cuda -Seed "42" -Epochs 40`

Что запускает:

- `az_thesis`;
- \(R=1,2,4,8,16\);
- один выбранный dataset/config;
- фиксированный seed.

Это закрывает пункт рецензента “почему столько режимов”.

### 25.4 Что всё ещё нельзя честно заявлять

Пока не запускали reviewer-pack до конца, в статье нельзя писать:

- “we outperform Attention U-Net / U-Net++”;
- “we prove the ablation experimentally”;
- “we studied R sensitivity”.

Можно писать:

- “we provide the reviewer-oriented experiment protocol”;
- “the current conference version includes analytical ablation and reproducible scripts”;
- “full retraining grid is planned / in progress”.

Локальная заметка: `train.py` smoke-run на Windows может падать на уровне runtime/OpenMP без Python traceback. Сама архитектура `unet_plus_plus` проверена отдельно через forward/backward; полный reviewer-pack лучше запускать в стабильном CUDA/conda окружении.

---

## 26) Финальный reviewer-pack от 2026-04-30: фактические результаты

Источник:

- DRIVE: `results/article3_reviewer_drive_ms_20260430_142712/all_metrics.json`
- CHASE_DB1: `results/article3_reviewer_chase_ms_20260430_142712/all_metrics.json`
- HRF_SegPlus: `results/article3_reviewer_hrf_ms_20260430_142712/all_metrics.json`
- Лог запуска: `logs/article3_reviewer_medical_pack/reviewer_medical_pack_20260430_142712.out.log`
- Команда: `powershell -ExecutionPolicy Bypass -File scripts/run_article3_reviewer_medical_pack.ps1 -Device cuda -Seeds "41,42,43" -Epochs 60`

Протокол:

- datasets: DRIVE, CHASE_DB1, HRF_SegPlus;
- variants: `baseline`, `attention_unet`, `unet_plus_plus`, `az_no_fuzzy`, `az_no_aniso`, `az_thesis`;
- seeds: `41,42,43`;
- epochs: `60`;
- threshold policy: validation sweep по Dice;
- loss policy: BCE + Dice/Tversky-style overlap + boundary/topology auxiliary terms по общему reviewer config;
- baseline architecture: standard U-Net;
- stronger baselines: Attention U-Net и U-Net++;
- ablation variants:
  - `az_no_fuzzy`: геометрическая часть AZ есть, fuzzy-согласование отключено;
  - `az_no_aniso`: fuzzy/локальный AZ-механизм есть, но анизотропия отключена;
  - `az_thesis`: полный вариант.

### 26.1 DRIVE

| Variant | Dice | IoU | clDice | Precision | Recall | Skel Prec | Skel Rec | Bal Acc | Aniso gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.7646 +- 0.0286 | 0.6198 +- 0.0370 | 0.7499 +- 0.0281 | 0.7753 +- 0.0322 | 0.7543 +- 0.0252 | 0.8291 +- 0.0165 | 0.6850 +- 0.0378 | 0.8617 +- 0.0151 | 0.0000 +- 0.0000 |
| attention_unet | 0.7668 +- 0.0239 | 0.6224 +- 0.0311 | 0.7568 +- 0.0235 | 0.7716 +- 0.0113 | 0.7625 +- 0.0363 | 0.8368 +- 0.0051 | 0.6919 +- 0.0395 | 0.8654 +- 0.0185 | 0.0000 +- 0.0000 |
| unet_plus_plus | 0.7748 +- 0.0304 | 0.6333 +- 0.0401 | 0.7652 +- 0.0329 | 0.7811 +- 0.0208 | 0.7693 +- 0.0444 | 0.8250 +- 0.0255 | 0.7148 +- 0.0471 | 0.8695 +- 0.0227 | 0.0000 +- 0.0000 |
| az_no_fuzzy | 0.7519 +- 0.0295 | 0.6033 +- 0.0374 | 0.7366 +- 0.0314 | 0.7663 +- 0.0210 | 0.7382 +- 0.0372 | 0.8355 +- 0.0101 | 0.6597 +- 0.0455 | 0.8533 +- 0.0198 | 2.1185 +- 0.0119 |
| az_no_aniso | 0.7511 +- 0.0288 | 0.6022 +- 0.0366 | 0.7353 +- 0.0384 | 0.7660 +- 0.0223 | 0.7368 +- 0.0346 | 0.8344 +- 0.0202 | 0.6581 +- 0.0496 | 0.8526 +- 0.0186 | 0.0000 +- 0.0000 |
| az_thesis | 0.7005 +- 0.0398 | 0.5405 +- 0.0461 | 0.6678 +- 0.0317 | 0.7260 +- 0.0444 | 0.6769 +- 0.0356 | 0.8114 +- 0.0382 | 0.5675 +- 0.0277 | 0.8204 +- 0.0212 | 9.3260 +- 0.1992 |

Вывод по DRIVE: текущий полный `az_thesis` не надо подавать как улучшение. Сильнейший результат здесь дает `unet_plus_plus`. При этом диагностически видно, что full AZ формирует очень большую анизотропию (`anisotropy gap` около `9.33`), но эта жесткая геометрия ухудшает overlap и skeleton recall. Для статьи это аргумент не в пользу "мы лучше", а в пользу необходимости калибровки силы анизотропии.

### 26.2 CHASE_DB1

| Variant | Dice | IoU | clDice | Precision | Recall | Skel Prec | Skel Rec | Bal Acc | Aniso gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.6907 +- 0.0422 | 0.5291 +- 0.0484 | 0.7156 +- 0.0152 | 0.6277 +- 0.0755 | 0.7761 +- 0.0232 | 0.7610 +- 0.0537 | 0.6783 +- 0.0225 | 0.8647 +- 0.0100 | 0.0000 +- 0.0000 |
| attention_unet | 0.7148 +- 0.0185 | 0.5566 +- 0.0226 | 0.7271 +- 0.0218 | 0.6622 +- 0.0244 | 0.7777 +- 0.0262 | 0.7768 +- 0.0247 | 0.6837 +- 0.0237 | 0.8695 +- 0.0127 | 0.0000 +- 0.0000 |
| unet_plus_plus | 0.7224 +- 0.0071 | 0.5655 +- 0.0087 | 0.7266 +- 0.0030 | 0.6893 +- 0.0217 | 0.7600 +- 0.0183 | 0.8040 +- 0.0105 | 0.6630 +- 0.0082 | 0.8634 +- 0.0074 | 0.0000 +- 0.0000 |
| az_no_fuzzy | 0.6975 +- 0.0051 | 0.5355 +- 0.0061 | 0.7103 +- 0.0063 | 0.6437 +- 0.0315 | 0.7650 +- 0.0380 | 0.7587 +- 0.0327 | 0.6705 +- 0.0354 | 0.8617 +- 0.0152 | 2.1270 +- 0.0076 |
| az_no_aniso | 0.6951 +- 0.0218 | 0.5331 +- 0.0258 | 0.6972 +- 0.0311 | 0.6613 +- 0.0055 | 0.7336 +- 0.0439 | 0.7878 +- 0.0045 | 0.6276 +- 0.0544 | 0.8486 +- 0.0212 | 0.0000 +- 0.0000 |
| az_thesis | 0.6864 +- 0.0161 | 0.5227 +- 0.0186 | 0.6819 +- 0.0106 | 0.6299 +- 0.0504 | 0.7609 +- 0.0381 | 0.7260 +- 0.0679 | 0.6511 +- 0.0468 | 0.8582 +- 0.0137 | 1.3284 +- 0.0117 |

Вывод по CHASE_DB1: Attention U-Net и U-Net++ сильнее текущего полного AZ по Dice/IoU. `az_no_fuzzy` немного выше vanilla baseline по Dice (`+0.0068`), но full `az_thesis` ниже baseline. Это говорит, что текущая комбинация fuzzy + anisotropy в полной модели переограничивает локальную агрегацию; сама геометрическая часть полезна только при мягком режиме.

### 26.3 HRF_SegPlus

| Variant | Dice | IoU | clDice | Precision | Recall | Skel Prec | Skel Rec | Bal Acc | Aniso gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.6551 +- 0.0184 | 0.4874 +- 0.0206 | 0.5546 +- 0.0118 | 0.6300 +- 0.0405 | 0.6842 +- 0.0064 | 0.6714 +- 0.0247 | 0.4727 +- 0.0108 | 0.7712 +- 0.0091 | 0.0000 +- 0.0000 |
| attention_unet | 0.6679 +- 0.0177 | 0.5016 +- 0.0199 | 0.5550 +- 0.0129 | 0.6611 +- 0.0330 | 0.6757 +- 0.0089 | 0.7029 +- 0.0300 | 0.4592 +- 0.0148 | 0.7771 +- 0.0098 | 0.0000 +- 0.0000 |
| unet_plus_plus | 0.6723 +- 0.0156 | 0.5066 +- 0.0175 | 0.5478 +- 0.0141 | 0.6723 +- 0.0164 | 0.6724 +- 0.0162 | 0.6984 +- 0.0472 | 0.4517 +- 0.0114 | 0.7791 +- 0.0105 | 0.0000 +- 0.0000 |
| az_no_fuzzy | 0.6578 +- 0.0162 | 0.4903 +- 0.0179 | 0.5366 +- 0.0213 | 0.6549 +- 0.0473 | 0.6639 +- 0.0168 | 0.7018 +- 0.0196 | 0.4357 +- 0.0305 | 0.7697 +- 0.0065 | 2.1940 +- 0.0035 |
| az_no_aniso | 0.6628 +- 0.0170 | 0.4959 +- 0.0191 | 0.5468 +- 0.0304 | 0.6605 +- 0.0499 | 0.6704 +- 0.0371 | 0.7305 +- 0.0251 | 0.4386 +- 0.0386 | 0.7735 +- 0.0097 | 0.0000 +- 0.0000 |
| az_thesis | 0.6510 +- 0.0066 | 0.4826 +- 0.0072 | 0.5345 +- 0.0087 | 0.6293 +- 0.0202 | 0.6751 +- 0.0117 | 0.6808 +- 0.0166 | 0.4405 +- 0.0143 | 0.7679 +- 0.0028 | 1.2951 +- 0.0329 |

Вывод по HRF_SegPlus: текущий full `az_thesis` ниже baseline по Dice (`-0.0041`) и clDice (`-0.0201`). На HRF лучше всего работает U-Net++ по Dice/IoU, а `az_no_aniso` дает небольшой прирост над baseline по Dice (`+0.0077`) и skeleton precision, но снижает skeleton recall. Это означает, что fuzzy/local filtering может повышать избирательность, но текущая постановка не сохраняет слабые продолжения сосудов достаточно хорошо.

### 26.4 Что честно писать в 3 статье после reviewer-pack

Не писать:

- "the proposed full AZ model outperforms U-Net, Attention U-Net and U-Net++";
- "anisotropic fuzzy convolution improves Dice on all medical datasets";
- "the current full configuration is final".

Писать:

- "The proposed layer is evaluated as a geometric local aggregation mechanism, and the reviewer-oriented experiment shows both its potential and its current limitations."
- "Strong U-Net-family baselines remain competitive and often outperform the current full AZ configuration."
- "The ablation indicates that the effect of geometry is sensitive to calibration: removing fuzzy weighting or disabling anisotropy may be better than the current full coupling on some datasets."
- "A large anisotropy gap does not automatically imply better segmentation. On DRIVE it correlates with lower skeleton recall, which suggests over-selective propagation along local modes."
- "Future work must include topology-aware loss calibration, learnable/regularized anisotropy strength, and a sweep over the number of local modes."

### 26.5 Как закрывать замечания рецензента

| Замечание | Что теперь есть | Честный статус |
|---|---|---|
| Слабый baseline | Добавлены `attention_unet` и `unet_plus_plus` | Закрыто частично; CS-Net/IterNet/transformer еще нет |
| Недостаточно medical datasets | DRIVE, CHASE_DB1, HRF_SegPlus | Закрыто для conference-level revision |
| Нет ablation | `az_no_fuzzy`, `az_no_aniso`, `az_thesis` | Закрыто |
| Нет multi-seed | seeds `41,42,43`, mean/std | Закрыто |
| Нет Precision/Recall/clDice | Таблицы выше включают Precision, Recall, clDice, skeleton precision/recall | Закрыто |
| Падение clDice | Теперь явно видно и объясняется как over-suppression weak vessel continuations | Закрыто объяснением, но не исправлено качественно |
| Почему число режимов | Есть launcher `run_article3_regime_count_sweep.ps1` | Протокол есть, но sweep еще надо запускать |
| Reproducibility | Код, configs и launchers в public repo | Закрыто |

### 26.6 Главная новая мысль для статьи

После этих результатов третью статью лучше строить не как "мы победили U-Net", а как честную инженерно-научную работу:

**"Анизотропная нечеткая локальная свертка как диагностируемый геометрический модуль: когда направленная локальная агрегация помогает, когда переограничивает сосудистую структуру, и какие метрики показывают этот режим."**

Так статья отличается от первых двух:

- статья 1: математические свойства оператора;
- статья 2: алгоритм сегментации и базовый эксперимент;
- статья 3: воспроизводимая медицинская валидация, сильные baseline, ablation, диагностика anisotropy gap / clDice / skeleton recall и честный анализ ограничений.

---

## 27) Source-of-truth для исправления замечаний рецензента

Этот раздел нужен как рабочая выжимка: что именно менять в статье и что писать в ответе рецензенту. Ниже только факты, которые уже подтверждены кодом и reviewer-pack от `2026-04-30`.

### 27.1 Главная корректировка позиции статьи

Старый рискованный claim:

> The proposed AZ method improves segmentation quality over U-Net.

Так больше писать нельзя, потому что полный reviewer-pack показывает: `az_thesis` не превосходит сильные U-Net-family baselines и не превосходит vanilla baseline на всех медицинских датасетах.

Новый корректный claim:

> This work evaluates anisotropic fuzzy local convolution as an interpretable geometric local aggregation module for thin-structure segmentation. The experiments show that the current full AZ configuration is not universally superior to strong U-Net-family baselines, but the proposed diagnostics reveal when anisotropic aggregation becomes over-selective and how this affects overlap and skeleton-based metrics.

Короткий русский смысл:

> Мы не заявляем "лучше всех". Мы показываем воспроизводимый анализ геометрического модуля: где он помогает, где переограничивает тонкие структуры, и какими метриками это видно.

### 27.2 Что вставить вместо старой основной таблицы

Для медицинской статьи основная таблица должна быть не про Roads_HF и не про SpaceNet3 как главный результат, а про медицинские датасеты:

- DRIVE;
- CHASE_DB1;
- HRF_SegPlus.

Минимальная таблица для paper body:

| Dataset | Variant | Dice | IoU | clDice | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| DRIVE | baseline | 0.7646 +- 0.0286 | 0.6198 +- 0.0370 | 0.7499 +- 0.0281 | 0.7753 +- 0.0322 | 0.7543 +- 0.0252 |
| DRIVE | Attention U-Net | 0.7668 +- 0.0239 | 0.6224 +- 0.0311 | 0.7568 +- 0.0235 | 0.7716 +- 0.0113 | 0.7625 +- 0.0363 |
| DRIVE | U-Net++ | **0.7748 +- 0.0304** | **0.6333 +- 0.0401** | **0.7652 +- 0.0329** | **0.7811 +- 0.0208** | **0.7693 +- 0.0444** |
| DRIVE | AZ full | 0.7005 +- 0.0398 | 0.5405 +- 0.0461 | 0.6678 +- 0.0317 | 0.7260 +- 0.0444 | 0.6769 +- 0.0356 |
| CHASE_DB1 | baseline | 0.6907 +- 0.0422 | 0.5291 +- 0.0484 | 0.7156 +- 0.0152 | 0.6277 +- 0.0755 | 0.7761 +- 0.0232 |
| CHASE_DB1 | Attention U-Net | 0.7148 +- 0.0185 | 0.5566 +- 0.0226 | **0.7271 +- 0.0218** | 0.6622 +- 0.0244 | **0.7777 +- 0.0262** |
| CHASE_DB1 | U-Net++ | **0.7224 +- 0.0071** | **0.5655 +- 0.0087** | 0.7266 +- 0.0030 | **0.6893 +- 0.0217** | 0.7600 +- 0.0183 |
| CHASE_DB1 | AZ full | 0.6864 +- 0.0161 | 0.5227 +- 0.0186 | 0.6819 +- 0.0106 | 0.6299 +- 0.0504 | 0.7609 +- 0.0381 |
| HRF_SegPlus | baseline | 0.6551 +- 0.0184 | 0.4874 +- 0.0206 | 0.5546 +- 0.0118 | 0.6300 +- 0.0405 | **0.6842 +- 0.0064** |
| HRF_SegPlus | Attention U-Net | 0.6679 +- 0.0177 | 0.5016 +- 0.0199 | **0.5550 +- 0.0129** | 0.6611 +- 0.0330 | 0.6757 +- 0.0089 |
| HRF_SegPlus | U-Net++ | **0.6723 +- 0.0156** | **0.5066 +- 0.0175** | 0.5478 +- 0.0141 | **0.6723 +- 0.0164** | 0.6724 +- 0.0162 |
| HRF_SegPlus | AZ full | 0.6510 +- 0.0066 | 0.4826 +- 0.0072 | 0.5345 +- 0.0087 | 0.6293 +- 0.0202 | 0.6751 +- 0.0117 |

Подпись к таблице:

`Table X. Multi-seed medical segmentation results (mean +- std over seeds 41/42/43). The current full AZ configuration is included as an interpretable geometric aggregation module rather than claimed as a state-of-the-art model.`

### 27.3 Таблица ablation для закрытия вопроса "что дает fuzzy и anisotropy"

Эту таблицу лучше вынести в paper body или appendix:

| Dataset | Variant | Dice | clDice | Skeleton Recall | Anisotropy gap | Interpretation |
|---|---|---:|---:|---:|---:|---|
| DRIVE | baseline | 0.7646 | 0.7499 | 0.6850 | 0.0000 | standard U-Net reference |
| DRIVE | AZ no fuzzy | 0.7519 | 0.7366 | 0.6597 | 2.1185 | geometry active, fuzzy off |
| DRIVE | AZ no aniso | 0.7511 | 0.7353 | 0.6581 | 0.0000 | fuzzy/local block without anisotropy |
| DRIVE | AZ full | 0.7005 | 0.6678 | 0.5675 | 9.3260 | over-selective anisotropy |
| CHASE_DB1 | baseline | 0.6907 | 0.7156 | 0.6783 | 0.0000 | standard U-Net reference |
| CHASE_DB1 | AZ no fuzzy | 0.6975 | 0.7103 | 0.6705 | 2.1270 | mild Dice gain over baseline |
| CHASE_DB1 | AZ no aniso | 0.6951 | 0.6972 | 0.6276 | 0.0000 | fuzzy-only/local effect |
| CHASE_DB1 | AZ full | 0.6864 | 0.6819 | 0.6511 | 1.3284 | full coupling below baseline |
| HRF_SegPlus | baseline | 0.6551 | 0.5546 | 0.4727 | 0.0000 | standard U-Net reference |
| HRF_SegPlus | AZ no fuzzy | 0.6578 | 0.5366 | 0.4357 | 2.1940 | small Dice gain, lower topology |
| HRF_SegPlus | AZ no aniso | 0.6628 | 0.5468 | 0.4386 | 0.0000 | best AZ ablation by Dice |
| HRF_SegPlus | AZ full | 0.6510 | 0.5345 | 0.4405 | 1.2951 | full coupling below baseline |

Главный вывод из ablation:

> The ablation does not show a simple monotonic benefit from adding both fuzzy weighting and anisotropy. Instead, it shows that the geometric module is sensitive to calibration. High anisotropy strength may suppress weak continuations and reduce skeleton recall.

### 27.4 Как переписать Results

Готовый английский блок:

`Table X reports multi-seed results on DRIVE, CHASE_DB1 and HRF_SegPlus. The strongest overlap scores are obtained by U-Net++ on all three datasets, while Attention U-Net also improves over the vanilla U-Net baseline. The current full AZ configuration does not outperform these stronger baselines and in several cases remains below vanilla U-Net. This result is important because it indicates that anisotropic fuzzy aggregation should not be presented as an accuracy-only replacement for established encoder-decoder architectures. Instead, it should be interpreted as a geometric local module whose behavior depends on calibration.`

Продолжение:

`The ablation study shows that disabling either fuzzy weighting or anisotropy can be preferable to the current full coupling on some datasets. On DRIVE, the full AZ model reaches a high anisotropy gap but suffers a substantial decrease in Dice and skeleton recall. On HRF_SegPlus, the no-anisotropy variant slightly improves Dice over vanilla U-Net, whereas the full AZ variant decreases both Dice and clDice. These results suggest over-selective propagation along local modes and motivate topology-aware calibration of anisotropy strength.`

### 27.5 Как переписать Discussion

Готовый английский блок:

`The main lesson from the reviewer-oriented experiment is that geometry-aware aggregation is not automatically beneficial when inserted into a segmentation network. The anisotropy gap provides a useful diagnostic: when it becomes too large, the model may become overly selective and suppress faint continuations of thin structures. This is consistent with the observed decrease in skeleton recall and clDice for the full AZ configuration. Therefore, future versions should regularize anisotropy strength, include topology-preserving losses, and tune the number of local regimes.`

Еще один блок:

`The negative and mixed results are valuable because they identify the operating regime of the proposed mechanism. The current full AZ layer is best viewed as an interpretable geometric component, not as a finished state-of-the-art architecture. Its diagnostic maps make it possible to connect local errors with internal orientation and anisotropy states, which is difficult to obtain from standard U-Net baselines.`

### 27.6 Как переписать Limitations

Готовый английский блок:

`This study has several limitations. First, although Attention U-Net and U-Net++ were added as stronger baselines, we have not yet included nnU-Net, transformer-based segmentation models, or vessel-specific architectures such as CS-Net or IterNet. Second, the current AZ configuration does not consistently improve overlap metrics, and its full fuzzy-anisotropic coupling may over-suppress weak vessel continuations. Third, the sensitivity to the number of local regimes has not yet been fully evaluated, although the repository includes a launcher for an R sweep. Finally, the current visualization estimates orientation axes modulo pi rather than signed flow direction.`

### 27.7 Что написать в ответ рецензенту

**Пункт 1. Слабый baseline.**

`Исправлено частично. Мы добавили сравнение с Attention U-Net и U-Net++ в дополнение к vanilla U-Net. Результаты приведены в новой таблице multi-seed эксперимента на DRIVE, CHASE_DB1 и HRF_SegPlus. Специализированные сосудистые методы CS-Net/IterNet и transformer/nnU-Net пока вынесены в ограничения и future work.`

**Пункт 2. Недостаточно медицинских датасетов.**

`Исправлено. Экспериментальная часть расширена до трех медицинских наборов: DRIVE, CHASE_DB1 и HRF_SegPlus. Для каждого набора использованы одинаковые варианты моделей, seeds 41/42/43 и единая политика выбора порога.`

**Пункт 3. Нет ablation study.**

`Исправлено. Добавлены варианты az_no_fuzzy, az_no_aniso и az_thesis. Это позволяет отдельно оценить вклад fuzzy weighting, анизотропии и их полной комбинации. Результаты показывают, что текущая полная комбинация не всегда оптимальна и требует калибровки.`

**Пункт 4. Падение clDice.**

`Исправлено объяснением и дополнительными метриками. Мы явно показываем clDice и skeleton recall. Падение clDice трактуется как признак over-suppression слабых продолжений сосудов при слишком селективной анизотропной агрегации. В Discussion добавлено, что это требует topology-aware loss, например clDice/skeleton loss.`

**Пункт 5. Масштабируемость и число режимов.**

`Частично исправлено. В репозиторий добавлен launcher scripts/run_article3_regime_count_sweep.ps1 для эксперимента R=1,2,4,8,16. Полный sweep пока не включен в основные результаты и указан как обязательное направление дальнейшей работы.`

**Пункт 6. Статистическая значимость.**

`Исправлено на уровне conference revision. Добавлен multi-seed протокол seeds 41/42/43 и mean +- std для всех основных метрик. Более строгие confidence intervals и per-image boxplots оставлены для расширенной версии.`

**Пункт 7. Ссылка на Аносова.**

`Исправлено концептуально. В финальной версии рекомендуется убрать акцент на Аносова и позиционировать направленную часть через steerable filters / anisotropic diffusion.`

**Пункт 8. Reproducibility.**

`Исправлено. Реализация, конфиги, launchers и скрипты визуализации доступны в публичном репозитории: https://github.com/fims9000/anza_lira.`

**Пункт 9. Roads_HF бесполезен как main metric.**

`Исправлено. Roads_HF не следует включать в основную медицинскую таблицу. Его можно оставить только как качественный/санити пример или убрать из статьи полностью, если статья позиционируется как медицинская.`

**Пункт 10. Precision/Recall не раскрыты.**

`Исправлено. Новые таблицы включают Dice, IoU, clDice, Precision, Recall, skeleton precision, skeleton recall и balanced accuracy.`

### 27.8 Что срочно поменять в `stdh2026_paper_draft_en.md`

1. Убрать старую таблицу, где HRF `AZ-Thesis` выглядит как улучшение Dice `0.6822`.
2. Заменить main quantitative table на таблицу из `27.2`.
3. Убрать формулировку `AZ improves HRF`; вместо этого написать:
   `On HRF_SegPlus, U-Net++ gives the best Dice/IoU, while the current full AZ configuration remains below the vanilla baseline.`
4. Переписать conclusion:
   - не "AZ is a practical mechanism when calibration is applied";
   - а "AZ is an interpretable geometric module whose current configuration reveals both potential and limitations".
5. Добавить subsection `Ablation and Diagnostic Metrics`.
6. Добавить subsection `Reviewer-Oriented Limitations`.
7. В References/Introduction не продавливать Аносова; лучше Freeman & Adelson / Perona-Malik / U-Net / Attention U-Net / U-Net++.

### 27.9 Минимальная новая структура статьи

1. Introduction
   - thin vascular structures;
   - why local geometry matters;
   - why this paper is not only about Dice, but about diagnostic geometry.
2. Method
   - AZ local aggregation;
   - orientation axis, not signed flow;
   - anisotropy gap and fuzzy memberships.
3. Experimental Setup
   - DRIVE, CHASE_DB1, HRF_SegPlus;
   - variants: U-Net, Attention U-Net, U-Net++, AZ ablations;
   - seeds, epochs, optimizer/loss/threshold.
4. Results
   - main table from 27.2;
   - ablation table from 27.3.
5. Discussion
   - strong baselines win;
   - current full AZ over-selects;
   - anisotropy gap explains failures;
   - why visualization is still useful.
6. Limitations and Future Work
   - no nnU-Net/transformer/CS-Net yet;
   - R sweep not completed;
   - need topology-aware loss;
   - axis not signed flow.
7. Conclusion
   - honest contribution: interpretable geometric module + reproducible diagnosis, not SOTA claim.
