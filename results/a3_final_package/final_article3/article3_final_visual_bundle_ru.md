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

