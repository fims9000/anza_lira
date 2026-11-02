# CT28: независимая проверка вклада ранее извлечённых CCTA-признаков

Дата вычисления: **2026-11-02**. Доказательство: [GitHub Actions, успешный реальный CPU-прогон #41505361958](https://github.com/fims9000/anza_lira/actions/runs/41505361958). Машиночитаемые `ct_signal_benchmark.json` и `ct_signal_candidate_val_predictions.csv`: [artifact #13171080326](https://github.com/fims9000/anza_lira/actions/runs/41505361958/artifacts/13171080326), GitHub retention до 2026-12-02.

**Исходные данные:** `artifacts/varvara/ct28_graph_lira/scene_candidate_features_train_val.csv`, SHA-256 `dc406fdac832cb55c7e59417102cb1adf8bcef3b6afa25c98c3ff6958a6cedee`, 15 092 кандидата. Признаки в этом файле ранее вычислены из настоящих ImageCAS CCTA-объёмов; **сам архив изображений при данном эксперименте не открывался**.

**Протокол:** 17 train-пациентов, 5 validation-пациентов; шесть held-out test не открыты. По одному `HistGradientBoostingClassifier` для PAIR и JUNCTION, фиксированные hyperparameters: `max_iter=100, max_leaf_nodes=15, learning_rate=0.06, l2_regularization=2, class_weight='balanced', early_stopping=False, random_state=0`; sklearn `1.5.2`. Метрики только по validation. Четыре заранее выбранные альтернативы: geometry-only; CT-only; geometry+CT; geometry+patientwise joint-shuffled CT (порядок геометрических строк и labels не меняется, все CT-колонки переставляются **совместно** внутри каждого пациента). Из числовых признаков исключены `scene_id`, `candidate_id`, `endpoint_ids`, `split`, `scan_id`, label `y`, `ct_sampling_valid`, `ct_sampling_error`, candidate counts.

## Основные фактические результаты

| Эмпирическое качество | PAIR (2298 val; 170 positives) | JUNCTION (1773 val; 72 positives) |
|---|---:|---:|
| Только geometry, AP | 0.736343 | 0.750113 |
| Только CT, AP | 0.453075 | 0.342654 |
| Geometry + CT, AP | **0.881366** | **0.849010** |
| Geometry + shuffled CT, AP | 0.713988 | 0.739318 |
| Только geometry, ROC-AUC | 0.976783 | 0.991459 |
| Только CT, ROC-AUC | 0.912320 | 0.898532 |
| Geometry + CT, ROC-AUC | **0.990168** | **0.993599** |
| Geometry + shuffled CT, ROC-AUC | 0.976451 | 0.989834 |

Разница AP по всему candidate-level validation-пулу: `PAIR +0.145023` (real CT vs geometry), `JUNCTION +0.098897`. Это **не** то же самое, что среднее различие AP по пациентам: размеры/превалентность различаются и AP нелинейна по числу наблюдений.

## Парные по пациентам результаты — описательная неопределённость (5 пациентов)

При парном сравнении пациент-специфичных AP/ROC и bootstrap, ресемплирующем **пациентов**, а не сцены, 2 000 повторений:

| Контраст | PAIR: mean ΔAP, 95% descript. range | JUNCTION: mean ΔAP, 95% descript. range |
|---|---|---|
| Real CT + geometry минус geometry | +0.16652 [0.09007, 0.23322] | +0.09089 [−0.00634, 0.19709] |
| Real CT + geometry минус shuffled CT + geometry | +0.18030 [0.09493, 0.24121] | +0.12478 [0.04662, 0.17591] |

Из пяти пациентов преимущество PAIR real CT против geometry по AP есть у **5/5**; для JUNCTION — **4/5**. Для JUNCTION интервал contrast vs geometry пересекает 0, и поэтому не следует утверждать, что добавление CT неизменно улучшает выбор JUNCTION у каждого пациента.

## Вывод и строгое ограничение

**Подтверждено:** в сохранённой CT28 candidate-level постановке истинные CCTA-derived числовые признаки предоставляют полезный сигнал сверх геометрии. Совместное перемешивание CT-признаков внутри пациентов ухудшает результаты при той же размерности входа HGB. Это исключает объяснение результатов **только** большей размерностью CT-входа, хотя не исключает все возможные shortcuts/каузальные конфаундеры.

**Не подтверждено этим экспериментом:**
- преимущество новой свёртки `AZConv3d` над 3D CNN (обе нуждаются в настоящих CT voxel arrays для обучения);
- итоговая точность/безопасность восстановленных `.nii.gz` сосудистых масок;
- выигрыш на естественных ошибках исходной модели сегментации;
- независимая обобщающая способность, поскольку 5 validation пациентов уже использовались для выбора исторических K10/15 variants;
- отсутствие ложных связей внутри одной GT-коронарной компоненты.

Источник метрик подписан checksum: `ct_signal_benchmark.json` SHA-256 `e30d30e35db54985414d755ea4722d7d9eabb8f514e50fccf2fde11c2d86cb38`; `ct_signal_candidate_val_predictions.csv` SHA-256 `fe957a791a8a8d99d0e3745a4ff441c0084a3d14b4b1310aae5c1dd93fe12e95`. Версия кода успешного расчёта: `a02a82d5e8d9f092b98608b551d89093931fd9b0`.

**Воспроизведение без медицинского архива:**

```bash
python scripts/research/ccta_graph_lira_safe_repair/ct28_ct_signal_benchmark.py \
  --output-dir varvara/anza_lira_local_results/ct28_real_CT_feature_signal \
  --max-iter 100 --bootstrap 2000 --seed 20261102
```

Пакеты успешного GitHub Actions прогона: Python 3.11, numpy 1.26.4, pandas 2.2.3, scipy 1.14.1, scikit-learn 1.5.2.

## Дополнительная проверка: 25 независимых перестановок CT-признаков

После основного эксперимента была отдельно проверена устойчивость результата к выбору *конкретной случайной перестановки*. Для каждого типа кандидата на тех же 17 train и 5 development-validation пациентах обучены 25 HGB моделей с разными заранее фиксированными seed. Внутри каждого пациента **совместно переставлялись целые CT-feature vectors**, сохраняя их распределение и набор кандидатов, но разрывая соответствие с геометрией и правильной меткой.

| Candidate | Реальный CT+геометрия AP | Mean sham AP (25) | Sham min–max AP | Sham AP >= real AP |
|---|---:|---:|---:|---:|
| PAIR | 0.881366 | 0.720090 | 0.695193–0.740626 | **0 из 25** |
| JUNCTION | 0.849010 | 0.729192 | 0.671627–0.795217 | **0 из 25** |

У пациента 966 JUNCTION-эффект в сравнении с перемешанными CT-признаками остался негативным в среднем по 25 повторам (**−0.032055 AP**); это не следует скрывать агрегированной таблицей.

В контрольной программе **отсутствует самостоятельный тест на иной клинической выборке**. «0 из 25» здесь — наблюдаемое число shuffled-null превосходств на тех же пяти post-selected validation пациентах, **не p-value и не доказательство причинности**.

- [Успешный повторный расчёт #41506191078](https://github.com/fims9000/anza_lira/actions/runs/41506191078)
- [JSON + 50 строк результатов всех повторов](https://github.com/fims9000/anza_lira/actions/runs/41506191078/artifacts/13171620128)
- `sham_robustness.json` SHA-256: `88d082d0e785b9cc90b3a039428b68ab81aca718cbe4dd217c807bce9b07b72f`
- `sham_repetitions.csv` SHA-256: `d5418c0dd92fd3345555b9d9fba66170f2aa47915916ad54fb7049ba4446b90b`

Воспроизведение:

```bash
python scripts/research/ccta_graph_lira_safe_repair/ct28_ct_sham_robustness.py \
  --output-dir varvara/anza_lira_local_results/ct28_real_CT_25_shams \
  --repetitions 25 --max-iter 100 --seed 20261102
```
