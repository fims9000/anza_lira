# Ответы на вопросы по моделям, данным и Graph-LIRA

Актуализировано: 2026-09-27  
Ветка: `research/coronary-connectivity-repair`

Ниже ответы именно на заданные вопросы. Если старые заметки противоречат этому файлу или `FINAL_HANDOFF_2026-09-27.md`, актуальны эти два файла.

## 1. Где `radial_hu_summary_v1`?

Это локальная binary PAIR model на radial CCTA features.

Архитектура:

`StandardScaler + LogisticRegression(C=1, class_weight="balanced")`

Канонический способ воспроизведения:

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_features.py
python scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_from_features.py \
  artifacts/varvara/ct28_pair/expanded_relation_features.csv \
  --out-dir ct28_pair_models
```

Результат:

`ct28_pair_models/radial_hu_summary_v1.joblib`

Если нужен готовый regenerated snapshot без retrain, он хранится как:

`artifacts/varvara/ct28_pair/models_payload/radial_hu_summary_v1.joblib.gz.b64`

и восстанавливается:

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_models.py
```

## 2. Где `geometry_hgb`?

Да: это `HistGradientBoostingClassifier` над семью PAIR geometry features `g0..g6`.

Воспроизводимый текущий код:

```bash
python scripts/research/ccta_graph_lira_safe_repair/train_geometry_hgb_from_pair_plan.py
```

По умолчанию он читает:

`artifacts/varvara/ct28_pair/relation_pair_plan.csv`

и создаёт:

`ct28_geometry_hgb/geometry_hgb.joblib`

Отдельный pre-generated snapshot HGB в pack не нужен, потому что он быстро и детерминированно retrain-ится из frozen pair plan.

Важно: этот `geometry_hgb` — binary PAIR baseline. Это **не** scene-level HGB relation head старого Graph-LIRA.

## 3. Где `geometry_plus_radial_v1` и она ли дала CT28 результат?

Да.

Это локальная binary **PAIR** model:

`geometry features + radial CCTA features -> StandardScaler -> LogisticRegression`

Она воспроизводится тем же `train_ct28_pair_from_features.py`.

Готовый regenerated snapshot:

`artifacts/varvara/ct28_pair/models_payload/geometry_plus_radial_v1.joblib.gz.b64`

Held-out CT28:

- AUROC 0.9847;
- recall 82.63%;
- FPR 1.80%;
- precision 97.87%;
- TP/FP/FN/TN = 138/3/29/164.

Да, именно от этой **PAIR relation** постановки получены эти цифры.

Это **не** end-to-end Graph-LIRA result и **не** JUNCTION model.

## 4. Где `run_graph_lira_large_scale.py`?

Его canonical copy в Git **нет**.

Это реальный historical reproducibility gap, а не проблема вашего clone.

Сохранены:

- large-scale result tables;
- model/threshold metadata;
- protocol/checkpoint documents;
- related exploratory Graph-LIRA scripts.

Главные исторические материалы:

- `docs/research/ccta_graph_lira_safe_repair/LARGE_SCALE_800_CASE.md`;
- `docs/research/ccta_graph_lira_safe_repair/LARGE_SCALE_800_CASE_REPORT.md`;
- `results/ccta_graph_lira_safe_repair/2026-09-20/`.

Не надо искать этот runner по скрытым папкам.

## 5. Где `graph_lira_large_scale/scenes_full.pkl`, `graph_lira_selective/models.joblib`, `graph_lira_relation_type/relation_type_model.joblib`?

Их canonical copies тоже **не были сохранены**.

Старые локальные binary artifacts не считаются текущей точкой воспроизводимости.

Мы не восстанавливаем их из агрегированных результатов и не делаем вид, что это те же самые файлы.

## 6. А что было раньше relation head?

В geometry-only 800-case Graph-LIRA было три уровня:

```text
PAIR candidate geometry model
+
JUNCTION candidate geometry model
        ↓
out-of-fold score distributions
        ↓
scene-level HGB relation-type head
        ↓
NONE / PAIR / JUNCTION / BOTH
        ↓
global graph compatibility
        ↓
confidence + perturbation consistency
        ↓
repair / abstain
```

Scene-level relation head действительно был HGB с validation-selected confidence `tau=0.85`.

Это **другой HGB**, не `geometry_hgb` из CT28 binary PAIR baseline.

Exact old training runner relation head не сохранился; его behavior/results зафиксированы в large-scale checkpoints.

## 7. Правильно ли понимать `ct_scene_graph_lira_hybrid.py` как новую схему?

Не совсем.

Этот файл — **старый six-patient pilot**.

Там CT использовался как auxiliary presence / add-veto signal поверх уже существующей geometry relation logic.

Он не реализует новую чистую 28-patient CT-conditioned four-class head.

И его нельзя брать как финальную архитектуру: на held-out patients direct CT add-only/hysteresis увеличивал false structural repair.

То есть сейчас **не** делаем:

```text
old geometry head
+ geometry_plus_radial_v1
-> просто добавить CT-positive repair
```

## 8. Нужны ли `PAIR_IMG` и `JUNC_IMG`?

Старые weights — нет.

Это exploratory six-case image-presence models.

Идея двух image evidence streams остаётся правильной:

- PAIR_CT;
- JUNCTION_CT.

Но patient-general CT28 evidence сейчас нормально подтверждён только для PAIR.

Поэтому отсутствие нового JUNCTION+CT сигнала — не ошибка: это **текущий научный gap и текущая задача**.

## 9. Где CT volumes 28 пациентов?

Raw CCTA намеренно не хранится в Git.

Для уже frozen PAIR baseline он больше не нужен: full feature table committed losslessly.

Для новой JUNCTION+CT feature extraction raw CT **нужен**.

Точный cohort, Kaggle source, archive parts и alignment checks:

`docs/varvara/CT28_DATA_ACCESS.md`

Источник:

`https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas`

Используется frozen cohort 953–984 (28 пациентов, не весь диапазон подряд), split 17/5/6.

## 10. Где `expanded_relation_features.csv` и predictions?

Gap закрыт.

Row-level predictions:

`artifacts/varvara/ct28_pair/expanded_relation_predictions.csv`

Полный feature table хранится losslessly:

`artifacts/varvara/ct28_pair/payload/expanded_relation_features.csv.xz.b64`

Восстановить:

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_features.py
```

Frozen SHA256 проверяется скриптом.

## 11. Что теперь является её задачей?

Не повторять six-case hybrid.

Сначала:

```text
JUNCTION geometry
vs
JUNCTION CT
vs
JUNCTION geometry + CT
```

на тех же 28 пациентах и frozen split.

После этого:

```text
PAIR geometry + PAIR CT
+
JUNCTION geometry + JUNCTION CT
        ↓
new CT-conditioned NONE / PAIR / JUNCTION / BOTH head
        ↓
frozen Graph-LIRA
        ↓
tau = 0.85
consistency = 0.60
        ↓
repair / abstain
```

## 12. Быстрая проверка, что pack не сломан

Из корня репозитория:

```bash
python scripts/research/ccta_graph_lira_safe_repair/verify_varvara_collab_pack.py
```

Он проверяет наличие collaborator files, payload hashes, 1,360-row split/class counts и frozen headline metrics.
