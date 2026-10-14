# CT28 Graph-LIRA artifact pack

Статус: development train/validation. Held-out test не открывался, path stage
не выполнялся. Основной baseline использует local
`MedianImputer → StandardScaler → LogisticRegression`; local HGB сохранён как
абляция.

Архитектура и provenance: `ARCHITECTURE_MAP.md`. Условия абляций:
`ablations/README.md`.

## Состав pack

| Путь | Содержимое |
|---|---|
| `scene_plan_train_val.csv` | 1 404 сцены, split и relation ground truth. |
| `scene_endpoints_train_val.csv` | 7 020 endpoints. |
| `scene_candidates_train_val.csv` | 15 092 PAIR/JUNCTION candidates. |
| `scene_protocol.json` | Frozen scene rules, counts, recall и SHA256. |
| `scene_candidate_features_train_val.csv` | Geometry и CT candidate features. |
| `scene_ct_alignment_train_val.csv` | Alignment checks для 22 CT. |
| `scene_candidate_feature_protocol.json` | CT schema, missing policy и SHA256. |
| `relation_local_scores_train_val.csv` | Patient-level OOF local scores. |
| `relation_predictions_val.csv` | Validation probabilities четырёх relation classes. |
| `relation_metrics_val.json` | Relation-level validation metrics. |
| `relation_models.joblib` | Шесть final local models и три relation heads. |
| `relation_model_protocol.json` | Folds, feature lists, parameters и SHA256. |
| `graph_lira_predictions_val.csv` | Structural decision по каждой validation-сцене. |
| `graph_lira_structural_metrics_val.csv` | Главные structural metrics. |
| `graph_lira_per_patient_val.csv` | Per-patient metrics. |
| `graph_lira_patient_bootstrap_val.csv` | Patient-cluster bootstrap и CI. |
| `graph_lira_by_scene_type_val.csv` | Metrics по scene family. |
| `graph_lira_by_junction_group_val.csv` | Competitive/isolated metrics. |
| `perturbation_consistency_val.csv` | Exact-identity consistency. |
| `graph_lira_protocol.json` | Frozen structural protocol и SHA256. |
| `ablations/` | Clean HGB и single-stress runs с тем же output schema. |

В каждой готовой ablation run-папке лежат пять `relation_*` файлов и восемь
Graph-LIRA файлов. Одинаковые имена относятся к разным моделям или inputs.
Только stressed `scene_plan_train_val.csv` и
`scene_ct_alignment_train_val.csv` побайтово совпадают с clean pack; они
сохранены для самодостаточного frozen stressed input.

## Scene и CT layer

`ct28_scene_protocol_v1` содержит 1 036 train и 368 validation-сцен.
Candidate recall в repair-сценах равен 100%.

PAIR использует 38 CT-признаков radial corridor. JUNCTION использует tensor
`3 × 17 × 25`: 38 summaries на arm агрегируются через
`mean/min/max/std` в 152 признака. Все 22 CT прошли SHA256, shape, spacing и
affine checks.

У 17 train-candidates sampling grid выходит за CT: 2 PAIR и 15 JUNCTION.
Их CT features равны `NaN`; применяется train-fitted median imputation без
missing indicator. На validation таких случаев нет.

## Local evidence

Train scores получены через пять patient-level OOF folds. Validation scores
получены final models, обученными на 17 train patients.

| Candidate | Evidence | Features | Validation AUROC | Validation AUPRC |
|---|---|---:|---:|---:|
| PAIR | geometry | 7 | 0.6740 | 0.1088 |
| PAIR | CT | 38 | 0.8051 | 0.2051 |
| PAIR | geometry+CT | 45 | 0.8510 | 0.2404 |
| JUNCTION | geometry | 9 | 0.9492 | 0.2939 |
| JUNCTION | CT | 152 | 0.9069 | 0.3127 |
| JUNCTION | geometry+CT | 161 | 0.9536 | 0.4176 |

## Relation heads

| Head | Input | Raw accuracy | Balanced accuracy | Accuracy at tau=0.85 | Coverage |
|---|---:|---:|---:|---:|---:|
| geometry-only | 42 | 0.8886 | 0.8144 | 0.8451 | 0.8451 |
| geometry+CT early fusion | 42 | 0.8397 | 0.7529 | 0.8071 | 0.7690 |
| CT-conditioned | 84 | 0.8940 | 0.8340 | 0.8315 | 0.8342 |

Relation representations строятся из сохранённых local scores. При
необходимости их можно экспортировать через `build_ct28_relation_features.py`.

## Structural validation

Geometry-only и CT-conditioned используют одинаковые scenes, candidates,
constraints, `tau=0.85` и consistency gate `0.60`.

| Model | Repair-needed exact | False repair | Incomplete | Coverage | False among accepted | Exact among accepted |
|---|---:|---:|---:|---:|---:|---:|
| geometry-only | 0.0146 | 0.0054 | 0.5462 | 0.0136 | 0.4000 | 0.6000 |
| CT-conditioned | 0.0097 | 0.0000 | 0.5543 | 0.0054 | 0.0000 | 1.0000 |

До consistency repair-needed exact равен `0.2961` для geometry-only и
`0.2913` для CT-conditioned. Текущий bottleneck — exact candidate identity
stability.

