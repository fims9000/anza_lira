# Карта архитектуры CT28 Graph-LIRA

Статус: development train/validation. Held-out test не открывался, path stage
не реализован. Результаты и состав pack приведены в `README.md`.

## Pipeline

```text
frozen PAIR/JUNCTION sources
  → controlled CT28 scenes
  → PAIR/JUNCTION candidates
  → geometry и CCTA features
  → patient-level OOF local scores
  → HGB relation head: NONE / PAIR / JUNCTION / BOTH
  → relation confidence tau = 0.85
  → endpoint-disjoint structural selection
  → perturbation consistency ≥ 0.60
  → repair или abstain
```

Построение сосудистого пути начинается после принятого repair и не входит в
текущий milestone.

## Local evidence

Для PAIR и JUNCTION независимо обучаются geometry, CT и geometry+CT binary
local models:

| Candidate | Geometry | CT | Geometry+CT |
|---|---:|---:|---:|
| PAIR | 7 | 38 | 45 |
| JUNCTION | 9 | 152 | 161 |

Основной local model:
`MedianImputer → StandardScaler → LogisticRegression`. Train scores получаются
через пять patient-level OOF folds; validation scores — final models,
обученными только на train.

## Relation decision

Historical representation одного score stream содержит 42 scene-признака:

- по 18 признаков распределения PAIR и JUNCTION scores: `top-1…top-7`, три
  top-score gap, mean, standard deviation, median, 75-й и 90-й percentiles,
  minimum, candidate count и доля top score в сумме;
- шесть cross-stream признаков.

Используются три независимые HGB relation heads:

| Head | Input |
|---|---:|
| geometry-only | 42 признака из `geometry_score` |
| geometry+CT early fusion | 42 признака из `geometry_plus_ct_score` |
| CT-conditioned | 42 geometry + 42 CT = 84 |

Каждая head выдаёт вероятности `NONE / PAIR / JUNCTION / BOTH`. Если
максимальная вероятность ниже `tau=0.85`, решение заменяется на `NONE`.

## Structural selection

- `PAIR` выбирает лучший PAIR candidate.
- `JUNCTION` выбирает лучший JUNCTION candidate.
- `BOTH` выбирает совместимую PAIR/JUNCTION-комбинацию с максимальным общим
  score.
- `NONE` не принимает связь.

PAIR и JUNCTION в `BOTH` не могут использовать общий endpoint. При конфликте
независимых максимумов перебираются endpoint-disjoint комбинации. Это
подтверждённое global compatibility rule; дополнительные canonical graph
constraints не восстановлены.

Решение повторяется 15 раз с position noise `0.8 мм` и tangent noise `20°`.
При exact-identity consistency ниже `0.60` система abstains.

## Provenance

| Компонент | Статус |
|---|---|
| Классы `NONE/PAIR/JUNCTION/BOTH` | Восстановлены точно. |
| Семейства `pair_only/junction_only/mixed/none_*` | Документированы. |
| Historical scene assembly | `NOT RECOVERED`; заменён `ct28_scene_protocol_v1`. |
| Candidate envelope | Документирован: PAIR `≤12 мм`, JUNCTION `≤16 мм`; anatomical labels не используются для inference. |
| Historical local weights | `NOT RECOVERED`; модели обучены заново на CT28 train. |
| 42 relation features | Восстановлены точно. |
| Historical relation weights/folds | `NOT RECOVERED`; heads обучены заново на CT28 train. |
| `tau=0.85` | Восстановлен точно. |
| Endpoint-disjoint compatibility | Восстановлена точно. |
| Historical consistency formula | `NOT RECOVERED`; используется `ct28_perturbation_consistency_v1` с gate `0.60`. |
| Path construction | `NOT RECOVERED`; исключён из результата. |

Reference sources для аудита:

```bash
bash scripts/research/ccta_graph_lira_safe_repair/restore_historical_graph_lira_sources.sh
```

Они разворачиваются во временную папку и не являются runtime dependency CT28.

