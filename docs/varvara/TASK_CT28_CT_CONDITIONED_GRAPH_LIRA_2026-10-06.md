# TASK — CT28 CT-conditioned Graph-LIRA

Date: 2026-10-06  
Assignee: Varvara Rodionova  
Canonical branch: `research/coronary-connectivity-repair`

## 0. Главный вопрос

Следующий этап должен ответить не на вопрос «можно ли ещё улучшить локальный JUNCTION classifier», а на вопрос:

> **Помогает ли CCTA evidence принимать более правильные структурные решения PAIR / JUNCTION / BOTH / NONE внутри Graph-LIRA при том же safety policy?**

Это главный следующий эксперимент. CNN / ANZA пока не начинаем.

Логика этапа:

```text
controlled candidates
        ->
local geometry evidence + local CCTA evidence
        ->
scene-level relation decision
NONE / PAIR / JUNCTION / BOTH
        ->
frozen Graph-LIRA compatibility
        ->
tau = 0.85
consistency = 0.60
        ->
repair / abstain
        ->
max-min path
```

---

## 1. Почему именно это делаем сейчас

### Что уже закрыто

#### PAIR

На frozen CT28 split уже есть воспроизводимый local PAIR experiment.

Best local PAIR evidence:

`geometry + radial 2.5-D CCTA`

Held-out result:

- AUROC 0.9847;
- recall 82.63%;
- FPR 1.80%;
- precision 97.87%.

Это сильный аргумент, что image context полезен локально, но это ещё не end-to-end Graph-LIRA.

#### JUNCTION train/validation

Controlled degree-3 JUNCTION benchmark теперь тоже достаточно зрелый для следующего этапа:

- 143 positives;
- 1,666 retained negatives;
- 17 train patients / 5 validation patients;
- candidate recall 100%;
- `segment_label` используется только как audit metadata;
- `competitive / isolated` разделены;
- held-out JUNCTION test пока не открыт.

Candidate CT representation:

- raw candidate tensor `3 x 17 x 25`;
- 38 CT summaries per arm;
- 152 symmetric CT features;
- 9 geometry features;
- 161 raw geometry+CT features;
- 22/22 train+val CT volumes прошли SHA / shape / spacing / affine checks.

Validation:

| model | candidate AUROC | top-1 all | top-1 competitive |
|---|---:|---:|---:|
| geometry | 0.9949 | 35/36 | 27/28 |
| CT | 0.9405 | 28/36 | 20/28 |
| raw geometry+CT | 0.9936 | 36/36 | 28/28 |

При validation-selected threshold:

- geometry: recall 0.9722, FPR 0.016, precision 0.814;
- raw geometry+CT: recall 0.9722, FPR 0.040, precision 0.636.

То есть **CT может улучшать ranking, но простая склейка всех 161 features ухудшает safe acceptance**.

Это и есть причина перейти от локального concat classifier к relation-level reasoning.

---

## 2. Рабочая гипотеза

Текущие результаты поддерживают следующую гипотезу, но пока её не доказывают:

> Geometry хорошо формирует и ранжирует большинство кандидатов. CCTA содержит дополнительный сигнал в части неоднозначных случаев. Этот сигнал полезнее использовать как отдельное evidence для решения о существовании/типе relation, чем как ещё 152 признака в одном локальном classifier.

Иными словами, сейчас проверяем:

```text
не "CT заменяет geometry"
а
"CT помогает relation decision понять,
когда геометрически правдоподобную связь можно принимать"
```

Особенно важен false structural repair: неправильная перемычка хуже, чем abstain.

---

## 3. Что НЕ является выводом текущего этапа

Пока нельзя утверждать:

- что JUNCTION+CT уже лучше geometry в целом;
- что ANZA нужен;
- что CNN нужен;
- что raw geometry+CT concat является JUNCTION-LIRA;
- что validation threshold metrics являются held-out result;
- что случай `966:left:58` сам по себе доказывает общий CT gain.

Случай `966:left:58` использовать как failure example: geometry ставит false candidate выше true, а CT даёт дополнительный сигнал. Но это n=1.

---

## 4. Сначала восстановить общую картину Graph-LIRA

Исторический exact `run_graph_lira_large_scale.py` и старые pickle/joblib не считаются canonical dependency и не должны восстанавливаться «по памяти».

В репозитории сохранены historical scripts/checkpoints. Использовать их как reference для:

- scene semantics;
- local-score -> relation feature construction;
- relation-type head;
- graph compatibility;
- perturbation consistency;
- structural metrics.

Для удобного чтения historical scripts есть:

`scripts/research/ccta_graph_lira_safe_repair/restore_historical_graph_lira_sources.sh`

Запуск:

```bash
bash scripts/research/ccta_graph_lira_safe_repair/restore_historical_graph_lira_sources.sh
```

Скрипт разворачивает reference sources во временную директорию и ничего не коммитит.

### Результат этого шага

До изменения кода создать короткий файл:

`artifacts/varvara/ct28_graph_lira/ARCHITECTURE_MAP.md`

В нём на 1–2 страницы зафиксировать:

1. где формируются PAIR/JUNCTION candidate scores;
2. как historical pipeline переходил от local scores к `NONE / PAIR / JUNCTION / BOTH`;
3. где находится global graph compatibility;
4. где применялись `tau` и perturbation consistency;
5. где начинался path stage;
6. какие части удалось восстановить точно, а какие придётся реализовать заново.

Если exact historical rule не найден — **не угадывать**. Пометить как `NOT RECOVERED` и описать новый explicit v1 rule отдельно.

---

## 5. Входы, которые уже есть

### PAIR

Canonical pack:

`artifacts/varvara/ct28_pair/`

Нужны прежде всего:

- `relation_pair_plan.csv`;
- `expanded_relation_features.csv`;
- `expanded_relation_predictions.csv`;
- `protocol.json`;
- alignment/provenance;
- frozen models / retraining scripts.

### JUNCTION

Canonical pack:

`artifacts/varvara/ct28_junction/`

В Git уже лежат:

- `baseline_metrics.json`;
- `junction_baseline_predictions.csv`;
- lossless split payload текущей train/val feature table;
- `restore_junction_features.py`.

Восстановление:

```bash
python artifacts/varvara/ct28_junction/restore_junction_features.py
```

Ожидаемый SHA256 feature table:

`c58f0aef645ea0cf4552b846cf4140c052908f6457efeb4bc6cdc3efa7532483`

Current code:

- `build_junction_plan.py`;
- `extract_junction_ct_features.py`;
- `train_junction_baselines.py`;
- `train_junction_lira_fusion.py`.

`train_junction_lira_fusion.py` пока считать **development diagnostic**, а не отдельным главным milestone.

---

## 6. Маленький freeze-check перед новой работой

Перед scene-level интеграцией добавить в `artifacts/varvara/ct28_junction/` из текущего локального run:

- exact `junction_relation_plan_train_val.csv`;
- exact `junction_candidate_endpoints_train_val.csv`;
- exact `junction_selection_audit_train_val.csv`;
- exact config, которым был выполнен frozen run;
- `junction_ct_alignment_train_val.csv`;
- SHA256 для этих файлов.

Это не отдельная исследовательская задача. Это просто фиксация exact inputs.

Важно: не перегенерировать их новым protocol. Сначала перенести именно те файлы, от которых получен текущий feature SHA и metrics.

---

## 7. Построить canonical CT28 scene layer — train/val сначала

Graph-LIRA работает не с одной строкой candidate classifier, а со структурной сценой.

Нужен новый explicit scene pack для CT28, который не зависит от потерянного `scenes_full.pkl`.

### Типы сцен

Сохранить historical semantics:

- `PAIR` — в сцене действительно есть pair repair;
- `JUNCTION` — есть junction repair;
- `BOTH` — присутствуют оба типа relation;
- `NONE` — ни одну структурную связь принимать нельзя.

Historical controlled scene families использовать как reference:

- `pair_only`;
- `junction_only`;
- `mixed`;
- `none_orphan_pair`;
- `none_incomplete_junction`.

### Критическое правило

Если exact старый способ сборки одной из scene families не восстанавливается из archived code/docs, не имитировать его молча.

Тогда:

1. описать новый deterministic `ct28_scene_protocol_v1`;
2. строить его только на train/val;
3. проверить class balance и candidate recall;
4. заморозить до test;
5. не называть его exact reproduction старого scene generator.

### Новый файл

Рекомендуемая точка входа:

`scripts/research/ccta_graph_lira_safe_repair/build_ct28_scene_plan.py`

Outputs:

`artifacts/varvara/ct28_graph_lira/scene_plan_train_val.csv`

и

`artifacts/varvara/ct28_graph_lira/scene_protocol.json`

---

## 8. Local evidence: geometry-only и CT-conditioned должны быть параллельными

Для каждой сцены relation head должен видеть evidence от PAIR и JUNCTION.

Минимальная логика:

```text
PAIR candidate geometry scores
PAIR candidate CT scores

JUNCTION candidate geometry scores
JUNCTION candidate CT scores
```

### Train leakage rule

Если local scores используются как входы следующего обучаемого relation head, train scores должны быть **patient-level OOF**.

Нельзя:

- обучить local model на всех train patients;
- получить её scores на тех же train rows;
- обучить сверху relation head и считать это честным train representation.

Для validation:

- base local models обучаются только на train;
- validation scores получаются frozen models;
- validation используется для model/threshold selection, но не для fit relation head.

### Что делать с raw concat

`geometry_plus_ct` оставить в таблицах как baseline.

Но основной CT-conditioned relation experiment должен позволять relation layer различать geometry evidence и image evidence. Не заставлять всю задачу сводиться к одному raw 161-feature JUNCTION score.

---

## 9. Relation-type head

Target:

`NONE / PAIR / JUNCTION / BOTH`

Historical reference: scene-level HGB trained from OOF local-score distributions.

### Порядок работы

1. Сначала восстановить exact historical relation representation, если она присутствует в archived source.
2. Если exact representation не восстановлена, сделать новый **минимальный, явный и документированный** `relation_features_v1`.
3. Geometry-only и CT-conditioned головы должны различаться только добавлением image evidence.
4. Не менять одновременно scene generator, graph constraints и relation model family.
5. Hyperparameters выбирать только train/val.

Рекомендуемые новые файлы:

`scripts/research/ccta_graph_lira_safe_repair/build_ct28_relation_features.py`

`scripts/research/ccta_graph_lira_safe_repair/train_ct28_relation_head.py`

Outputs:

- `relation_features_train_oof.csv`;
- `relation_predictions_val.csv`;
- `relation_model_protocol.json`;
- `relation_metrics_val.json`.

---

## 10. Сначала обязателен geometry-only end-to-end baseline

Перед заявлением CT gain новый canonical runner должен уметь выполнить ту же сцену без CT:

```text
geometry local evidence
-> relation type
-> Graph-LIRA
-> selective policy
-> repair / abstain
```

Это baseline для paired comparison.

Не требуется численно повторить старый 800-patient checkpoint на CT28: cohort и scene protocol другие.

Требуется, чтобы **на одном и том же новом CT28 scene pack** geometry-only и CT-conditioned варианты отличались только image evidence.

---

## 11. CT-conditioned Graph-LIRA

После geometry-only baseline добавить CT evidence в relation layer.

Не менять:

- candidate generation;
- patient split;
- graph topology;
- graph compatibility constraints;
- `tau = 0.85`;
- perturbation consistency gate `>= 0.60`;
- path rule.

Главный comparison:

```text
geometry-only Graph-LIRA
vs
CT-conditioned Graph-LIRA
```

Не сравнивать разные scene packs.

Рекомендуемая новая точка входа:

`scripts/research/ccta_graph_lira_safe_repair/run_ct28_graph_lira.py`

Runner должен сохранять explicit inputs/protocol/results и не зависеть от hidden pickle state.

---

## 12. Path stage

Path строится только после принятия structural relation.

Не использовать качество path как замену relation decision.

Если historical `a(e)` / max-min path implementation не удаётся восстановить точно:

- не придумывать новый path score внутри этого же эксперимента;
- сначала завершить relation + graph structural comparison;
- явно отметить path implementation gap;
- согласовать новый `a(e)` отдельно.

---

## 13. Что считать результатом на train/val

Нужна одна paired table для двух систем:

- geometry-only Graph-LIRA;
- CT-conditioned Graph-LIRA.

Обязательные structural metrics:

- repair-needed exact;
- false structural repair;
- incomplete / abstain;
- coverage;
- false among accepted;
- exact among accepted;
- relation-type accuracy / confusion matrix;
- per-patient metrics;
- patient-cluster uncertainty;
- results by scene type.

Local AUROC/AUPRC сохранить как diagnostic, но не делать главным end-to-end выводом.

---

## 14. Test firewall

Held-out JUNCTION patients:

`954, 958, 972, 973, 980, 984`

Пока НЕ:

- генерировать/просматривать JUNCTION test candidates;
- подбирать relation features по test;
- менять threshold после test;
- менять `tau`;
- менять consistency;
- выбирать между альтернативными scene protocols по test.

### Когда test можно открыть

Только когда в Git зафиксированы:

- scene protocol;
- local-score protocol;
- relation feature list;
- relation model family;
- validation-selected settings;
- graph constraints;
- `tau=0.85`;
- consistency `0.60`;
- structural metrics;
- exact run command.

После этого test запускается один раз.

---

## 15. Failure analysis после первого end-to-end результата

Не переходить сразу к CNN/ANZA.

Каждый крупный failure отнести к одному уровню:

1. correct candidate absent;
2. candidate ranking;
3. relation existence/type;
4. graph compatibility/conflict;
5. calibration / confidence;
6. path construction.

Особенно отдельно посмотреть:

- false structural repairs;
- abstentions на repair-needed scenes;
- пациентные outliers;
- ambiguous bifurcations;
- LAD / LCX where metadata/audit permits;
- degree-4 оставить audit-only, если новый protocol всё ещё не даёт достаточной выборки.

Output:

`artifacts/varvara/ct28_graph_lira/failure_cases_val.csv`

и короткий `RESULTS.md`.

---

## 16. Когда начинаем CNN vs ANZA

Только после получения первого clean end-to-end result.

Тогда:

```text
radial hand-crafted CT
vs
compact CNN
vs
compact ANZA
```

при полностью одинаковых:

- patients;
- candidates;
- scene protocol;
- relation task;
- Graph-LIRA;
- tau / consistency;
- evaluation.

Меняется только image encoder.

Если ANZA не выигрывает у compact CNN, это не отменяет основной результат CT-conditioned selective Graph-LIRA.

---

## 17. Что сейчас не делать

Не надо:

- снова тюнить JUNCTION negatives;
- менять 16 mm span;
- делать новую candidate geometry family без конкретного recall failure;
- открывать JUNCTION test;
- тюнить `tau` или consistency;
- строить большой CNN / Transformer / Mamba;
- делать ANZA до первого Graph-LIRA result;
- возрождать hidden pickle workflow;
- использовать anatomical segment labels как inference features;
- считать validation-tuned recall/FPR финальной generalization metric.

---

## 18. Что принести на следующий review

Минимальный результат следующего review:

1. `ARCHITECTURE_MAP.md`;
2. exact frozen JUNCTION local artifact files/config from текущего run;
3. `scene_protocol.json` + train/val scene table;
4. patient-OOF local evidence table;
5. geometry-only relation head;
6. CT-conditioned relation head;
7. geometry-only Graph-LIRA validation results;
8. CT-conditioned Graph-LIRA validation results;
9. paired structural comparison;
10. false-repair / abstention failure list;
11. exact reproduction commands.

Если до Graph-LIRA дойти за один заход не получается, промежуточная точка — **готовый reproducible scene pack + две relation heads на train/val**, но не новый локальный classifier.

---

## 19. Когда остановиться и спросить

Сразу остановиться и написать, если:

- historical scene semantics противоречат текущему CT28 setup;
- exact old relation feature construction не находится;
- graph optimizer невозможно отделить от missing pickle state;
- возникает необходимость менять `tau=0.85` или consistency `0.60`;
- для решения хочется открыть held-out test;
- новый scene protocol требует ground-truth anatomical labels как model inputs;
- path stage невозможно восстановить без нового определения `a(e)`.

В этих случаях не подбирать решение молча: зафиксировать проблему, что именно найдено в repo, и предложить 1–2 explicit alternatives.

---

## 20. Definition of done

Этап считается закрытым, когда существует воспроизводимое paired comparison:

```text
same CT28 scenes
same candidates
same graph
same safety policy

geometry-only relation evidence
        vs
geometry + CCTA relation evidence
```

и можно ответить:

> При фиксированном false-repair ориентире CCTA evidence реально увеличивает число правильно разрешённых repair-needed scenes или нет?

До этого CNN/ANZA не являются следующей задачей.
