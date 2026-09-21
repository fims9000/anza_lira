# Roadmap для Варвары

Ниже не список всех возможных идей, а конкретная последовательность, по которой сейчас стоит двигать работу.

## Этап 0. Разобраться в текущей постановке

Сначала прочитать:

1. `VARVARA_START_HERE.md`
2. `docs/varvara/ARTICLE_DIRECTION.md`
3. `docs/varvara/RESULTS_TO_USE.md`

После этого должно быть понятно:

- почему простой nearest-neighbor repair недостаточен;
- чем candidate ranking отличается от relation acceptance;
- зачем нужен CT;
- зачем нужен Graph-LIRA;
- зачем нужен abstention.

Не нужно начинать с просмотра всех scripts.

## Этап 1. Зафиксировать текущий CT baseline

Текущая сильная локальная модель:

`geometry + radial 2.5-D CT`

Она уже прошла 28-patient promotion check.

Ничего в этом baseline сейчас не улучшать.

Не менять:

- train/val/test split;
- controlled pair definitions;
- threshold selection rule;
- test thresholds.

Цель этапа — считать этот результат отправной точкой.

## Этап 2. Перенести CT evidence внутрь Graph-LIRA

Это главный следующий технический этап.

Нужно взять уже рассчитанный image-conditioned relation signal и добавить его в существующий relation layer.

Логика:

```text
candidate
    ↓
geometry evidence
    +
CT evidence
    ↓
PAIR / JUNCTION / BOTH / NONE
    ↓
global Graph-LIRA
```

Важно: не переписывать candidate generator и не менять весь pipeline одновременно.

### Что оставить frozen

- geometry candidate generation;
- compatibility logic;
- global Graph-LIRA constraints;
- relation confidence tau = 0.85;
- perturbation consistency = 0.60.

### Что можно менять

Только способ, которым CT evidence входит в relation existence / relation type calibration.

### Что должно получиться на выходе

Таблица:

| model | repair-needed exact | false structural | incomplete | coverage | false among accepted | exact among accepted |
|---|---:|---:|---:|---:|---:|---:|

Сравнить:

- geometry-only Graph-LIRA;
- CT-conditioned Graph-LIRA.

## Этап 3. Проверить переносимость по пациентам

Нельзя ограничиваться одной общей accuracy.

Нужно:

- per-patient metrics;
- patient-cluster bootstrap;
- paired bootstrap CT vs geometry;
- отдельный разбор test patients;
- confidence intervals.

Если выигрыш есть только у одного пациента, модель не считается устойчивой.

## Этап 4. Проверить сложные сосудистые группы

Заранее важные группы:

- LAD;
- LCX;
- OM1;
- OM2;
- IM;
- D2;
- R-PLA;
- high-degree junctions.

Отдельно смотреть неправильные возможные соединения:

- D1 ↔ LAD;
- D2 ↔ LAD;
- LCX ↔ OM1;
- LCX ↔ OM2;
- LCX ↔ LM;
- R-PDA ↔ RCA;
- R-PLA ↔ RCA.

Цель — понять не только «стало ли лучше», а где именно CT помогает и где остаются опасные ошибки.

## Этап 5. Только после Graph-LIRA — ANZA

ANZA сейчас не надо сразу делать центральной частью.

Сначала должен быть доказан сам image-conditioned graph approach.

После этого проводится чистый architecture ablation:

```text
same patients
same candidate relations
same train/val/test
same Graph-LIRA
same selective policy

radial hand-crafted CT
vs
compact conventional CNN
vs
compact ANZA encoder
```

Если ANZA выигрывает в этих условиях, можно говорить про отдельный архитектурный вклад.

Если нет — основная статья всё равно остаётся полноценной за счёт risk-controlled graph repair.

## Этап 6. Если radial CT перестанет хватать

Следующая ступень representation:

1. candidate-aligned 3-D tube;
2. full cross-section local encoder;
3. CNN/ANZA tokens вдоль сосудистого сегмента;
4. только затем Transformer/Mamba для sequence context.

Не начинать с Transformer над mean/std признаками — этот путь уже не показал преимущества.

## Этап 7. Подготовить статью

Параллельно после Graph-LIRA результата можно собирать:

### Introduction

- проблема разрывов в coronary segmentation;
- риск неправильного post-hoc connection;
- недостаточность только геометрии;
- selective structural repair.

### Related Work

- topology-preserving segmentation: clDice, Skeleton Recall;
- explicit reconnection: OGMC;
- coronary reconnection/reconstruction: CorSegRec;
- image-to-graph approaches;
- отличие: risk-controlled repair with abstention.

Готовая база:

`docs/research/ccta_graph_lira_safe_repair/RELATED_WORK_AND_DIFFERENTIATION_2026.md`

### Methods

- data;
- candidate graph;
- geometry evidence;
- radial CCTA evidence;
- relation head;
- Graph-LIRA;
- selective policy;
- metrics.

### Experiments

- geometry baselines;
- CT local relation experiment;
- Graph-LIRA integration;
- anatomy subgroups;
- risk/coverage;
- architecture ablation.

## Правила, которые нельзя нарушать

1. Не подбирать threshold по test.
2. Не переносить anatomical labels в model inputs.
3. Не менять split ради красивого результата.
4. Не смешивать сразу новый encoder, новый graph algorithm и новый threshold.
5. Не выдавать 6 test patients за клиническую population validation.
6. Не писать, что ANZA лучше, пока нет clean ablation.
7. Не скрывать false structural repairs: это одна из главных метрик работы.

## Когда текущий этап можно считать завершённым

До перехода к полноценной ANZA-части должны быть готовы:

- CT-conditioned Graph-LIRA;
- frozen selective evaluation;
- patient-level bootstrap;
- hard-anatomy breakdown;
- clean geometry vs CT-Graph comparison.

После этого появляется нормальная точка для отдельной архитектурной задачи Варвары: сделать compact ANZA encoder и проверить его против обычного CNN и radial baseline.
