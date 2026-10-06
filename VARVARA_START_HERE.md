# Варвара — начать отсюда

Каноническая рабочая ветка:

`research/coronary-connectivity-repair`

С 2026-10-06 вся актуальная coronary / CCTA работа снова собрана в этой
ветке. Старые experiment/research branches — только checkpoints.

После обновления репозитория читать в таком порядке:

1. `docs/varvara/CURRENT_TASK.md`
2. `docs/varvara/TASK_CT28_CT_CONDITIONED_GRAPH_LIRA_2026-10-06.md`

Второй файл — полное текущее ТЗ: зачем нужен следующий этап, какие inputs уже
есть, что именно строить, какие части historical Graph-LIRA использовать как
reference, какие outputs сохранить и когда можно открывать held-out test.

## Коротко о текущем состоянии

PAIR local CCTA evidence уже frozen/reproducible.

JUNCTION train/validation local evidence тоже готов к следующему уровню:

- frozen controlled generator;
- 143 positives + 1,666 negatives;
- 17 train / 5 validation patients;
- candidate recall 100%;
- CT tensor `3 x 17 x 25`;
- geometry / CT / geometry+CT baselines;
- held-out JUNCTION test пока закрыт.

Следующая задача — **не CNN/ANZA и не новый generator**.

Следующая задача:

```text
PAIR + JUNCTION local evidence
        ->
NONE / PAIR / JUNCTION / BOTH
        ->
frozen Graph-LIRA
        ->
repair / abstain
```

После первого clean end-to-end результата и failure analysis уже будет
делаться radial vs compact CNN vs compact ANZA.

## JUNCTION feature table

Восстановить committed train/val feature table:

```bash
python artifacts/varvara/ct28_junction/restore_junction_features.py
```

## Historical Graph-LIRA reference code

Для чтения старых archived scripts:

```bash
bash scripts/research/ccta_graph_lira_safe_repair/restore_historical_graph_lira_sources.sh
```

Они только reference. Новый canonical runner не должен зависеть от hidden
historical pickle/joblib state.

## Test firewall

Не открывать JUNCTION test patients:

`954, 958, 972, 973, 980, 984`

до фиксации scene/relation/graph protocol на train/validation.
