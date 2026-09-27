# Варвара — начать отсюда

Каноническая ветка:

`research/coronary-connectivity-repair`

После `git pull` открыть:

`docs/varvara/FINAL_HANDOFF_2026-09-27.md`

И больше ничего не искать до его прочтения.

Быстрая integrity-проверка:

```bash
python scripts/research/ccta_graph_lira_safe_repair/verify_varvara_collab_pack.py
```

Если она проходит, frozen CT28 PAIR pack на месте.

## Что уже есть

28-patient binary PAIR result:

**geometry + radial 2.5-D CCTA**

- AUROC 0.9847;
- recall 82.63%;
- FPR 1.80%;
- precision 97.87%.

PAIR features/predictions/models are reproducible from `artifacts/varvara/ct28_pair/`.

## Что делать сейчас

Не Graph-LIRA integration сразу.

Сначала закрыть:

**28-patient JUNCTION+CT**

на том же frozen split.

Raw CT access для этой новой feature extraction:

`docs/varvara/CT28_DATA_ACCESS.md`

После этого:

`PAIR_CT + JUNCTION_CT -> NONE/PAIR/JUNCTION/BOTH -> frozen Graph-LIRA -> repair/abstain`.

Все прямые ответы на вопросы про HGB, old pkl/joblib, `ct_scene_graph_lira_hybrid.py`, `PAIR_IMG/JUNC_IMG`:

`docs/varvara/ANSWER_TO_QUESTIONS_2026-09-26.md`
