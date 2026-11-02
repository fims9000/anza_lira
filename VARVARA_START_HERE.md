# Варвара — начать отсюда

Каноническая ветка: `research/coronary-connectivity-repair`.

1. [Текущее состояние и следующий шаг](docs/varvara/CURRENT_TASK.md).
2. [Выбранный CT28 Graph-LIRA validation-вариант и источники результатов](artifacts/varvara/ct28_graph_lira/README.md).
3. [Исходное ТЗ](docs/varvara/TASK_CT28_CT_CONDITIONED_GRAPH_LIRA_2026-10-06.md) и [validation plan](docs/varvara/CT28_GRAPH_LIRA_VALIDATION_PLAN.md) — требования и последовательность, написанные до текущих результатов.

Текущий претендент **K10/15**: на пяти validation-пациентах после gate принято 52 точных и 0 ложных repair. Это не held-out test. Подробная история выбранных и отклонённых вариантов находится внутри `artifacts/varvara/ct28_graph_lira/`; старые результаты не следует подменять этим итогом или наоборот.

Для воспроизведения отдельного PAIR baseline: [PAIR pack](artifacts/varvara/ct28_pair/README.md). Frozen JUNCTION plan и признаки: [JUNCTION pack](artifacts/varvara/ct28_junction/README.md); исходные VTK и сегментации — в игнорируемой Git папке `varvara/anza_lira_local_data/Varvara_CT28_junction_source_pack/`. [Handoff от 2026-09-27](docs/varvara/FINAL_HANDOFF_2026-09-27.md) сохранён как исторический документ.

Held-out Graph-LIRA test остаётся закрытым до фиксации scene, relation, graph и path protocol на train/validation. `tau=0.85` и consistency `0.60` не подбираются на test.
