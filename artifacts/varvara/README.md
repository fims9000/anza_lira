# CT28: карта артефактов

- [ct28_pair/](ct28_pair/README.md) — завершённый отдельный PAIR baseline: frozen plan, lossless feature payload, predictions, модели и проверка alignment. Для воспроизведения raw CT не нужен.
- [ct28_junction/](ct28_junction/README.md) — frozen JUNCTION plan/endpoints, CT feature payload и ранние local baselines на train/validation. Исходные VTK и сегментации хранятся в игнорируемой Git папке `varvara/anza_lira_local_data/Varvara_CT28_junction_source_pack/`.
- [ct28_graph_lira/](ct28_graph_lira/README.md) — текущая scene-level разработка. Выбранный validation-претендент K10/15: 52 точных и 0 ложных repair после gate; test не открыт. Внутри отдельно описаны абляции, consistency и новый path protocol.

Проверка исходного PAIR collaboration pack:

```bash
python scripts/research/ccta_graph_lira_safe_repair/verify_varvara_collab_pack.py
```

Raw CCTA хранится вне Git. Источник, frozen cohort и alignment checks описаны в [CT28_DATA_ACCESS.md](../../docs/varvara/CT28_DATA_ACCESS.md). Актуальный порядок работы — в [CURRENT_TASK.md](../../docs/varvara/CURRENT_TASK.md); [handoff 2026-09-27](../../docs/varvara/FINAL_HANDOFF_2026-09-27.md) отражает старый этап.
