# S58 — экспорт целых машинных 2D-цепочек, а не объявление физических треков

**26 октября 2026 · ветка `research/bubble-chamber-graph-lira` · пользователю не требуется выполнять ручную разметку.**

S56 выбрал 37 геометрически согласованных стыков из 182 предварительных S39-предложений. S58 сохраняет исходные машинные полилинии и строит графовые компоненты только из этих стыков. Каждый связанный компонент становится **упорядоченной гипотезой о 2D-пути**; отдельные участки соединения представлены как `PREDICTED_GAP_NOT_OBSERVED_AND_NOT_PHYSICS_GT`, а не в виде подтверждённых пикселей физического трека. Оригинальные 567 кандидатов и 511 графовых узлов не изменяются.

**Непосредственно на настоящих H50Z:** 35 связанных компонент, из которых **34 состоят из двух исходных машинных линий** и только **одна — из четырёх линий**, соединённых тремя гипотетическими разрывами. Физически подтверждено **0** соединений; идентичность всех частиц остаётся `UNKNOWN`. Следовательно, реальный конвейер ещё не демонстрирует массовое восстановление целых длинных физических треков. Нельзя подменять этот факт хорошими числами S56 на искусственных разрывах.

Целостность упорядоченного пути обеспечивается ограничением на один стык для каждого конца и запретом разветвления в рассматриваемой подзадаче открытой 2D-линии. При включении неизвестных развилок, вершин и замкнутых траекторий необходимы иные физико-геометрические правила; в S58 такого восстановления нет. Весь вывод — диагностический слой поверх существующих H50Z машинных кандидатов, а не обучающая GT.

В приватном ZIP S58 присутствуют `S58_MACHINE_PATH_COMPONENTS_PRIVATE.json` с полными исходными/предполагаемыми сегментами и `S58_LONGEST_MACHINE_PATH_PRIVATE.jpg`, а также вся офлайн-галерея S57. Эти файлы содержат фото и координаты; **их нельзя помещать в публичный GitHub**. Публичный `S58_PUBLIC_MACHINE_PATH_COMPONENTS.json` содержит только агрегированные числа.

Воспроизведение:

```bash
python scripts/research/bubble_chamber/s58_export_native_path_hypotheses.py \
  --s39-private /PRIVATE/S39_GRAPH_RELATIONS_PRIVATE.json \
  --s56-private /PRIVATE/S56_NATIVE_PATH_GRAPH_PRIVATE.json \
  --photo-tiles /PRIVATE/source_39_H50Z_crops \
  --out-private /PRIVATE/S58_MACHINE_PATH_COMPONENTS_PRIVATE.json \
  --out-public docs/bubble_chamber/S58_PUBLIC_MACHINE_PATH_COMPONENTS.json \
  --private-preview /PRIVATE/S58_LONGEST_MACHINE_PATH_PRIVATE.jpg
python scripts/research/bubble_chamber/s58_package_private_paths.py \
  --s57-dir /PRIVATE/ANZA_LIRA_S57_MACHINE_ONLY_SHOWCASE \
  --s58-private /PRIVATE/S58_MACHINE_PATH_COMPONENTS_PRIVATE.json \
  --s58-public docs/bubble_chamber/S58_PUBLIC_MACHINE_PATH_COMPONENTS.json \
  --preview /PRIVATE/S58_LONGEST_MACHINE_PATH_PRIVATE.jpg \
  --out-dir /PRIVATE/ANZA_LIRA_S58_MACHINE_ONLY_SHOWCASE \
  --out-zip /PRIVATE/ANZA_LIRA_S56_S58_PRIVATE_MACHINE_DEMO.zip
```
