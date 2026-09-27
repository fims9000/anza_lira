# Сообщение Варваре — актуальная версия

Вар, проверили ваши вопросы и после этого ещё раз привели handoff в порядок. Вы ничего не пропустили: часть старых Graph-LIRA `.pkl/.joblib` действительно никогда не была сохранена как canonical artifacts.

Сейчас всё, что нужно для старта, находится в ветке:

`research/coronary-connectivity-repair`

После `git pull` откройте **один файл**:

`docs/varvara/FINAL_HANDOFF_2026-09-27.md`

Он сейчас главный. Старые roadmap/notes можно не читать, пока не понадобится конкретный контекст.

Что уже закрыто:

- CT28 PAIR feature table восстановлен и сохранён в Git в компактном lossless payload;
- row-level predictions сохранены;
- frozen pair plan сохранён;
- есть скрипты восстановления features;
- есть скрипт retrain трёх PAIR моделей;
- есть convenience snapshots `.joblib`;
- сильный `geometry_hgb` тоже воспроизводится отдельным скриптом.

То есть **raw CT и старые pickle-файлы для старта PAIR baseline больше не нужны**.

Если хотите проверить baseline локально:

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_features.py
python scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_from_features.py \
  artifacts/varvara/ct28_pair/expanded_relation_features.csv \
  --out-dir ct28_pair_models
```

Ожидаемый held-out результат `geometry_plus_radial_v1`:

- AUROC 0.9847;
- recall 82.63%;
- FPR 1.80%;
- precision 97.87%.

Важно: это **локальная PAIR relation модель**, а не полный CT-conditioned Graph-LIRA.

Поэтому текущая новая задача теперь очень конкретная:

```text
JUNCTION geometry
vs
JUNCTION CT
vs
JUNCTION geometry + CT
```

на тех же 28 пациентах и том же frozen patient split.

После этого уже собираем:

```text
PAIR geometry + PAIR CT
+
JUNCTION geometry + JUNCTION CT
        ↓
NONE / PAIR / JUNCTION / BOTH
        ↓
frozen Graph-LIRA
        ↓
tau = 0.85
consistency = 0.60
        ↓
repair / abstain
```

Старый `ct_scene_graph_lira_hybrid.py` воспринимайте только как pilot. Его add/veto integration повторять как финальный метод не нужно: на held-out test он увеличивал false structural repair.

Старые `PAIR_IMG/JUNC_IMG` — тоже pilot на маленьком cohort, их веса не считаем canonical.

Подробная текущая задача:
`docs/varvara/CURRENT_TASK.md`

Только важные отрицательные результаты:
`docs/varvara/NEGATIVE_RESULTS_THAT_MATTER.md`

Ответы именно на ваши вопросы по моделям/артефактам:
`docs/varvara/ANSWER_TO_QUESTIONS_2026-09-26.md`

Если при первом запуске что-то не поднимается по окружению, просто пришлите команду + полный traceback. Архитектуру пока менять не надо — сначала добиваем воспроизводимый JUNCTION+CT baseline на том же protocol.
