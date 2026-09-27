# Сообщение Варваре — 2026-09-27

Вар, ваши вопросы оказались полезными — там действительно было несколько реальных дыр в старом handoff, а не то, что вы что-то не нашли.

Я всё дочистил в общей ветке:

`research/coronary-connectivity-repair`

После `git pull` начните, пожалуйста, с:

`docs/varvara/FINAL_HANDOFF_2026-09-27.md`

Там сейчас уже одна актуальная схема без противоречий со старыми экспериментами.

Главное: gap с CT28 PAIR-артефактами закрыт. В Git теперь есть exact pair plan, row-level predictions, summary, protocol и полный feature table в lossless compressed payload. Его можно восстановить одной командой, raw CT для воспроизведения старого PAIR baseline больше не нужен.

Если нужны именно сохранённые `geometry / radial_hu_summary_v1 / geometry_plus_radial_v1` joblib snapshots, они тоже теперь лежат в compact payload и восстанавливаются скриптом. Отдельно добавлен простой скрипт, который воспроизводит `geometry_hgb` из frozen pair plan.

То есть на поиски старых `scenes_full.pkl`, `models.joblib`, `relation_type_model.joblib` время больше не тратьте — это действительно старые несохранённые local artifacts.

По самой задаче важное уточнение остаётся таким:

сильный результат `geometry + radial CT = AUROC 0.9847 / recall 82.63% / FPR 1.80% / precision 97.87%` — это пока **PAIR+CT**, не полный Graph-LIRA.

Следующий новый блок — **JUNCTION+CT на тех же 28 пациентах и том же split**:

```text
JUNCTION geometry
vs
JUNCTION CT
vs
JUNCTION geometry + CT
```

После этого уже объединяем:

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

Старый `ct_scene_graph_lira_hybrid.py` буквально повторять не надо — это six-patient pilot с add/veto логикой, и на held-out test он увеличивал false structural repair.

Конкретный task расписан в:

`docs/varvara/CURRENT_TASK.md`

А из отрицательных результатов достаточно прочитать:

`docs/varvara/NEGATIVE_RESULTS_THAT_MATTER.md`

Если при воспроизведении baseline или при формировании JUNCTION dataset где-то возникает неоднозначность по коду/математике — лучше сразу пишите, разберём конкретное место, чтобы вы не тратили время на восстановление старой исследовательской истории.
