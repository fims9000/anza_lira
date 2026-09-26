# Короткое сообщение Варваре

Привет! Я собрал отдельную ветку по направлению восстановления связности коронарных сосудов на CCTA:

`research/coronary-connectivity-repair`

Начать лучше с файла:

`VARVARA_START_HERE.md`

Там коротко описано, что уже сделано и в чём сейчас основная идея.

Если совсем в двух словах: после сегментации сосуды могут быть разорваны, но просто соединять ближайшие части опасно — особенно около ветвлений. Поэтому мы разделяем задачу на геометрические кандидаты, подтверждение по исходному CCTA, графовую проверку совместимости и selective repair/abstain.

На 28 matched ImageCAS/ImageCAS-X пациентах уже есть сильный результат: geometry + radial CT даёт 82.63% recall при 1.80% FPR на held-out test, тогда как сильный geometry baseline даёт 37.72% recall при 1.20% FPR.

Следующая главная задача — перенести CT evidence внутрь Graph-LIRA и проверить end-to-end structural repair при уже замороженных `tau=0.85` и consistency `0.60`.

После этого — чистая архитектурная абляция radial CT vs compact CNN vs ANZA.

Подробно:
- `docs/varvara/ARTICLE_DIRECTION.md` — как это выглядит как статья;
- `docs/varvara/RESULTS_TO_USE.md` — какие результаты сейчас реально использовать;
- `docs/varvara/ROADMAP.md` — что делать по шагам;
- `docs/varvara/REPO_MAP.md` — где что лежит в репозитории.

Старые exploratory и неудачные эксперименты сохранены в технической части репозитория, но читать их подряд не нужно.


## 2026-09-26 — current execution pack

For the current task, read these files before touching the old exploratory scripts:

- `docs/varvara/CURRENT_TASK.md` — exact next task: 28-patient JUNCTION+CT, then CT-conditioned relation head, then frozen Graph-LIRA;
- `docs/varvara/NEGATIVE_RESULTS_THAT_MATTER.md` — only the negative results that materially constrain the architecture;
- `docs/varvara/REPRODUCE_CT28_PAIR_BASELINE.md` — exact interpretation and reproduction of the 28-patient PAIR baseline;
- `docs/varvara/ARTIFACT_MAP_AND_CURRENT_TASK_2026-09-26.md` — which models/files really exist and which old local artifacts are missing;
- `artifacts/varvara/ct28_pair/` — compact collaboration metadata/provenance;
- `scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_from_features.py` — retrains and saves the three lightweight PAIR models once the generated feature table is available.

Do not search for the old local `models.joblib`, `relation_type_model.joblib` or `scenes_full.pkl` as if they were hidden somewhere in the branch. They are not committed canonical artifacts.
