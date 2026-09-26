# Варвара — начать отсюда

Эта ветка собрана как рабочая версия проекта, которую можно читать без истории всех проб, неудачных запусков и промежуточных гипотез.

Ветка:

`research/coronary-connectivity-repair`

Главная тема работы:

**контролируемое восстановление связности коронарных сосудов по CCTA с использованием локальной геометрии, информации исходного КТ-изображения и глобальных графовых ограничений.**

Проще: после сегментации коронарного дерева иногда появляются разрывы. Соединять ближайшие куски напрямую опасно — около бифуркаций можно соединить не те ветви. Поэтому задача формулируется не как «дорисовать сосуд любой ценой», а как:

1. найти возможные продолжения;
2. оценить, есть ли между ними подтверждение в исходном CCTA;
3. проверить, совместимо ли это соединение со всей структурой дерева;
4. выполнить repair только если уверенность достаточна;
5. иначе оставить случай на проверку.

Это и есть основная логика Graph-LIRA в текущем направлении.

## Что уже сделано

Есть проверенная связка оригинальных ImageCAS CCTA и разметки ImageCAS-X.

Для расширенного эксперимента использовано 28 пациентов:

- train: 17;
- validation: 5;
- test: 6.

Все 28 CT прошли проверку соответствия ImageCAS-X по shape, spacing и affine.

На локальной задаче определения правильного продолжения сосудистой ветви получен сильный результат.

Held-out test, порог выбирается только на validation при FPR <= 5%:

| метод | Recall | FPR | Precision | AUROC |
|---|---:|---:|---:|---:|
| geometry HGB | 37.72% | 1.20% | 96.92% | 0.9685 |
| radial 2.5-D CT | 68.86% | 4.19% | 94.26% | 0.9440 |
| **geometry + radial CT** | **82.63%** | **1.80%** | **97.87%** | **0.9847** |

То есть CCTA-контекст действительно добавляет информацию, которой одной геометрии не хватает.

Это подтверждено patient-cluster bootstrap: для geometry+CT против geometry HGB медианный прирост recall около +44.7 п.п., а 95% интервал остаётся полностью выше нуля.

## Что это уже позволяет говорить

Можно обоснованно говорить, что:

- геометрия хорошо ранжирует кандидатов, но плохо отвечает на вопрос «можно ли безопасно принимать это соединение»;
- локальный CCTA-контекст существенно повышает долю правильно принимаемых связей;
- эффект наблюдается не только в pooled-метриках, но и при bootstrap по пациентам;
- особенно заметный выигрыш есть на заранее зафиксированных сложных ветвях: LAD, OM, IM, D2, R-PLA;
- поэтому image-conditioned relation evidence стоит переносить в Graph-LIRA.

## Чего пока говорить нельзя

Пока не надо писать, что:

- задача полностью решена end-to-end;
- Graph-LIRA с CT уже доказал улучшение всего сосудистого дерева;
- ANZA уже лучше CNN;
- false repair risk доказан на клинической популяции;
- модель готова к клиническому использованию.

Эти шаги ещё впереди.

## Куда идти дальше

Следующий главный эксперимент:

```text
geometry candidates
        +
CCTA relation evidence
        ↓
PAIR / JUNCTION / BOTH / NONE
        ↓
Graph-LIRA
        ↓
tau = 0.85
        ↓
consistency = 0.60
        ↓
repair / abstain
```

При этом `tau=0.85` и `consistency=0.60` не подбираются заново на test.

После этого, если CT-сигнал сохраняет преимущество и не ломает false-repair control, делается чистая архитектурная абляция:

```text
radial CT
vs
compact CNN
vs
ANZA encoder
```

## Что читать

В первую очередь:

1. `docs/varvara/ARTICLE_DIRECTION.md`
2. `docs/varvara/RESULTS_TO_USE.md`
3. `docs/varvara/ROADMAP.md`
4. `docs/varvara/REPO_MAP.md`

Остальной репозиторий — технический архив и воспроизводимость. Его не нужно читать подряд.

Сырые медицинские данные в Git не хранятся.


## 2026-09-26 — current execution pack

For the current task, read these files before touching the old exploratory scripts:

- `docs/varvara/CURRENT_TASK.md` — exact next task: 28-patient JUNCTION+CT, then CT-conditioned relation head, then frozen Graph-LIRA;
- `docs/varvara/NEGATIVE_RESULTS_THAT_MATTER.md` — only the negative results that materially constrain the architecture;
- `docs/varvara/REPRODUCE_CT28_PAIR_BASELINE.md` — exact interpretation and reproduction of the 28-patient PAIR baseline;
- `docs/varvara/ARTIFACT_MAP_AND_CURRENT_TASK_2026-09-26.md` — which models/files really exist and which old local artifacts are missing;
- `artifacts/varvara/ct28_pair/` — compact collaboration metadata/provenance;
- `scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_from_features.py` — retrains and saves the three lightweight PAIR models once the generated feature table is available.

Do not search for the old local `models.joblib`, `relation_type_model.joblib` or `scenes_full.pkl` as if they were hidden somewhere in the branch. They are not committed canonical artifacts.

## Актуальное уточнение задачи

После проверки артефактов текущая задача уточнена.

Прочитать обязательно:

- `docs/varvara/ARTIFACT_MAP_AND_CURRENT_TASK_2026-09-26.md`
- `docs/varvara/CURRENT_TASK.md`
- `docs/varvara/NEGATIVE_RESULTS_THAT_MATTER.md`
- `docs/varvara/REPRODUCE_CT28_PAIR_BASELINE.md`

Компактные derived artifacts для старта:

`artifacts/varvara/`

Главный недостающий научный блок сейчас: **28-patient JUNCTION+CT evidence**, затем CT-conditioned `NONE / PAIR / JUNCTION / BOTH` head и только после этого frozen Graph-LIRA evaluation.
