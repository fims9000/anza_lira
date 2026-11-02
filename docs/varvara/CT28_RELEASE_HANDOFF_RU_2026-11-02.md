# Варвара: передача CT28 Graph-LIRA — инженерный релиз-кандидат

Дата: 2026-11-02. Ветка с изменениями: `audit/ct28-pretest-safety-2026-11-02`. Целевая ветка Варвары: `research/coronary-connectivity-repair`. PR: https://github.com/fims9000/anza_lira/pull/1

## Что не меняем

K10/15 остаётся **единственным выбранным validation baseline**: 155/206 точных до gate; после — 52 точных, 0 ложных среди 368 сцен, из них 48 PAIR, 2 JUNCTION и 2 BOTH. Заморожены `tau=0.85`, consistency `0.60`, 15 evaluation perturbations, модельные веса и клинический test firewall. Последний исходный коммит Варвары — `55e52efa`.

## Что уже подготовлено в PR

1. **Корректность safety gate.** Отвергаем `BOTH`, когда невозможно выбрать две совместимые связи. Регрессионные тесты подтверждают, что сохранённые 16/52 решений не изменились. Тесты рассматривают некорректные типы, отсутствие кандидатов, низкую consistency и смену endpoint IDs.
2. **Топологически контролируемое построение маски.** `ct28_mask_reconstruction.py` переводит принятые `paths_voxel` в дополнительные воксели, используя физический affine. Отклоняет попадание на третью компоненту, пересечение двух отношений BOTH, превышение объёма и отсутствие CT-support при переданной маске. На отказе исходная маска не меняется.
3. **Постфактум метрики 3D.** `ct28_mask_metrics.py`: Dice/clDice, компоненты до/после, объём новых вокселей, доля подтверждённых разметкой добавлений. GT не попадает в принятие решений.
4. **Математика для следующего эксперимента.** `ct28_structured_objective.py`: дифференцируемая нормализованная вероятность по допустимым PAIR/JUNCTION/BOTH candidate set и JS-consistency между clean и perturbations. Есть synthetic unit tests, autograd и численная проверка градиента. **Обучения на CT28 не проводилось.** Не пытаться backpropagировать эту функцию через HGB: для нового опыта нужна дифференцируемая relation/identity head.
5. **Правильная статистика.** `ct28_selected_statistics.py`: пациентский paired bootstrap из **актуального** `selected_validation/graph_lira_predictions_val.csv`. Старые root-level CSV не брать в статью.
6. **CI.** Автоматизированные smoke-тесты на PR отдельно проверяют safety/3D и structured loss, не открывая test.

## Что делать на машине с архивом CT и локальными моделями

**Шаг 1 — проверить инженерную сборку (без raw CT).**

```bash
python -m pytest -q tests/test_ct28_safety_gate.py tests/test_ct28_scene_shortcuts.py tests/test_ct28_candidate_permutation.py tests/test_ct28_mask_reconstruction.py tests/test_ct28_mask_reconstruction_cli.py tests/test_ct28_mask_metrics.py tests/test_ct28_mask_metrics_cli.py tests/test_ct28_selected_statistics.py tests/test_ct28_structured_objective.py
python scripts/research/ccta_graph_lira_safe_repair/ct28_selected_statistics.py --out-md /tmp/ct28_results.md
```

**Шаг 2 — один train-only full-mask пилот.** Берём train-пациента, формируем документированный искусственный разрыв *внутри* полного сосуда (или используем patient-disjoint предсказанную разорванную маску), **не передавая исходную полную разметку в solver**. Строим путь по CT и подаём его в новый mask-reconstruction CLI. Выбираем radius и max-new-voxels только на train, фиксируем входные SHA/affine и все причины abstain. На train проверяем, соединены ли именно требуемые компоненты и нет ли третьих контактов. Если условие слишком жёсткое, записываем failure case, не ослабляем его вслепую.

**Шаг 3 — воспроизводимая validation-проверка.** После фиксации правил сравниваем на тех же пяти validation-пациентах: no repair; geometry-only selective; CT-conditioned selective; CT-conditioned+mask gate. Репортим accepted/false, mean delta Dice/clDice по пациентам, branch errors и runtime. Это ещё development result, а не финальный test.

**Шаг 4 — один научный эксперимент для JUNCTION/BOTH.** Используем именно joint structured type/identity JS objective из `CT28_RELEASE_ARCHITECTURE_MATH_2026-11-02.md`. Один train-only выбор `lambda`/temperature через grouped CV, а не новый большой sweep на пяти validation пациентах. Важно сохранить identical candidate pool и сопоставимые scores для исходного и нового варианта. Если прироста в JUNCTION нет, фиксируем отрицательный результат и переходим к реальным segmentation errors.

**Шаг 5 — решение перед test.** Сначала freeze scene generator, контроли shortcut, модель, пути, radius/budget, GT post-hoc протокол и критерии успеха. Потом однократное открытие Graph-LIRA held-out test без перенастройки. Данные 954/958/972/973/980/984 здесь не читались.

## Научная граница

Сейчас можно заявлять выборочное восстановление структурных отношений в контролируемых CT28-сценах, но нельзя заявлять улучшение реальной 3D-сегментации или клиническую безопасность. Приоритет — доказать пользу на damaged predicted masks, а не максимизировать 52/0 на тех же пяти людях.

Полный метод и формулы: [CT28_RELEASE_ARCHITECTURE_MATH_2026-11-02.md](CT28_RELEASE_ARCHITECTURE_MATH_2026-11-02.md).
