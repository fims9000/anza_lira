# Варвара — начать отсюда

Каноническая ветка:

`research/coronary-connectivity-repair`

## Сейчас читать только это

После `git pull` первым открыть:

`docs/varvara/FINAL_HANDOFF_2026-09-27.md`

Это актуальная точка входа после всех вопросов по HGB, PAIR models, старым .pkl/.joblib, PAIR_IMG/JUNC_IMG и CT28 данным.

Если старые roadmap/notes где-то формулируют задачу шире или иначе, приоритет у FINAL_HANDOFF.

## Коротко: что уже доказано

На 28 matched ImageCAS / ImageCAS-X пациентах проверена локальная binary PAIR relation задача.

Split:

- 17 train;
- 5 validation;
- 6 held-out test.

Лучший текущий PAIR result:

**geometry + radial 2.5-D CCTA**

- AUROC 0.9847;
- recall 82.63%;
- FPR 1.80%;
- precision 97.87%.

Это значит: CCTA context даёт полезный patient-general signal для PAIR identity.

Это **не** означает, что полный CT-conditioned Graph-LIRA уже проверен.

## Что с данными для старта

CT28 PAIR collaborator pack уже complete.

`artifacts/varvara/ct28_pair/` содержит exact pair plan, predictions, protocol, provenance и compact lossless full feature table.

Для воспроизведения PAIR baseline raw multi-GB CT больше не нужен.

## Текущая задача

Незакрытый блок:

**JUNCTION + CT на тех же 28 пациентах и том же frozen split.**

Сравнить:

1. JUNCTION geometry;
2. JUNCTION CT;
3. JUNCTION geometry + CT.

После этого:

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

## Что читать дальше при необходимости

1. `docs/varvara/CURRENT_TASK.md`
2. `docs/varvara/NEGATIVE_RESULTS_THAT_MATTER.md`
3. `docs/varvara/ANSWER_TO_QUESTIONS_2026-09-26.md`
4. `docs/varvara/REPRODUCE_CT28_PAIR_BASELINE.md`
5. `docs/varvara/REPO_MAP.md`

Для широкого научного контекста:
- `docs/varvara/ARTICLE_DIRECTION.md`
- `docs/varvara/RESULTS_TO_USE.md`
- `docs/varvara/ROADMAP.md`

Не надо начинать с просмотра всей истории exploratory scripts.
