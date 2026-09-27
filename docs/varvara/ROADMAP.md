# Roadmap для Варвары — актуальная последовательность

Актуализировано: 2026-09-27.

Для текущего исполнения сначала читать:

`docs/varvara/FINAL_HANDOFF_2026-09-27.md`

Этот roadmap описывает последовательность исследования, но не заменяет FINAL_HANDOFF.

## Этап 0. Проверить collaborator pack

Из корня репозитория:

```bash
python scripts/research/ccta_graph_lira_safe_repair/verify_varvara_collab_pack.py
```

PAIR baseline уже frozen и не является новой задачей.

## Этап 1. При необходимости воспроизвести CT28 PAIR

Восстановить features и retrain:

```bash
python scripts/research/ccta_graph_lira_safe_repair/restore_ct28_pair_features.py
python scripts/research/ccta_graph_lira_safe_repair/train_ct28_pair_from_features.py \
  artifacts/varvara/ct28_pair/expanded_relation_features.csv \
  --out-dir ct28_pair_models
```

Главный frozen result `geometry_plus_radial_v1`:

- AUROC 0.9847;
- recall 82.63%;
- FPR 1.80%;
- precision 97.87%.

Не тюнить его заново.

## Этап 2. Новый обязательный блок — JUNCTION+CT

На тех же 28 пациентах и том же split построить junction-specific dataset/representation.

Сравнить:

1. JUNCTION geometry;
2. JUNCTION CT;
3. JUNCTION geometry + CT.

Raw CT source/alignment:

`docs/varvara/CT28_DATA_ACCESS.md`

Не использовать anatomical branch names как inference features.

## Этап 3. CT-conditioned relation-type head

Только когда есть patient-general PAIR_CT и JUNCTION_CT:

```text
PAIR geometry + PAIR CT
+
JUNCTION geometry + JUNCTION CT
        ↓
NONE / PAIR / JUNCTION / BOTH
```

Сравнить с canonical geometry-only relation head.

Model/threshold selection — train/validation only.

## Этап 4. Frozen Graph-LIRA

Первый end-to-end graph comparison сделать без test retuning.

Сначала оставить:

- graph compatibility/optimizer — frozen;
- relation confidence `tau=0.85`;
- perturbation consistency `0.60`.

Сравнить:

- geometry-only Graph-LIRA;
- CT-conditioned Graph-LIRA.

Основные structural metrics:

- repair-needed exact;
- false structural repair;
- incomplete / abstain;
- coverage;
- false among accepted;
- exact among accepted.

## Этап 5. Patient/anatomy robustness

Обязательно:

- per-patient metrics;
- patient-cluster bootstrap;
- paired bootstrap;
- LAD / LCX / OM / IM / D1 / D2 / R-PDA / R-PLA;
- high-degree junctions;
- все false structural repair cases.

## Этап 6. Только после этого — encoder ablation

Не начинать с большой нейросети.

Честная абляция при одинаковом graph/split/policy:

```text
radial hand-crafted CT
vs
compact conventional CNN
vs
compact ANZA encoder
```

Если ANZA не выигрывает, основная работа всё равно остаётся про image-conditioned selective graph repair.

## Этап 7. Richer 3-D / sequence context

Только если после предыдущего этапа остаётся конкретный failure mode:

- candidate-aligned 3-D tube;
- full cross-sections;
- CNN/ANZA tokens;
- затем Transformer/Mamba при необходимости.

Старый sequence experiment на сильно сжатых statistics не является основанием сразу брать Transformer.

## Неподвижные правила

1. Не тюнить held-out test.
2. Не менять frozen split ради результата.
3. Не использовать branch labels как inference input.
4. Не смешивать одновременно новый encoder, graph algorithm и threshold policy.
5. Не называть CT28 PAIR result полным Graph-LIRA result.
6. Не скрывать false structural repair за общей accuracy.
7. Не повторять old six-case CT add/veto pilot как финальную архитектуру.
