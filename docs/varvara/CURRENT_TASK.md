# CURRENT TASK — CT-conditioned Graph-LIRA

Date: 2026-09-26

This is the current execution target for the coronary connectivity-repair line.

## What is already solved well enough to freeze

The 28-patient matched-CCTA pair-relation experiment is complete.

Official patient split:

- train: 17 patients;
- validation: 5 patients;
- held-out test: 6 patients.

The best local PAIR evidence is currently:

**geometry + radial 2.5-D CCTA**

Held-out test:

- AUROC: 0.9847
- recall: 82.63%
- FPR: 1.80%
- precision: 97.87%

This result is a local binary **PAIR relation** result. It is not yet a full Graph-LIRA structural-repair result.

## Important architectural gap

The current strong CT28 evidence exists for PAIR relations.

The corresponding 28-patient **JUNCTION + CT** evidence has not yet been built and validated.

Therefore the next work is not "put the pair score into the graph and tune until it works".

The next work is:

### Task A — JUNCTION + CT

On the same 28 matched CCTA patients and the same official split, construct candidate-aligned CCTA evidence for junction candidates.

Compare:

1. geometry junction evidence;
2. CT-only junction evidence;
3. geometry + CT junction evidence.

Keep patient separation fixed.

Do not use anatomical branch labels as model inputs.

Thresholds / model selection are validation-only.

Report per-patient results and patient-cluster uncertainty.

### Task B — CT-conditioned relation-type head

After both local evidence streams exist:

- PAIR geometry + CT;
- JUNCTION geometry + CT;

construct a scene-level relation representation and predict:

- NONE;
- PAIR;
- JUNCTION;
- BOTH.

The clean comparison is:

**canonical geometry relation head vs CT-conditioned relation head**

The canonical geometry relation head is an HGB scene-level classifier trained from out-of-fold geometry score distributions. It is not the same model as the CT28 binary geometry HGB baseline.

### Task C — frozen Graph-LIRA integration

Feed the relation decision into the existing global structural compatibility layer.

Initially keep the already validated selective policy frozen:

- relation confidence tau = 0.85;
- perturbation consistency = 0.60.

Do not tune these values on the held-out test set.

Primary structural outputs:

- repair-needed exact;
- false structural repair;
- incomplete / abstain;
- coverage;
- false among accepted;
- exact among accepted;
- patient-level results;
- patient-cluster bootstrap;
- LAD / LCX / high-degree-junction analysis.

## What not to do yet

Do not:

- start a large CNN / Transformer / Mamba first;
- tune geometry thresholds again;
- copy the old six-patient CT add-only rule as the final method;
- use the old PAIR_IMG / JUNC_IMG weights as canonical models;
- tune on test patients 954, 958, 972, 973, 980, 984;
- claim end-to-end CT-conditioned Graph-LIRA improvement before the structural experiment exists.

## After this task

If CT-conditioned Graph-LIRA improves repair-needed exact while preserving the low-false objective, run a clean local image-encoder ablation:

same patients + same candidates + same graph + same safety policy:

1. radial hand-crafted CT;
2. compact conventional CNN;
3. compact ANZA encoder.

Only then decide whether ANZA becomes part of the main contribution.

## Code status for the later Graph-LIRA integration

The immediate JUNCTION+CT experiment is **not blocked** by the missing old `run_graph_lira_large_scale.py`.

However, before Task C becomes publication-grade, do not pretend that the missing historical runner is a clean reusable module.

Current repository reality:

- old large-scale results/protocol are preserved;
- archived scripts contain earlier graph logic/pilots;
- exact old `run_graph_lira_large_scale.py` + `scenes_full.pkl` + model pickles are not canonical artifacts;
- the collaborator-facing PAIR code is now clean/reproducible;
- the new JUNCTION+CT code should be written cleanly;
- when PAIR_CT + JUNCTION_CT are ready, the full integration should be promoted into a **new canonical Graph-LIRA runner/module** with explicit inputs, saved protocol, and reproducible outputs rather than reviving hidden/local pickle state.

So the execution order remains:

1. build/validate JUNCTION+CT;
2. freeze its artifacts;
3. build the new CT-conditioned relation head;
4. promote the graph integration into canonical code;
5. run the frozen structural evaluation.

Do not spend the first JUNCTION iteration reconstructing the old missing runner.
