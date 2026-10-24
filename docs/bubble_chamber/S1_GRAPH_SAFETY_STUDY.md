# S1: Synthetic-only graph-safe PAIR/NONE relation selection

2026-10-24. Active branch `research/bubble-chamber-graph-lira`.

## Goal

Test whether explicit graph constraints and abstention reduce false joins **after** a candidate-level classifier, without changing source candidate generation or synthetic training data. Inspired by the CT28 Graph-LIRA safe-repair architecture, but this is deliberately narrower: only `PAIR` versus `NONE`, not the real CT28 `NONE/PAIR/JUNCTION/BOTH` architecture. No CT28 pretrained weights or decision thresholds are reused.

## Protocol

- Synthetic oracle 2D trace fragments from `bubble_chamber/relation_study.py`, **not** segments generated from real JINR bubble photographs.
- Seed **20266024** is separate from the earlier S0 development and test seeds. Grouped train 160 scenes; validation 70; held-out test 70. Candidate generator is identical across both comparison arms.
- Image-conditioned logistic local pair model uses endpoint geometry, local quadratic discrepancy and observed photo darkness. Ground-truth fragment identity is excluded from model inputs.
- Candidate probability threshold is selected on validation to satisfy `FPR <= 0.02` with maximum candidate positive recall. No use of test labels for selection.
- Graph gating compares the SAME trained model and SAME candidate scores: mutual-best endpoint, minimum probability margin against competitors at each endpoint, max one link per endpoint and acyclic joining. The margin `[0,.05,.10,.15,.25]` is selected on validation only, among choices meeting candidate FPR<=2%; selected margin **0.10**.
- Graph policy produces `abstain` or `PAIR` on a synthetic relation, but does not draw pixels or export physical measurements. **Production real-image links remain human review only.**

## Held-out synthetic test results

| | Local pair decision | Graph-constrained pair decision |
|---|---:|---:|
| True accepted / candidate positives | 40 / 173 | 38 / 173 |
| False accepted / candidate negatives | 1 / 120 | **0 / 120** |
| Candidate positive recall | 23.1214% | 21.9653% |
| Candidate FPR | 0.8333% | **0% observed** |
| Accepted proposals | 41 | 38 |
| Scenes containing >=1 false bridge | 1 / 70 | **0 / 70** |

Compared with the exact same S1 local scores, graph constraints eliminated one spurious accepted edge, rejecting two correct links as well. Among 120 negative pairs, an observed 0 false accepts leaves a one-sided 95%-confidence binomial upper limit of about 2.47% (assuming independent identically distributed candidates, which themselves share scenes, so this is at best illustrative). It is insufficient to claim <2% real-world risk. The data are synthetic and non-independent at candidate level; final uncertainty must be clustered by acquisition/run.

This is **not** a physically validated particle-track connection method. The S1 result must never be combined directly with S0 numbers as if from the same held-out cohort. S1's purpose is an intra-protocol paired comparison of local versus global decisions.

## Reproduction

```bash
python scripts/research/bubble_chamber/run_graph_safety_study.py --out results/bubble_chamber/s1_graph_safety.json
python -m pytest -q tests/test_bubble_chamber_graph_safety.py
```

Full numeric protocol/result is retained in `docs/bubble_chamber/s1_graph_safety_metrics.json`. The deterministic test verifies no image ID truth is given to the model and global selection never auto-merges any real fragment.

## Next gate

Physical prototype requires labeled hydrogen/propane BMP triplets and matched `visualMetro` measurements plus optical calibration. Confirm that a false connection is rejected using appropriate multi-view evidence; only then consider real held-out Graph-LIRA comparisons. A deterministic architecture is not automatically an improvement in physics.
