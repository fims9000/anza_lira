# S56 frozen protocol — machine-only 2-D path assembly

**Frozen before evaluating S56 heldout data, 26 Oct 2026. Branch `research/bubble-chamber-graph-lira` only.**

## Falsifiable research question

Does a graph-level, endpoint-capacitated, cycle-free optimizer recover more *complete artificial three-piece paths* than a node-level mutual-best heuristic, **without increasing wrong joins**? Its 2-D forest prior is only a model of this open-path synthetic benchmark, **not** a general claim about possible charged-particle trajectories. Neither variant is the CT28 learned Graph-LIRA relation head.

## Data and frozen split

- The 145 audited film-070 visualMetro frames supply 2-D measured coordinate *shapes* only. Use pre-existing S37 whole-frame 79/30/36 train/cal/test split and its SHA256; never use measurements as H50Z native-photo labels.
- Authentic H50Z 005 and 006 512px crops are *unannotated backdrops* only. Train and calibrate on one photograph, test on the other, also reverse. They originate from a single physical run; the split cannot prove real-world generalization.
- Generate 2 artificial 3-piece open trajectories (31 points per piece) plus 1 hard distractor (7 nodes, exactly 4 artificial true adjacent joins per scene). Geometry derives from source-frame disjoint visualMetro templates. Simulated endpoint extraction noise, crossing/parallel/decoy/empty-gap variants, randomized fragment orientations. Draw synthetic strokes onto genuine photo backdrops for private QA; the native background provides NO physics GT.
- Fixed generator sizes: 64 train, 64 calibration, 64 test scenes per test seed; two independent test seeds per direction, 20265603/04 and 20265613/14. Train 20265601/11; calibration 20265602/12. If one direction has too few templates, fail closed. No redefining these settings after test inspection.

## Frozen scoring and selection arms

Same **six S35 geometry features**, z-score StandardScaler + balanced logistic regression `C=0.3`, `random_state=1337`, trained only on synthetic TRAIN. Single per-pair candidate cutoff is estimated using the frozen S35 `calibrated_negative_threshold` with maximum **5% empirical negative CAL FPR**. This threshold is not a native safety guarantee.

Compare, at identical trained scores and cutoff:

1. `node_mutual`: baseline one continuation maximum per fragment; score margin 0.04 against *all* edges of either node. Cannot assemble >2 pieces by design.
2. `port_greedy`: allow one edge at each geometric endpoint, no cycles; select descending score with per-port rival margin >=0.04.
3. `global_forest`: maximum-weight selected endpoint matching, subject to no cycles (iterative exact MILP cycle cuts). Edge utility `score - cutoff` (strictly positive); for every selected edge require the optimal objective to drop by >=0.04 when that edge is forbidden, otherwise abstain. Failed/timeout solvers return zero suggestions and a recorded failure, never an invented success.

No learned pixel masks or AZUNet predictions in the S56 relation-score target. All native S39 candidates/graph nodes preserved; new native output is advisory and never authorizes physical track joining.

## Metrics and controls

For each held-out scene report correct artificial joins, wrong joins, scenes with at least one wrong join, true-link recall, exact complete-path recovery, graph component count, suggestion count, abstention, and synthetic pair AUROC. Compare arms on the **same scenes**. Report 95% paired scene-bootstrap intervals for changes in correct and wrong joins, conditional on the simulator only, not independent physical runs.

Software controls: reversal invariance under input polyline order; port capacity, acyclicity, fail-closed malformed scores, no use of test labels in model fit/threshold/selection, entire-scene confidence aggregation, preservation of native source coordinates outside public outputs. Any negative or zero improvements must be reported.

## Gate

Do not claim physical particle recovery, identity, stereoscopic 3D, momentum, mass, PID, validated false-join rate on real events, or certified photo-independent generalization. No automatic physical joins and no private media or per-photo coordinates in public GitHub.
