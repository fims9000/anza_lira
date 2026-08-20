# TraceGraph P0 Endgame V1 frozen STOP

Fresh relation train/calibration/development streams were generated from the
frozen TRACEGRAPH_RELATION_V2 geometry with disjoint seeds and frozen SBPP V3-B
at `tau_s=0.20`, `K=12`. The exact historical
`path_completion.pair_classifier.EndpointPairClassifier` was imported rather
than reimplemented and trained for the fixed 20 epochs on seed 41.

Calibration found a feasible safety threshold `0.9991924167` with RR 0.469866,
FB 0.019965, and WB 0.001153. On fresh development, ranking remained strong
(AUROC 0.978948, top1 0.978503), but CCR 0.461420 and all-positive RR 0.447173
failed the frozen 0.87/0.84 gates. FB 0.016493, WB 0.002687, and NONE recall
0.983507 passed. The validator independently recomputed all denominators and
reported PASS with research status `STOP_P0_RELATION_SELECTOR`.

Post-gate diagnostic attribution is `NONE_SCORE_SEPARATION`, not candidate
competition: at CCR >=0.87 the minimum observed source-level FB is 0.155382.
At the frozen threshold, 18 of 19 accepted NONE sources are
`independent_collinear_fault`. This diagnostic does not change the threshold or
reopen the gate.

E4 path, confirm, CRACKS, expert, Transformer, candidate repair, ANZA changes,
and seeds 42/43 remained unopened. Any further work needs a separate
predeclared protocol; do not repair P0 in this frozen cycle.
