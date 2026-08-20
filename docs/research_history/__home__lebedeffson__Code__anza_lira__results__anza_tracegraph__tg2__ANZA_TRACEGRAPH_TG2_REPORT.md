# ANZA-TraceGraph TG1 Stop Report

Status: `STOP_TRACEGRAPH_CANDIDATE_BOTTLENECK`

The canonical frozen-prediction audit stopped before TG2 training. A relation model cannot recover a true continuation absent from its shared candidate set.

- frozen dense source: `ANZA-KIR R0_static_residual`
- dense checkpoint SHA-256: `95ed21bfdf3fbddf693c3158ac5d83626134af76cdd65f7ec1a5de2b988272f6`
- development scenes: `2048`
- positive / NONE: `1024 / 1024`
- source availability: `1.000000`
- CandidateRecall: `0.798828` (required `>=0.90`)

| Stratum | Positive sources | Candidate recall |
|---|---:|---:|
| straight | 64 | 0.875000 |
| curved | 64 | 0.937500 |
| s_curve | 64 | 0.828125 |
| x_crossing | 64 | 0.578125 |
| acute_crossing | 64 | 0.671875 |
| close_parallel | 64 | 0.781250 |
| parallel_gap_confuser | 64 | 0.828125 |
| weak_branch | 64 | 0.765625 |
| y_junction | 64 | 0.796875 |
| t_junction | 64 | 0.875000 |
| long_gap | 64 | 0.687500 |
| none | 64 | 0.890625 |
| multiple_plausible | 64 | 0.750000 |
| low_contrast | 64 | 0.843750 |
| cluttered_corridor | 64 | 0.828125 |
| partial_occlusion | 64 | 0.843750 |

## Required answers

1. Candidate generation was not sufficient; it failed the frozen 0.90 gate.
2. Scene-level relation modeling was not compared with P0 in the canonical run because TG2 training remained locked.
3. The effect of NONE was not estimated in a trained canonical relation model.
4. Incremental ANZA attention value was not tested in the canonical run.
5. P1G remained locked, so absorption by a generic learned bias was not tested.
6. Path geometry was not tested; TG3 remained locked.
7. The largest candidate-recall failures were X, acute-crossing, and long-gap scenes.
8. CRACKS is not legally allowed by this protocol.

## Controlled diagnostic boundary

A separate generator-visible relation-isolation diagnostic exists under `results/anza_tracegraph/tg2_visible_diagnostic`. It trained P0/P1/P2 but is not the canonical TG2 result because its tracelets were not produced by the frozen segmentation source. It must not reopen TG2 or support a real-pipeline claim.

## Locks

P0/P1/P2 canonical training, TG3, confirm, CRACKS, expert, P1G, and seeds 42/43 were not opened.
