# ANZA-TraceGraph TG0--TG2 frozen result

The canonical frozen-prediction run stopped in TG1 with research status
`STOP_TRACEGRAPH_CANDIDATE_BOTTLENECK`. The frozen source was ANZA-KIR
`R0_static_residual` at its existing calibration threshold 0.35; its checkpoint
SHA-256 is `95ed21bfdf3fbddf693c3158ac5d83626134af76cdd65f7ec1a5de2b988272f6`.

On 2,048 development scenes (1,024 positive, 1,024 `NONE`, 512 X/parallel hard
scenes), source-tracelet availability was 1.000 but CandidateRecall was only
0.798828 versus the frozen 0.90 gate. The weakest positive strata were
X-crossing 0.578125, acute crossing 0.671875, and long gap 0.687500. Curved was
the only stratum above 0.90 at 0.937500.

Therefore canonical P0/P1/P2 training was never opened. TG3, confirm, CRACKS,
expert, P1G, and seeds 42/43 remain locked. The validator passed protocol,
source, checkpoint and split provenance, sample sizes, candidate-gate failure,
training lock, and all downstream access locks.

A separate earlier diagnostic under
`results/anza_tracegraph/tg2_visible_diagnostic` used generator-visible
tracelets. It trained P0/P1/P2 and itself found no meaningful P1 gain, but it is
not canonical TG2 because it did not use predicted tracelets. Preserve it only
as a relation-isolation diagnostic; it cannot override the TG1 stop or support
a real-pipeline claim.

