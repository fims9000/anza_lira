# TraceGraph SBPP V3-B soft-support pass

V3-B fixed the causal defect in V3-A by applying `tau_s` before skeletonization
inside a source-directed sector. The hard graph and hard source port remained
frozen. Soft components used the predeclared H1 hard-anchor or H2 coherent
self-support rules and were clustered with hard branches without truth.

On the independent repair calibration, the hard-only reference was 2598/2688
= 0.966518. Soft thresholds 0.30, 0.25, and 0.20 yielded 0.968006, 0.969122,
and 2608/2688 = 0.970238 respectively. Only 0.20 passed every frozen gate and
was selected. Candidate burden remained median 1 and p95 5; wrong-near-endpoint
cases decreased from 78 to 73; B6 was zero.

The previously untouched V2 development was then opened once. V3-B achieved
2611/2688 = 0.971354 BranchCandidateRecall@12 with Wilson 95% interval
0.964344--0.977019, median 1 and p95 5 candidates, B6 zero, and every
predeclared main stratum at or above 0.90. Status is
`SBPP_V3_B_BRANCH_COVERAGE_PASS`.

The important residual is `weak_branch_continue`: 117/192 = 0.609375. It was
not a predeclared main-stratum V3-B gate and accounts for 75 of 77 development
misses. Do not claim weak-branch success. P0 is authorized only as a separate
phase; P0/P1/P2, Transformer, ANZA, path, confirm evaluation, CRACKS, expert,
and all training remained unopened in V3-B.
