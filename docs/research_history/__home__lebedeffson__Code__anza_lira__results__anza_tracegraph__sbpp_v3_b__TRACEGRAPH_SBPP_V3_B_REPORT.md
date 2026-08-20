# TRACEGRAPH SBPP V3-B

Status: `SBPP_V3_B_BRANCH_COVERAGE_PASS`

Candidate-only soft support was evaluated without training or changing the frozen hard graph.

| Variant | Recall@12 | Median | P95 | Wrong-near rate | Eligible |
|---|---:|---:|---:|---:|---:|
| hard_reference | 0.966518 | 1.0 | 5.0 | 0.029018 | reference |
| soft_0.30 | 0.968006 | 1.0 | 5.0 | 0.029018 | False |
| soft_0.25 | 0.969122 | 1.0 | 5.0 | 0.028274 | False |
| soft_0.20 | 0.970238 | 1.0 | 5.0 | 0.027158 | True |

Selected tau_s: `0.2`

## Development

- successes: `2611/2688`
- BranchCandidateRecall@12: `0.971354`
- Wilson 95%: `0.964344..0.977019`
- median / p95 candidates: `1.0 / 5.0`
- miss taxonomy: `{"B0": 0, "B1": 2, "B2": 0, "B3": 0, "B4": 21, "B5": 54, "B6": 0}`

| Stratum | N | Success | Recall@12 | Wilson 95% |
|---|---:|---:|---:|---:|
| straight_gap | 192 | 191 | 0.994792 | 0.971093..0.999080 |
| curved_gap | 192 | 192 | 1.000000 | 0.980385..1.000000 |
| s_curve_gap | 192 | 192 | 1.000000 | 0.980385..1.000000 |
| long_gap | 192 | 192 | 1.000000 | 0.980385..1.000000 |
| x_crossing_correct | 192 | 192 | 1.000000 | 0.980385..1.000000 |
| acute_crossing_correct | 192 | 192 | 1.000000 | 0.980385..1.000000 |
| t_junction_continue | 192 | 192 | 1.000000 | 0.980385..1.000000 |
| y_junction_continue | 192 | 192 | 1.000000 | 0.980385..1.000000 |
| weak_branch_continue | 192 | 117 | 0.609375 | 0.538865..0.675594 |
| close_parallel_continue | 192 | 191 | 0.994792 | 0.971093..0.999080 |
| low_contrast_continue | 192 | 192 | 1.000000 | 0.980385..1.000000 |
| partial_occlusion_continue | 192 | 192 | 1.000000 | 0.980385..1.000000 |
| multiple_plausible_correct | 192 | 192 | 1.000000 | 0.980385..1.000000 |
| cluttered_corridor_continue | 192 | 192 | 1.000000 | 0.980385..1.000000 |

`weak_branch_continue` remains a localized failure (not one of the predeclared V3-B main-stratum gates); no weak-branch success claim is permitted.

## Boundary

No P0/P1/P2, Transformer, ANZA, path, confirm metrics, CRACKS, expert data, optimizer, or training was opened.
