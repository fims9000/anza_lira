# S85-INTEROP: fixed source-only detector/graph benchmark outcome

**28 Oct 2026 · `ANZA-LIRA-BC-S85-INTEROP-20261028`**. No genuine H50Z image or visualMetro measured curve was used for this new comparison. Synthetic generator and background were independently re-created using public frozen seeds; all metrics below are on **generated scenes only**. All source code and protocol digests are stored in `S85_INTEROP_PUBLIC_TOY_AGGREGATES_2026_10_28.json`.

## Global MILP results — 20 scenes per condition (40 possible 3-part paths)

| Candidate detector | Clean correct / other / complete | Blur 1.1 correct / other / complete | Scratch/grain correct / other / complete |
|---|---|---|---|
| Hessian ridge | **27 / 20 / 10** | **13 / 16 / 5** | **28 / 22 / 10** |
| LSD | 12 / 47 / 4 | 8 / 54 / 1 | 10 / 28 / 2 |
| Hough-P | 4 / 36 / 0 | 4 / 20 / 0 | 3 / 38 / 1 |
| Injected oracle lines — evaluator-only upper bound, not a deployable detector | 46 / 10 / 13 | 46 / 10 / 13 | 46 / 10 / 13 |

The generator deliberately produces two 3-part tracks and a hard artificial decoy per scene. The best detector here did **not** reach the oracle control; remaining error is not purely a graph problem. A straight-line detector can find many segments but still not assemble correct curved tracks. No measured H50Z physical precision or recall can be calculated from these numbers.

## Global MILP versus simpler port-greedy (same candidates & same uncalibrated scorer)

| Method | Clean complete paths (greedy → global) | Clean synthetic-correct links (greedy → global) | Clean other links (greedy → global) |
|---|---:|---:|---:|
| Hessian ridge | 10 → 10 | 27 → 27 | 20 → 20 |
| LSD | 3 → 4 | 7 → 12 | 21 → **47** |
| Hough-P | 0 → 0 | 1 → 4 | 13 → **36** |
| Oracle artificial fragments | 12 → 13 | 42 → 46 | 6 → 10 |

For LSD the extra **one** complete path comes with **26 additional noncorrect links**. That is not a safe improvement. Scene-bootstrap intervals from only 20 scenes are wide (`global − greedy` for LSD complete paths: mean `+0.05` per scene, 95% interval `[0, 0.15]`). The result is **diagnostic negative/inconclusive for the claim that MILP globally improves precision**. Global is not selected over greedy by this benchmark.

## Scientific decision

- **Technical reusability:** demonstrated. Three detector families feed the exact same strict S59 + S56 geometry graph via a stable x-y line API and per-detector score binding. Cross-detector reuse of previous model or CAL is intentionally blocked.
- **Scientific advantage for real H50Z:** NOT demonstrated. Original S84 full synthetic-on-real-background results remain the only relevant previous H50Z-background tests; they did not justify upgrading S72. The learned S85 model is still not trained/validated on private data in this session.
- **Recommended integration:** only as a read-only overlay / candidate review tool. Make `port_greedy`, `node_mutual` and `none` available as comparators; use `global_milp` only after a CAL selection and independent validation. A new learned segmentation baseline must have its own geometry scorer trained and CAL threshold frozen.
- **Research bottleneck:** detector fragments and the availability of reliable endpoint-localized pieces, not simply the choice of global solver. Obtain independent physical runs and small expert-confirmed review samples before physics claims.

**References, API contract and adoption checklist:** `DEVELOPER_INTEGRATION_2026_10_28.md`.
