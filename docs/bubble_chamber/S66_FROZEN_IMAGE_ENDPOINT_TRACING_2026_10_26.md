# S66 preregistration — image-derived ridge endpoint tracing (2026-10-26)

**Frozen before any S66 CAL or TEST.** S64 and S65 TEST metrics have already been inspected and are *prior exploratory motivation*, not used to tune S66. S66 employs **new** source-frame-disjoint CAL and TEST scenes and seeds and retains the original unmodified S39 and S56 baselines. No human annotations or H50Z native particle GT.

## Question

Can conservative pixel-only forward tracing recover missing endpoint extent of short ridge skeletons without using simulator coordinates and without increasing false joins? Compare the full image-to-graph pipeline and the one-to-one synthetic proxy endpoint matching, rather than only count masks or union-of-polylines.

## Frozen candidates (all fed only image pixels)

Start with the identical S39 `extract_ridges` polylines. At each end, trace outwards along the endpoint tangent, permitted a discrete [-12,0,+12] degree deviation per 1 px step. A candidate point is accepted only when a local multiscale ridge response passes a predeclared percent-rank budget AND local darkness exceeds a predeclared floor. On the first failed step STOP (no jumping synthetic gaps); near boundaries STOP. Never look at source geometry, source ID or generator truth.

- `s39_original`: no extension.
- `ridge_extend_strict`: high original ridge percentile budget 3.5%, darkness >=.012, max 8 px per endpoint.
- `ridge_extend_relaxed`: wider budget 7.5%, darkness >=.008, max 10 px per endpoint.

No change to S56 fit, CAL-only threshold <=5% negative pairs, S59 strict pair policy, MILP global stability margin .04 or posthoc synthetic evaluator.

## Frozen CAL/TEST

79/30/36 whole-frame visualMetro (film 070) split, same TRAIN 64 / original S56 CAL 64 synthetic scenes on SOURCE photo as S64/S65, source is 005 or 006. Separate source-photo CAL: seeds `20266601` and `20266611`, 32 scenes per direction; choose one variant per source WITHOUT looking at S66 TEST. Eligibility requires at least **+2 synthetic-correct selected links OR +5 uniquely matched injected true fragments**, no increase in *CAL synthetic-noncorrect selected joins* and no increase in *uniquely matched synthetic decoys*. Tie-break: more synthetic-correct joins, more matched true fragments, fewer wrong joins, then strict before relaxed. Otherwise retain original S39.

New TEST for source 005->006: seeds `20266603,20266604`, source 006->005: `20266613,20266614`, 24 scenes each = 96 independent synthetic TEST scenes. These are not S64/S65 scene seeds. Record both baseline and CAL-selected variant even when choice is baseline, including source fragment match, wrong endpoint count, synthetic-noncorrect proposals, whole artificial three-piece chains, scene-level paired bootstrap, solver failures, and abstentions. No threshold tuning after opening TEST.

## Safety / privacy

These are artificially injected curves on two genuine H50Z photo backgrounds of the **same physical run** and geometry from *different* film 070. The inferred identity on real H50Z remains UNKNOWN; no physical automatic join is authorized. Save image pixels, predicted point arrays and per-scene coordinates only in PRIVATE outputs, never in public Git. Public outputs may contain aggregate metrics and code only.
