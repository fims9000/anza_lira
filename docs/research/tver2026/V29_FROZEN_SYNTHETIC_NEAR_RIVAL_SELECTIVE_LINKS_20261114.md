# Tver V29 — frozen synthetic challenge: true alternatives, false bridges and a matched-risk decision gate

**Registered 14 November 2026 before looking at V29 synthetic results.** V10–V16 and V28 previous results have already been inspected, so the research question and method family are informed by historical experiments. This is an **additional independently sampled synthetic generator**, not an external test of geological fault instances or original pretrained ANZA–LIRA. Do not call it a full ANZA/LIRA system outcome: the input is an *idealized, procedural soft segmentation probability map*, not a learned AZConv prediction from a seismic detector. It isolates **the actual repository V23/V25 LIRA-style automatically proposed continuation/NO_LINK policy**.

## Why this experiment

V28 repeated AZConv2d outperformed ordinary Conv2d by 9–11 linked expert anchor pairs per seed on the same CRACKS F3 cube, at markedly higher ORANGE pixel errors; HGB remained a stronger accuracy-risk competitor. On prior synthetic V10/V12 near-line alternatives were almost never accepted correctly (5/508 in V10; 11/510 frozen V12; 35/510 new V12, at higher false acceptance). No unrestricted general-model-superiority statement is warranted. V29 measures the **risk–recovery trade-off** with explicit source-instance identities and automatic, non-oracle ports in a genuinely fresh synthetic scene family.

## Frozen data-generating scheme, only procedural, no oracle inference

- Scene grid **112×168** pixels, one to three visible thin curved line instances with 1–2 px stroke; smooth low background (<0.22) plus spatially correlated nuisance, bright visible strokes (>0.80); true missing segments have soft confidence below fixed threshold **0.60**, occasionally faint residual support below that threshold. Every scene has the same pixel budget and source distribution; no ground-truth label, latent line ID, artificial hidden gap mask, or endpoint is passed to inference.
- Four conditions for each nuisance background group, each with a known true physical line-instance assignment:
  1. `easy_positive`: a single curved source trace with a hidden central segment, connecting two fragments is correct.
  2. `near_rival_positive`: same true gap plus a close roughly parallel visible distractor fragment; a bridge to the distractor is a *wrong branch*.
  3. `indistinguishable_negative`: **identical observable soft probability map as `easy_positive`**, but two independently sourced fragments have NO physical connection. This makes the latent labels fundamentally non-identifiable from that 2-D input. Its one-to-one positive counterfactual is deliberately balanced; no algorithm can be correct on both. Verify bitwise identical arrays.
  4. `distinct_negative`: two unrelated, noncollinear fragments close enough to tempt a purely nearest-endpoint bridge but with visibly incompatible directions; no physical link is valid.
- All four conditions share nuisance parameters and RNG group, but distinct deterministic positive/negative truth labels; max gap length 36 px, several gap bands 8–14/15–24/25–36. For close competitors, offset 6–12 px. Split groups with disjoint deterministic seeds **CAL: 75 groups ×4 = 300 scenes**, **TEST: 150 groups ×4 = 600 scenes**. No TEST tuning. Group is the bootstrap unit (not pixels or model predictions), paired counterfactuals counted together.
- Original visible fragments, latent connectivity, intended source/destination and distractor support are retained in the generator's *private evaluator object only*; all methods see solely the same soft probability array.

## Exact fixed actual repository methods vs fair simple controls

Each scene evaluated with unchanged, existing repo components, fixed threshold `0.60`:
- `V23_nearest` = existing `scripts.cracks_real_auto_core_v23.run_pipeline(strategy='nearest')`;
- `V23_geometry` = same with `strategy='geometry'`;
- `V23_context` = same with `strategy='context'`;
- `V25_region_hybrid` = existing `scripts.cracks_real_region_v25.run_regions(mode='region_hybrid')`;
- `morph_close_radius4` = classical binary morphology closing with radius4, **not an ANZA–LIRA method**;
- `NO_LINK` = never connect anything, risk-zero negative control.
All are run on the identical generated, not neural-predicted probability map. The original score gates and candidate/graph rules are kept exactly as code; do NOT retrain AZConv/HGB on this generator. For scored link policies, optionally apply ONE additional method-specific reject threshold to the **already-selected scored edges**. That threshold is determined on CAL only, maximizing CAL positive correct-and-no-wrong rate under at most 2% empirical false-bridge fraction over CAL **distinct_negative** scenes only (75 physically identifiable negatives; the 75 observationally identical counterfactual negatives are reported separately and cannot be safely classified from this input); rejection may only remove previously selected paths, never add links or rerank on TEST. This calibrated reject-only variant is a *new policy selection step*, not an unchanged historical LIRA policy; report both original and calibrated.
- Candidate-generation counts, accepted links and NO_LINK rates are reported for each algorithm. Negative false bridge is any automatically inserted connector between independently generated fragments. Positive correct means the original two genuine anchors become 8-connected after reconstruction **AND no wrong distractor gets attached**. Wrong branch is explicitly reported separately. Morphology has no learned score; report fixed radius4 only (no fair calibrated comparison implied).

## Metrics and evidence gates

Primary: (i) correctly restored positive easy and near-rival links per condition, (ii) false bridges on **distinguishable** `distinct_negative` as the CAL risk gate and on `indistinguishable_negative` as a **separate impossibility diagnostic**, (iii) wrong distractor connections on positives, (iv) abstentions, (v) candidate availability and (vi) **risk–coverage Pareto comparison** under the CAL 2% false-bridge target. Report TEST risk even if >2%: the CAL constraint does **NOT** guarantee TEST or physical risk control. Bootstrap descriptive 95% paired confidence intervals using original 150 TEST nuisance groups; report uncertainty for the TEST difference vs geometry, not pixel-bootstrap.

Two mathematical sanity checks: 
- Exact matching observable positive/negative soft arrays must produce the same deterministic link actions, proving no 2D-only method can infer their distinct hidden identities. The paired identical counterfactual contributes equally to true/false rates.
- At the observationally identical paired instances, no method can simultaneously restore the positive and avoid false link in the negative on the same input; this is an identifiability negative control, not a property specific to LIRA.

## Scientific interpretation constraints

If V25 improves correct near-rival reconnection compared with geometry at matched risk, one may state an improvement for **this idealized procedure and synthetic distribution only**. If no improvement, retain the negative result. Neither result licenses `ANZA–LIRA is superior to state-of-the-art segmentation`; there is no matched Frangi, U-Net, clDice-trained network or independently heldout geological survey comparison in V29. Existing CVPR 2021 clDice and classical multiscale vesselness are relevant methods in the literature, but they are not experimentally measured comparators here.

Do not claim the synthetic generator simulates true F3 geological fault physics. Do not promote the oracle segmentation probability input to full end-to-end inference. Keep V28 main paper and accepted IOI/PIERE work unchanged; only add a carefully scoped addendum if empirical evidence is credible.

## Explicit pre-outcome analytic separation
The deliberately **observation-identical** `easy_positive` / `indistinguishable_negative` counterfactual pair is **never used to calibrate the attainable low-FPR policy or claim an identifiable-class benchmark**. It is reported as a separate proof-by-construction: with exactly the same observed soft segmentation, any deterministic method necessarily makes the same link decision for both latent worlds. Risk calibration at the 2% empirical budget therefore uses only `distinct_negative`, whose visible fragment orientations differ. The 75 CAL and 150 TEST identical pairs remain evaluated and their unavoidable trade-off is reported, including negative false links if the method connects. This separation was locked before first V29 execution.

## Exact locked reject-threshold selection implementation
For each of the four scored V23/V25 methods, gather scores of **already accepted** links from CAL scenes and form 21 empirical score quantiles at 0%,5%,…,100% plus finite sentinels −10^6 (leave all originally accepted links) and +10^6 (reject all links). On CAL `distinct_negative` only, allow at most `floor(0.02×75)=1` scene with a false bridge. Among acceptable thresholds maximize positive correct scenes across the 150 CAL `easy_positive` and `near_rival_positive` scenes, then minimize wrong competitor links, then minimize distinct-negative false links, then prefer the STRICTER (larger) threshold. On TEST apply the single chosen threshold unchanged. No path reranking, topology modification, new candidate proposals or ground-truth endpoint use is allowed. Scored policies share the original implicit V23/V25 accept gate; this extra threshold only removes accepted paths. The morphological/no-link controls remain fixed. This implementation-specific amendment was made **before any V29 CAL or TEST run**.
