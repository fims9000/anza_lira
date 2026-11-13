# V24 post-V23 exploratory candidate-retrieval repair (13 Nov 2026)

## Trigger and scientific status

**Designed after inspecting V23 expert evaluation**: on 40 CRACKS F3 expert sections, HGB-based fully automatic image-only prototype V23 connected 29/44 annotated evaluation anchor pairs from the raw segmentation; context repair still 29/44, while original geometry repair achieved 30/44. Six annotated paths had both anchors detected but disconnected, yet only **1/6 had a proposed direct candidate**, among 12,810 total candidate pairs. Full results from run 43383156467, artifact 14017283271. Thus the candidate stage, not just the scorer, is currently limiting.

V24 is **post-hoc exploratory**. This same already-inspected dataset cannot be used for confirmatory superiority claims or risk guarantees. The additional candidates and parameters below are fixed before the first V24 run.

## Frozen V24 alternatives on the identical real-image model source

Segmentation/training (identical to V23): crowd BLUE≥3 vs ORANGE≥3 on training sections 340,342,…,392, ignoring uncertain GREEN and unlabeled WHITE; crowd calibration 311–327 odd sections; no expert masks in training, threshold unchanged.

Compare these methods on identical predicted probability maps and same expert masks evaluated **only after** inference:

1. `raw`: no accepted link.
2. `context`: V23 terminal-only candidate search within 36 px, 70-degree directed angle gate, up to 8 per terminal, frozen score `-geometry+1.3*soft_support`, accept if score≥−0.3.
3. `geometry`: original V23 terminal-only geometry heuristic.
4. `terminal_extended`: proposed same visible terminal ports, but search ≤68 px, 85-degree directed angle gate, 16 per terminal; reject candidates whose ports already lie in one connected component of the **predicted** segmentation.
5. `virtual_landing`: `terminal_extended` plus virtual landing ports sampled evenly (max 12 per predicted trace interior, at least 8 pixels from trace ends), and at most 8 terminal-to-interior candidates per source. Source heading and target axial alignment each must pass the same 85-degree gate. Virtual candidates pay geometry penalty 0.15. Reject candidates between already connected predicted components. No source from expert ground truth.

Both new policies use exactly the original frozen V23 context score, accept gate and max 120 accepted links; no tuning to V24 expert outcomes.

Evaluation: explicit expert BLUE recall within 2px, false predicted pixels on explicit ORANGE, connected annotation-only anchor pairs / eligible paths, candidate upper bound from expert mask used **only as diagnostic** (not to generate or select links), per-section CSV and cluster bootstrap. Do not invent absent expert labels. The dataset has **no physical fault-instance LINK/REJECT/AMBIGUOUS labels**, and source masks come from one correlated F3 volume.

## Success criterion / claim

In this one-volume exploratory follow-up, a useful engineering gain requires both (a) greater annotated-anchor connectivity than V23 context and (b) a tolerable increase in explicit-ORANGE errors. Since no empirical natural false-link FPR is measurable, do **not** declare reliable geological false-bridge safety or statistically confirmatory superiority regardless of result. An increase in candidate count alone is not success.

Original pretrained neural ANZA/LIRA checkpoints and a second volume with independently judged physical links are still required for the original fully validated claim.
