# V42b: bounded-compute followup to timed-out V42 port allocation audit

Original V42 GitHub Action run [43619324978](https://github.com/fims9000/anza_lira/actions/runs/43619324978) was **cancelled on timeout**, with no results artifact, after the full 4×640-scene source-port candidate recomputation. No scientific improvement or negative result may be inferred from that run.

V42b keeps the original frozen ANZA V33 source, exact `.65` threshold, 48-pair budget and the same **four previously registered** allocations 4×12, 6×8, 8×6 and 12×4. To make the falsification affordable, it intentionally narrows the output to **the original 120 LINK gaps in the previously viewed Thebe REL_CAL (60 source sections)** and reuses the already generated original 4×12 candidates rather than needlessly recomputing them. It only recomputes 6×8/8×6/12×4 on the same 120 scenes (360 heavy SBPP calls instead of 2560), with no TRAIN or synthetic NO_LINK variants. No TEST7, no model tuning, no new labels as input.

## Registered evaluation under the reduced scope
- Original `_candidate_scenes` must reproduce exactly 65/120 true candidate availability and 1/120 geometric top1. Every candidate rule remains unchanged except source-port count and per-port budget.
- For all 120 originally annotated LINK gaps, report original/additional true candidate availability, rescues and losses, correct top1, candidate count max 48, available source ports before cap, and dense-mask refusals. Values are correct only if original candidates exactly match source-recomputed baseline in source/target identity **by construction from the original scene**.
- This CAL has been repeatedly inspected. Reduced V42b is diagnostic and cannot be used as a confirmatory improvement; omitting NO_LINK explicitly precludes false-link risk evaluation. If any variant rescues positives, it needs a separate full end-to-end risk study before use; if not, stop this port-budget hypothesis.
- Preserve cancellation and source SHA in provenance. No TEST7 opened.
