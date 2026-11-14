# V42 controlled Thebe source-port budget redistribution (posthoc)

Frozen before V42 execution 14 November 2026. Same original real Thebe V33 ANZA checkpoint SHA256 `ba0c3b8060fd0e3cb1c15f6c1598b94a977677c2b075663b5414207df382c25e` and 0.65 segmentation threshold. No new fit, no U-Net, no external TEST7 access.

## Falsifiable hypothesis
Original V34 real relation CAL has 18/120 gaps without candidates, 37/120 with candidates but no true continuation, and 65/120 with a true proposal. Among 102 scenes with candidates, the stored source_count is **4** in every case. This is the `MAX_SOURCES=4` cap, not evidence that exactly four input ports existed before cap. Source ports are selected by ascending distance to the expert-scheduled hidden gap rectangle center, then confidence; many unrelated nearby ports may crowd out a true source.

V42 asks whether reallocating an unchanged **48-pair proposal budget** across more source ports improves true candidate availability, rather than repeating segmentation threshold retuning.

## Registered bank allocation at the same 48-pair limit
- baseline 4 source ports × 12 landing candidates;
- alternative 6 source ports × 8 candidates;
- alternative 8 source ports × 6 candidates;
- alternative 12 source ports × 4 candidates.

All retain original source probability, grayscale, expert-defined intervention rectangle (only for gap masking and local scene) and original SBPP orientation/distance gates. Do NOT use hidden labeled expert endpoints to rank, choose, generate, or deduplicate candidate pairs. Labels enter only after all proposals. All candidates ranked by repository's existing negative geometric cost, not a new scorer. Do not alter threshold 0.65, data, model weights or train split.

Report on original TRAIN1 sections 0..99 and previously inspected VAL1 REL_CAL sections 40..99: number of eligible source ports **before** cap; all-method candidate recall/positive top1; rescues and losses against exact original candidate sets; mean candidate count and synthetic NO_LINK candidate exposure. **Fail closed** if exact original baseline differs or any variant exceeds 48 pairs; no hidden TEST7.

## Scope
The same Thebe REL_CAL was viewed in V34/V38. This is posthoc diagnostic, not fresh independent evaluation. The 48-pair cap and lack of natural geological fault instance IDs preclude safety/superiority claims. Negative outcomes must remain visible. A candidate recovery gain still requires a matched-label real ranking model, risk-coverage calibration and new independent field confirmation.