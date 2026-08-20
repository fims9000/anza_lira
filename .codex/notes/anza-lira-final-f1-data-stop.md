# ANZA-LIRA Final Endgame F1 real-gap data STOP

The final non-Anosov LIRA protocol froze the complete research ledger and a
section-disjoint CRACKS layout before natural-gap counts were read. Frozen T1
U-Net seeds 41/42/43 were averaged without retraining. Dense calibration on
held-out nonexpert annotators selected `tau_h=0.30` under the predeclared
precision-at-least-0.75 rule.

CRACKS annotations are raster semantic masks, not polyline or instance files.
The audit therefore accepted only ordered non-junction, non-loop,
non-border-truncated skeleton segments as local trace identities. This is
adequate for local same-segment gaps and different-local-trace negatives, but
does not identify geological faults through crossings or across disconnected
components.

The frozen natural-gap definition found only 3 positive gaps in
`lira_calibration` and 1 in `lira_development`, versus 150 required and an
absolute STOP floor of 75. Across all 334 already-opened non-confirm sections
there were 76 gaps total, which cannot populate independent
calibration/development/confirm cohorts and does not authorize changing the
frozen split after counts were observed. Confirm inference/counts and expert
annotations remained unopened.

Final status for this line is `STOP_LIRA_REAL_GAP_DATA_INSUFFICIENT`. F2 real
SBPP, F3 P0 fine-tuning, path, relation seeds 42/43, confirm, and expert were
not run. Do not weaken the gap definition, reuse the pooled diagnostic as a
test set, or create another ANZA/LIRA rescue architecture under this protocol.

