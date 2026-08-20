---
name: structural-metrics
description: Evaluate visible segmentation and latent branch identity on CrossingTraceBench without conflating the two tasks.
---

# Structural metrics

Use only after the `crossing-trace-bench` target contract passes. Observed
segmentation metrics use `visible_fault_mask`; latent completion metrics use
`latent_fault_mask` and must be explicitly prefixed. Never emit an unqualified
Dice, IoU, or clDice for this benchmark.

Compute positive gap recovery together with matched-negative false bridge rate.
A negative bridge requires both sufficient gap occupancy and connectivity
between its two endpoint neighborhoods. Compute false merge and false split
against canonical overlapping latent `instance_masks[N,H,W]`. Ground-truth
continuation relations come from generator lineage only, never angle heuristics.
Score X pairings separately while respecting the distinct T and Y relations.

Define all empty cases explicitly and reject NaN or Inf. Freeze thresholds
before opening the synthetic test stream. Synthetic statistics use generated
samples as independent units, never pixels.
