---
name: crossing-trace-bench
description: Build and validate deterministic seismic-like synthetic cases with exact branch identity and routing truth.
---

# CrossingTraceBench

Use `docs/research/anza_v2_master_spec.md` sections 25-46 and 89-90.
Generate samples on demand; do not store a large PNG corpus. Train, validation,
and test use disjoint frozen RNG streams. A sample must preserve overlapping
per-instance and per-branch masks because a single 2D integer map cannot encode
both identities at a crossing.

The image must contain layered seismic-like texture with controlled fault throw,
not foreground lines rendered on an empty background. Exact branch-pairing and
instance claims are allowed only on this benchmark, never inferred from CRACKS
semantic masks. Freeze synthetic test configuration before opening test metrics.

Keep observed segmentation and latent completion mathematically separate:
`visible_fault_mask` is the segmentation target, `latent_fault_mask` is full
instance geometry, and `gap_mask = latent_fault_mask & ~visible_fault_mask`.
Canonical identity truth is overlapping `instance_masks[N,H,W]`, never a scalar
ID raster. Pairing truth comes only from generator lineage. Include matched
negative gaps and a `nontrivial_pairing` stratum where minimum turning angle is
not the true continuation. X, T, and Y keep distinct topology contracts.

Do not implement or run structural metrics until these invariants pass their
dedicated tests. Do not report synthetic smoke numbers as scientific results.
