---
name: anza-v2-transport
description: Implement and verify mode-resolved axial and half-mode directional transport without altering AZConv2d v1.
---

# ANZA-LIRA v2 transport

Use `docs/research/anza_v2_master_spec.md` sections 4-24 and 85-87. Keep
`models/azconv.py:AZConv2d` unchanged as the v1 baseline. Implement v2 in a
separate module and preserve mode states across more than one transport block
before fusion.

Primary v2a uses symmetric center/neighbor anisotropic geometry and axial
orientation compatibility. V2b adds +/- half-mode direction and source-based
row-stochastic local transport. Use shared projections, vectorized local
operations, finite bounded diagnostics, and explicit tests for pi periodicity,
mass conservation, junction score, limits, gradients, and memory/runtime.

Do not use Anosov or ergodicity claims. The supported language is local
determinant-one paired expansion/contraction inspired by hyperbolic splitting.
