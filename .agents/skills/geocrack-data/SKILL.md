---
name: geocrack-data
description: Download, pair, split, normalize, load, and audit GeoCrack without source-image leakage.
---

# GeoCrack Data

Use `docs/research/geocrack_master_spec.md` sections 5–10. Reuse existing
segmentation tensor conventions. Download only official patched data through the
Dataverse API, verify sizes/checksums/pairs, and make reruns idempotent. Derive
`source_image_id` from patch names and group split with seed 2026. Compute
normalization from train only. Test leakage before implementation and freeze the
test CSV hash before any model evaluation.
