---
name: human-disagreement
description: Compare CRACKS models and annotators with the available expert subset and analyze model uncertainty against crowd disagreement by seismic section.
---

# Human disagreement

Use only the 40 released expert sections and annotations that share the same
section ID. Evaluate novice and practitioner masks with the same policy and
metric implementation used for models. Report agreement with the available
expert annotation; never claim that a model is better than humans.

Aggregate the primary uncertainty analysis by seismic section. Pixels may
contribute to a section summary but are not independent statistical samples.
Compare human entropy with routing entropy, `1-rho`, model error, junction
score, and anisotropy only where the corresponding model-native field exists.
Use Spearman correlation and section-cluster bootstrap. Preserve negative and
undefined results explicitly; do not replace unavailable internal fields with
invented proxies under the same name.

Expert data may be read only after the Setting A threshold freeze receipt.
Human-comparison results must retain annotator role, section ID, mask policy,
and source artifact provenance.
