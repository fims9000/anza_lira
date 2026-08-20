---
name: experiment-verification
description: Run deterministic phase gates and final validation for the GeoCrack study.
---

# Experiment Verification

Use `scripts/check_current_phase.py` for compact dataset, smoke, training,
traces, and final gates. Validate actual file contents, hashes, finiteness,
GeoJSON parseability, and consistency rather than filenames alone. A missing
artifact is incomplete, never implicitly successful. Final COMPLETE requires all
nine runs and a task state marked complete.

For ANZA-LIRA v2 on CRACKS, verify all frozen synthetic artifacts, all 15
Setting A crowd runs and their full non-expert threshold receipt, the guarded
expert evaluation, all 20 Setting B fold runs, all 15 Setting C fold runs,
human/disagreement outputs, section bootstrap, figures, evidence, and the
scientific audit. A negative quality gate may be a complete result, but a
missing run is never complete.
