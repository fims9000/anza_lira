# Varvara collaborator artifacts

Canonical pack:

`artifacts/varvara/ct28_pair/`

There are no parallel top-level CT28 CSV mirrors anymore. All collaborator-facing CT28 artifacts are under this one directory.

## Current status

The earlier generated-artifact gap is closed.

The pack contains:

- exact 1,360-row frozen pair plan;
- row-level predictions;
- train/val/test + per-patient summaries;
- protocol and SHA provenance;
- CT alignment and expected geometry;
- headline result, paired bootstrap and risk/coverage;
- lossless compressed full feature table;
- regenerated PAIR model snapshots;
- scripts to restore/retrain models without raw CCTA.

Run the integrity check:

```bash
python scripts/research/ccta_graph_lira_safe_repair/verify_varvara_collab_pack.py
```

Expected:

`PASS: Varvara CT28 collaborator pack is internally consistent.`

## Raw CT

Raw ImageCAS CCTA is intentionally outside Git.

It is not needed to reproduce the frozen PAIR baseline.

It is required for new JUNCTION+CT image-feature extraction.

Exact source/cohort/alignment:

`docs/varvara/CT28_DATA_ACCESS.md`

## Current task

`28-patient JUNCTION+CT -> CT-conditioned relation head -> canonical frozen Graph-LIRA evaluation`

Start from:

`docs/varvara/FINAL_HANDOFF_2026-09-27.md`
