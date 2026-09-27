# Varvara collaborator artifacts

Canonical active pack:

`artifacts/varvara/ct28_pair/`

Use that directory for the frozen CT28 PAIR baseline.

## Current status

The earlier generated-artifact gap is closed. The canonical pack contains:

- exact 1,360-row frozen pair plan;
- row-level predictions;
- train/val/test + per-patient summaries;
- protocol and SHA provenance;
- CT alignment and expected geometry;
- lossless compressed full feature table;
- regenerated PAIR model snapshots;
- scripts to restore/retrain the models without raw CCTA.

Run the pack integrity check from repository root:

```bash
python scripts/research/ccta_graph_lira_safe_repair/verify_varvara_collab_pack.py
```

Expected result begins with:

`PASS: Varvara CT28 collaborator pack is internally consistent.`

## Legacy top-level files

Files named `artifacts/varvara/ct28_*.csv` are compatibility mirrors created before the canonical `ct28_pair/` pack was finalized.

Do **not** use them as the primary handoff path.

They are retained only so old notes/commits do not break; new work must read from:

`artifacts/varvara/ct28_pair/`

## Raw CT

Raw ImageCAS CCTA is intentionally not stored in Git.

It is **not needed** to reproduce the already-frozen PAIR baseline.

It **is needed** for the new JUNCTION+CT image-feature extraction.

Exact cohort/source/integrity instructions:

`docs/varvara/CT28_DATA_ACCESS.md`

## Current task

`28-patient JUNCTION+CT -> CT-conditioned relation head -> frozen Graph-LIRA`

Start from:

`docs/varvara/FINAL_HANDOFF_2026-09-27.md`
