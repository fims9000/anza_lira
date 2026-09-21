# Repository branch strategy

Date: 2026-09-21

## Current branches

### `main`

Default repository branch.

Purpose: stable shared codebase.

Do not use it as a scratchpad for exploratory research.

### `release/anza-lira-final-main`

Old release snapshot.

At the time of branch audit it is 7 commits behind `main` and has no unique commits ahead of it.

Treat as historical/release provenance, not an active development line.

### `research/anza-lira-q1-journal`

Small journal-writing/research branch, one commit ahead of `main` at audit time.

Keep separate unless its content is explicitly incorporated into the coronary connectivity-repair paper.

### `research/ccta-graph-lira-safe-repair`

Historical predecessor of the current CCTA/Graph-LIRA research line.

It contains the large technical research history and was 179 commits ahead of `main` at audit time.

No new work should be added here.

### `research/varvara-ccta-graph-lira`

Temporary collaborator-handoff branch created from the full CCTA research state and then extended with six human-readable handoff files.

It was 186 commits ahead of `main` before the canonical neutral branch was created.

Keep it as a snapshot so no handoff material is lost, but do not continue development there.

### `research/coronary-connectivity-repair`

**Canonical active research branch.**

Created from `research/varvara-ccta-graph-lira`, so it starts with all of:

- the full `research/ccta-graph-lira-safe-repair` history;
- all 28-patient matched-CCTA results;
- geometry baselines;
- patient-cluster bootstrap results;
- risk/coverage outputs;
- anatomy subgroup results;
- raw-data provenance/checksums;
- executable/reproducibility scripts;
- related-work notes;
- article direction;
- collaborator roadmap and repository map.

All future work for this project should go here.

## Rule going forward

Unless there is a deliberate release or publication-specific reason:

```text
main
  └── research/coronary-connectivity-repair   <-- active scientific work
```

Publication-specific branches can later branch from the canonical research branch, e.g.:

```text
research/coronary-connectivity-repair
  ├── paper/<venue-name>
  └── experiment/<specific-ablation>
```

Do not create person-named branches for the research line. People can receive handoff files inside the canonical branch without becoming part of the branch identity.

## Loss-prevention policy

The predecessor branches are intentionally not deleted during this cleanup.

This avoids losing:

- old commit history;
- exploratory negative results;
- handoff material;
- exact intermediate checkpoints.

Once the canonical branch has been used successfully for some time, predecessor branches may be archived/deleted only after an explicit review.
