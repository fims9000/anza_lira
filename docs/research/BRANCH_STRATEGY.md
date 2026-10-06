# Repository branch strategy

Updated: 2026-10-06

## One active research line

The repository now has one stable branch and one canonical active research
branch:

```text
main
└── research/coronary-connectivity-repair
```

### `main`

Stable/default ANZA-LIRA repository state.

Do not use `main` as an exploratory research scratchpad.

Do not merge the current coronary/CCTA research line into `main` until the
end-to-end method/protocol is deliberately promoted as a stable repository
state.

### `research/coronary-connectivity-repair`

**Single canonical active branch** for the coronary connectivity-repair /
CCTA Graph-LIRA line.

On 2026-10-06 the completed train/validation CT28 JUNCTION work from
`experiment/ct28-junction` was fast-forwarded back into this branch.

All current research state now lives here.

## Historical/checkpoint branches

The following branches are preserved only to avoid losing provenance or
breaking existing local checkouts. No new commits should be added to them.

### `release/anza-lira-final-main`

Old release snapshot. Historical only.

### `research/anza-lira-q1-journal`

Old one-commit journal checkpoint. Its commit is already an ancestor of the
current coronary research history.

### `research/ccta-graph-lira-safe-repair`

Historical predecessor containing the large CCTA/Graph-LIRA development
history.

### `research/varvara-ccta-graph-lira`

Historical collaborator handoff snapshot.

### `experiment/junction-ct28`

Redundant old pointer that matched the pre-JUNCTION canonical branch.
Do not use.

### `experiment/ct28-junction`

Short-lived JUNCTION development branch. Its work was promoted into
`research/coronary-connectivity-repair` on 2026-10-06.
Do not continue new work there.

## Rule for future experiment branches

Prefer working directly on the canonical research branch for small,
well-understood steps.

Create `experiment/<specific-purpose>` only when isolation is genuinely
useful. Such a branch must:

1. branch from `research/coronary-connectivity-repair`;
2. have one narrow purpose;
3. contain no independent roadmap/handoff hierarchy;
4. be merged/promoted back promptly after the experiment is accepted;
5. never become a second long-lived source of truth.

Do not create person-named active branches.

## Documentation authority

Current instructions:

1. `docs/varvara/CURRENT_TASK.md`
2. `RESEARCH_START_HERE.md`
3. this file

Dated handoffs and dated reviews are historical provenance unless explicitly
marked current.

## Loss-prevention policy

Historical branches are intentionally not deleted during this normalization.
This prevents accidental loss and avoids breaking collaborators with local
checkouts.

Deletion can be done later only after confirming that no collaborator still
depends on those remote branch names.
