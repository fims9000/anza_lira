# CT28 JUNCTION work branch

This branch is the working branch for the current JUNCTION+CT experiment.

Start with:

1. `docs/varvara/JUNCTION_GENERATOR_REVIEW_2026-10-03.md`
2. `scripts/research/ccta_graph_lira_safe_repair/build_junction_plan.py`

The uploaded draft generator is committed unchanged as the starting point.

Do not generate/freeze held-out test candidates yet.

Immediate development target:

- revise train/val generator using the review;
- add geometry-adversarial hard-negative mining;
- audit adaptive-cut cases;
- re-run geometry-only sanity checks;
- freeze the train/val protocol;
- only then implement JUNCTION CT features.

Canonical parent research branch remains:

`research/coronary-connectivity-repair`

When the JUNCTION protocol is frozen and verified, merge the resulting implementation/results back into the canonical research branch.
