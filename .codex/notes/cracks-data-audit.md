# CRACKS data audit

Read `results/cracks_study/archive_inventory.json`,
`expert_color_audit.json`, and `split_feasibility.json` before changing the
loader or split. The official paper and repository define orange as certain
no-fault, green as uncertain fault, and blue as certain fault. White is present
in the actual palette but is not one of the three documented semantic classes;
keep it ignored/unassigned.

The old expert-only 200/25/40 stop was resolved by the approved ANZA-LIRA v2
protocol revision. Do not reuse that split as an active gate.

The verified official archives contain 396 images and 12,603 masks from 35
annotator directories, but only 40 expert masks. Their section IDs end at 300.
The old declared intervals contain 33 train, 3 validation, and 0 test expert
pairs. The historical audit remains evidence, but the active protocol uses
crowd-to-expert Setting A, frozen 5-fold expert Setting B, and image-disjoint
Setting C. See `docs/research/anza_v2_master_spec.md`.

The frozen 2026-08-18 T1 audit confirmed that white occupies 74.79% of the
12,480 non-expert annotation files and must remain ignored, not negative. A
per-annotator partial-label protocol strongly improved held-out explicit crowd
metrics for both U-Net and legacy ANZA, but also expanded full-image foreground
by roughly 24-25 percentage points and reduced precision. Treat this as a
crowd-supervision result with unresolved unknown-region overprediction, not as
expert-quality evidence or an ANZA advantage. Evidence is under
`results/final_practical_cycle/cracks_t1/`.
