# CRACKS Data Contract V2

Status: `PASS_WITH_RELEASE_AND_SPATIAL_LIMITATIONS`.

## Verified release

- `images.zip`: official expected MD5 matches.
- `Fault segmentations.zip`: official expected MD5 matches.
- Extracted seismic images: **396**.
- Nominal IDs absent from the checksum-verified image archive: **9, 185, 249,
  336**.
- Local extraction contains every image present in that verified archive. The
  difference from the paper's nominal 400 sections is therefore a release
  inventory fact, not a silent local extraction loss.
- Non-expert annotators: **34** (`26 novice`, `8 practitioner`).
- Non-expert annotation files: **12,563**.
- Image sections with at least one non-expert annotation: **393**.
- Images without non-expert annotations: **49, 73, 385**.

The machine inventory, per-file SHA-256 values, duplicate-hash groups, and
archive digests are stored in
`results/anza2/phase0/data_contract.json`.

## Label semantics

The frozen policy follows the published CRACKS colors:

- blue: certain fault, positive;
- green: uncertain fault, lower-confidence positive;
- orange: certain no-fault, explicit negative;
- white: unassigned/unknown and ignored as a direct label.

White is never silently converted to confident background.

## Expert lock

The Phase-0 audit checks only that the expert directory and filenames exist. It
does not hash or decode expert files, read pixels, compute scores, or use expert
sections for model selection. `expert_data_accessed=false`.

## Protocol G feasibility

No coordinate/adjacency metadata is present in the two verified archives.
Physical spatial coordinates and section orientation are therefore
`NOT_ESTABLISHED` and are not guessed from filenames.

A deterministic five-fold grouped OOF protocol is nevertheless feasible over
all 393 annotated image sections:

- each outer section is evaluated exactly once;
- train, DEV, outer, and numeric-ID exclusion buffers are section-disjoint;
- no annotator from an outer section enters that fold's training data;
- numeric IDs are grouped contiguously with a two-ID exclusion radius;
- this is reported as grouped section OOF, not proven physical-spatial OOF.

The exact folds are frozen in
`results/anza2/phase0/SPLIT_PROTOCOL_V2.json`. This Protocol G is distinct from
the optional transductive crowd-to-expert Protocol T.
