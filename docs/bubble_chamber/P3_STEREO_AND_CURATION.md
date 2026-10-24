# P3: Strict stereo input manifests and expert-controlled 2D track curation

Date: 2026-10-24. Branch: `research/bubble-chamber-graph-lira` only.

## Main result

This milestone connects two previously separated pilot paths:

1. For future **real three-view** datasets, a verifiable manifest records three original BMPs per event, film/run provenance, calibration file references, file hashes and a run-disjoint split. A three-photo dataset has **not** been acquired yet. The schema cannot infer which frames belong together: a physicist provides explicit event IDs and view paths in the CSV.
2. Human review results (`same_track` / `different_track` / `uncertain`) now flow into a 2D curation module. Only a validated explicit `same_track` label can join two fragments. A proposed gap remains explicitly **unobserved**; it is not synthesized as new pixels. Contradictory or structurally impossible continuity claims are rejected.

## 3-view CSV contract

Header in `docs/bubble_chamber/TRIPLET_INPUT_TEMPLATE.csv`:

```csv
event_id,chamber,run_id,film_id,frame_id,view1,view2,view3,optical_file,fiducial_file,magnetic_field_file
```

- Provide **real** identifiers and three different BMP files for each event. Files must be under one explicitly supplied local root. Do not mix views from distinct events; a filename alone cannot establish this.
- Optical, fiducial and field-map files are optional only for the data inventory. If missing, status is `INCOMPLETE`. If all exist, status is `FILES_PRESENT_UNVERIFIED`: hash checks are **not** numerical validation of calibration.
- Exact SHA256 of images and calibration references is pinned, and verified before any 2D image processing. Duplicate source images and path traversal outside the root are rejected.
- If at least three independent chamber+run groups exist, a fixed-seed split assigns **whole runs** to train/validation/test. With fewer groups no fake test split is created (`INSUFFICIENT_RUN_GROUPS_FOR_HOLDOUT`). The grouping scheme is preliminary; a final domain-transfer protocol should also consider chamber type, date and film development.
- **Important:** even with valid manifests, this release does not perform cross-view track matching or 3D triangulation. Real camera optics include refractive glass/media effects, so a generic pinhole stereo model should not be substituted without validation.

## Scripts

```bash
python scripts/research/bubble_chamber/prepare_stereo_dataset.py \
  --input-csv /local/acquisitions.csv \
  --image-root /local/bubble_films \
  --out results/bubble_chamber/stereo_inventory.json

python scripts/research/bubble_chamber/analyze_triplet.py \
  --manifest results/bubble_chamber/stereo_inventory.json \
  --event-id <real_event_id> \
  --segmentation frangi --ridge-fraction 0.16 \
  --out-dir results/bubble_chamber/triplet_candidate_views

python scripts/research/bubble_chamber/curate_reviewed_tracks.py \
  --result results/bubble_chamber/action_p2/result.json \
  --labels /local/exported_human_labels.json \
  --out results/bubble_chamber/reviewed_tracks.json
```

In the three-view example above, `analyze_triplet.py` only runs **three independent P2 2D detectors**. It writes per-view results and a summary marked `NO_STEREO_MATCHING_NO_3D_NO_MOMENTUM`. Do not describe it as physical reconstruction.

## Curation contract

- The reviewer export is bound to its grayscale image SHA, exact pipeline version and canonical result JSON SHA. A change in any of these invalidates the labels.
- Each observed fragment can have a maximum of two continuity neighbors, one per endpoint. A single endpoint cannot attach twice; cycles, contradictions to `different_track` and ambiguous graph topologies require separate expert treatment.
- Output contains ordered original fragment IDs, original or reversed orientation, 14 observed sample points **per source fragment**, explicitly listed accepted but unobserved gaps, and a diagnostic quadratic fit across observed samples. No pixels are silently interpolated into the film.
- Candidate line connection is based only on human-reviewed decisions. `uncertain` and unreviewed links remain disconnected.
- **No physical start/stop, charge, momentum, dE/dx, neutral particle, or particle species is inferred.**

## Verification and real-photo probe

- `python -m pytest -q tests/test_bubble_chamber_*.py` — **43 passed locally** including photo checksum mutation, path traversal, run splits, reviewer digest matching, rejected contradictory human labels and ordering of reversed links.
- With one existing unlabeled archive JPEG and P2 Frangi, `37` fragment groups were exported with **zero accepted joins** from an intentionally empty label bundle. This is a workflow smoke test, **not** a physically verified segmentation or automatic reconstruction.
- A test created three artificial distinct BMP views, verified checksums, ran P2 on each and asserted that the output explicitly disclaims 3D. This synthetic triplet does not constitute real experimental geometry.
- No external images were downloaded due to DNS restrictions of the execution environment. No image from the user's archive is committed publicly.

## Next hard research gate

Obtain a properly paired three-BMP dataset, with optical and field calibration and human-measured tracks from visualMetro. Freeze runs before experimenting. Then implement a physically correct cross-view correspondence adapter based on the existing solver's documented optical transformations, plus track-identity labels and a falsifiable false-bridge evaluation. **Do not guess** a visualMetro track-file schema; inspect real per-track file samples first.
