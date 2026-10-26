# S67 — instance-aware continuity at skeleton crossings: bounded negative result

**Experiment ID:** `ANZA-LIRA-BC-S67-20261026`. Frozen 26 October 2026; source and CAL+TEST generated and checked locally; only aggregated data retained in public Git.

## Why

S64–S66 showed the bottleneck is extraction of 2D polylines from image pixels, especially nearby line crossings and endpoint ambiguity. S67 proposed collinear continuation of *pixel-derived* skeleton branches across high-degree junctions, while refusing ambiguous alternatives. No simulator source curves or physical identities are provided to the inference function.

## Protocol

The frozen protocol is `S67_FROZEN_CROSSING_INSTANCE_PROTOCOL_2026_10_26.md`. It uses 145 visualMetro film-070 frames split by whole frame 79/30/36 (same film only), and 39 private crop backgrounds from two H50Z photos of one physical run. The image-to-graph code is tested on **synthetically inserted** seven-fragment scenes. S56 geometry fit, CAL cutoff and global selection policy are unchanged. S67 uses new CAL seeds 20266701/20266711, 16 scenes per direction; fresh TEST seeds 20266703/04/13/14, 24 scenes each, 96 scenes total. Synthetic truth is used **only for posthoc assessment**.

## New TEST results (96 synthetic scenes)

| Method | Correct synthetic joins | Other selected joins | Full three-piece synthetic paths | Scenes with noncorrect proposals |
|---|---:|---:|---:|---:|
| Original S39 | 8 | 78 | 2 | 45 |
| S67 crossing-aware | 8 | 72 | 2 | 41 |

**CAL decision:** Both directions selected `s39_original` before opening TEST. Under the frozen rule S67 did not demonstrate a CAL gain in correct artificial joins and is **not promoted**. Reduction in some TEST false proposals is reported as exploratory, not used to revise the frozen CAL decision. Neither method has expert native H50Z GT; no physics accuracy is claimed. All found machine connections remain `UNKNOWN` and reversible.

## Data integrity and reconstruction

- Two genuine H50Z BMPs belong to one run and do not match film 070; 3-view physical trajectory is not reconstructed.
- Private input photos/crops, raw 7z files, measured-point coordinates, per-scene images, model checkpoints, synthetic injected polylines, individual frame IDs and link identities are **not committed**.
- Public IDs are research experiment IDs and frozen code hashes, **not particle identifiers**.
- Follow-up: improve pixel-instance extraction/branch separation on new independent data; do not iterate on S67 held-out TEST for model selection.

See `S67_PUBLIC_SYNTHETIC_RESULTS_2026_10_26.json` for direction-specific CAL/TEST and paired scene bootstrap 95% intervals.
