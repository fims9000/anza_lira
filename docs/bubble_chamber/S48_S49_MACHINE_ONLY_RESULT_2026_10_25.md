# S48-S49: Independent fiducial jackknife and autonomous machine-only demonstration

Date: 2026-10-25. Only `research/bubble-chamber-graph-lira` was modified.
This study uses the original private H50Z `005` and `006` BMP scans.
No expert/native particle ground truth is available; there are zero confirmed
physical joins. No manual work from the user is required for this stage.

## Why this matters

S44 used six common fiducials to align the two H50Z scans. Its original
1.507 px mean / 2.707 px max residual was measured **on the markers used
for fitting**. S47 showed that repeat cues can suddenly disappear under
larger normal displacements. S48 measures prediction error more honestly by
leaving each fiducial out of the fit, and measures how this affects the 567
native machine proposals. S49 packages the results for physicists.

## Frozen S48 protocol and safeguards

See `S48_FROZEN_REGISTRATION_JACKKNIFE_PROTOCOL_2026_10_25.md`, frozen
before accessing the original S48 comparison results. No change to the
registered S45 threshold (both-image ridge support >=0.70; advantage over
within-image displaced path >=0.20). The alternative 2D similarity family
is identical to S44. Six fits omit one marker each. At least four alternative
fits must support an existing repeat for its warning to be classed as stable.

All original candidates and graph proposals are retained. Marking a new
possible repeat NEVER means a demonstrated scratch; lack of a repeat NEVER
means the line is a verified particle track. No 3D calibration, physical
track joining, mass/charge inference or new dense labels are created.

LOO fit results (pixel error at heldout marker):

| Statistic | Original in-sample fit | Six-marker leave-one-out |
| --- | ---: | ---: |
| Mean error | 1.5073 px | **2.4467 px** |
| Maximum error | 2.7072 px | **4.6024 px** |

LOO displacements at the 567 machine centerlines, relative to S44 registration:
median 2.0042 px; 90th percentile 3.3048 px; maximum 3.6885 px.
These are **jackknife sensitivity figures, not calibrated confidence
intervals**. Six marks are far too few to certify worst-case registration
over the entire film or between independent physical runs.

## S48: all original native machine candidates

| Machine output | Count |
| --- | ---: |
| Original S39 lines preserved | **567** |
| S45 flagged repeats still stable in 4+ alternate registrations | **36** |
| Newly possible registered-repeat warnings | **6** |
| Separate known fiducial-proximity candidates | **56** |
| Not strongly repeated by this test | 467 |
| Fewer than four usable variant positions | 2 |
| Original graph relation proposals retained | **182** |
| S40 known fiducial-risk edges | 43 |
| S45 registered-repeat edges | 18 |
| S48 *additional* new graph edges with pose-only warning | **4** |
| Union of all three warning types | **58** |
| Graph edges without these warnings (still unverified) | **124** |
| Native physical joins automatically executed | **0** |

**Important:** This is exploratory transductive inference. S48 reads both
H50Z scans, so the resulting private per-candidate labels and image-derived
warnings **cannot** be used as a pseudo-teacher for a leave-one-photo-out
model and then reported as heldout-photo performance.

## S48 synthetic controls (NEW seed 20261048)

156 image-grounded synthetic controls on 39 real H50Z backgrounds, one
artificial line and four counterfactuals each. This is separate injected
geometry from S46-S47 but the same underlying two scans and one run.

| Injected condition | Original S45 alarm | S48 any-of-six alarms | S48 at least 4/6 alarms |
| --- | ---: | ---: | ---: |
| Line in source photo only | 0/39 | 0/39 | 0/39 |
| Same line at exactly registered coordinates in both | 37/39 | 37/39 | 37/39 |
| Line in both with 4 px *normal* offset | 36/39 | 37/39 | 22/39 |
| Line in both with 8 px *normal* offset | 0/39 | 0/39 | 0/39 |

The 4 px condition exposes **an inherent ambiguity**: a wrongly registered
single repeated structure and two physically distinct nearby lines can
produce exactly the same two pixel arrays. With no independent optical or
identity labels, an image-repeat test cannot resolve those latent causes.
In particular, 37/39 successful detection of repeated strokes is not
particle/scratch classification accuracy.

## S49: private science demo

- One self-contained offline HTML interface; 78 JPEG cropped image assets:
  two photographic views of each of the 39 original machine-sampled areas.
- Toggle line candidates, graph proposals, only uncertain lines, only
  risk-flagged graph edges. Filter by source photo `005`/`006`; navigate
  through every original crop. Export all machine hypotheses to a JSON file.
- Source pixel SHA-256 matched 39/39 previous S39 records before the
  display JPG copies were saved. JPG thumbnails are intentionally lossy;
  the algorithms analyzed the original 8-bit BMP pixels.
- No reliance on external CDN, JS bundles, fonts or network fetches.
- Chromium checked the complete viewer with image bytes inserted as data
  URIs for browser QA. The machine browser's policy blocks direct `file://`
  navigation; **direct file URL was not tested in this runtime**, although
  the packaged viewer uses relative image URLs and inline JSON, compatible
  with a normal local browser.
- Private gallery, per-candidate source geometry, JSON, JPEGs, registration
  and raw BMPs must NEVER be committed to public GitHub.

## Reproduction

Public code includes:

- `bubble_chamber/s48_registration_jackknife.py`
- `scripts/research/bubble_chamber/s48_jackknife_registered_risk.py`
- `scripts/research/bubble_chamber/s49_build_machine_showcase.py`
- `scripts/research/bubble_chamber/s49_offline_template.html`
- `tests/test_s48_registration_jackknife.py`
- `tests/test_s49_build_machine_showcase.py`
- `docs/bubble_chamber/S48_PUBLIC_AGGREGATE_RESULTS.json`
- `docs/bubble_chamber/S49_PUBLIC_SHOWCASE_SUMMARY.json`

S48 invocation with PRIVATE data:

```bash
python scripts/research/bubble_chamber/s48_jackknife_registered_risk.py \
  --bmp-directory PRIVATE/H50Z_originals_PRIVATE \
  --s39-private PRIVATE/S39_MACHINE_HYPOTHESES_PRIVATE.json \
  --s44-private PRIVATE/S44_PRIVATE_CROSSFRAME_EVIDENCE.json \
  --s40-private PRIVATE/S40_FIDUCIAL_CANDIDATES_PRIVATE.json \
  --s39-graph PRIVATE/S39_GRAPH_RELATIONS_PRIVATE.json \
  --s40-graph PRIVATE/S40_GRAPH_FIDUCIAL_GUARDS_PRIVATE.json \
  --private-out PRIVATE/S48 --public-out docs/bubble_chamber
python -m pytest -q tests
```

S49 uses S48 private results, the native BMPs, S39 private candidate/graph
records, and the public aggregates. Private output can be opened by
`S49_MACHINE_ONLY_DEMO/index.html` after extraction in a normal web browser.

## Next scientific experiment

The current pipeline is ready to show real machine suggestions to physicists.
To move beyond image-repeat ambiguity WITHOUT asking the user to draw labels,
focus on obtaining an independent third view and optics/field calibration
for the *same event*, or process independent runs with machine-only review.
In the meantime, native image results remain hypotheses: do not assume that
all long dark lines are tracks or that all repeated structures are defects.
The source AZUNet and Baseline U-Net architectures are unchanged.
