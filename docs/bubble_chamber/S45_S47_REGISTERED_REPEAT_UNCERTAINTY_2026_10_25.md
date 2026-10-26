# S45-S47: Machine-only registered repeat uncertainty and calibration sensitivity

Date: 2026-10-25. Active branch: `research/bubble-chamber-graph-lira`.
Status: implemented, executed on the original private H50Z scans, tests passed.

**Scientific boundary:** Native particle-identity ground truth is absent. These
experiments do not establish native particle detection, native scratch removal,
correct 2D particle links, physical mass, or calibrated 3D reconstruction.
The user does not need to label images for this machine-only study.

## Source provenance and constraints

- Inputs: original H50Z `005` and `006` uncompressed 8-bit BMP scans;
  supplied fiducial labels; S39 machine candidates and graph; S40 known
  fiducial guards; S42 **source-only Hessian geometry pseudo masks**; and
  the S44 registered same-coordinate ridge evidence. All image bytes, actual
  pixel coordinates, machine masks and registration matrix remain private.
- Exactly two H50Z photos from one physical run, not independent runs.
  The film-070 measured visualMetro points are a separate source of geometry,
  not dense labels for these H50Z images. S46/S47 synthetic curves were
  added programmatically to these genuine photo backgrounds.
- Six common fiducials register the two native scans with mean 1.507 px and
  maximum 2.707 px residual at those six points. This does not guarantee
  the same alignment error throughout the full film.
- A line at the same registered coordinates in two photos may be a scanner
  mark, fiducial, recurring background, chance overlap or a physical signal.
  Repeat support **must never be called proof of a scratch**.

## S45: native registered-repeat uncertainty, NOT a destructive filter

`bubble_chamber/s45_registered_unknown.py` adds a conservative, fixed
post-S44 machine heuristic, requiring (1) both photo ridge supports >=0.70;
(2) the minimum support exceeds the within-crop displaced-line control by
>=0.20; and (3) the line is not already known to be near a fiducial.
The thresholds are exploratory based on S44, *not* independently physics
calibrated. Existing S39/S40 machine nodes remain unchanged.

Only source/counterpart ridge intersection **within selected original machine
line corridors** can cause `0` or `1` *machine pseudo labels* to become `255`
(unknown). Labels cannot be promoted from unknown to a definite class.
The source baseline is S42 photo-only Hessian pseudo labels, not neural
predictions trained on the opposing H50Z photo.

Crucial: S45 views BOTH H50Z photos. Its output is a **transductive image
uncertainty/advisory layer**, NOT a legitimate training mask for a
leave-one-photo-out model. Using it to train such a model and reporting
heldout-photo accuracy would leak the evaluation photo into supervision.

Audited results on the 39 original crops:

| Metric | Result |
| --- | ---: |
| Native candidate centerlines preserved | 567 |
| Registered repeat-risk candidates (after separate fiducial guard) | 36 |
| Source-only S42 provisional positive pixels | 64,331 |
| Provisional positives demoted to `unknown` | 4,112 |
| Provisional negatives demoted to `unknown` | 0 |
| Selected paired-ridge uncertainty-region pixels | 14,190 |
| Existing geometric graph hypotheses preserved | 182 |
| S45 registered-repeat risky graph hypotheses | 18 |
| Overlap of S45 repeat and S40 fiducial-risk edges | 7 |
| New S45 risky edges not already marked in S40 | **11** |
| Union of S40/S45 flagged edges | **54 / 182** |
| Unflagged but still physically unverified edges | 128 |
| Native physical particle links automatically performed | **0** |

Exploratory crop bootstrap 95% interval for positive downgrade fraction:
[0.04128, 0.14895]. The 39 crops are correlated and from only two photos
of one run. This is not between-run uncertainty.

## S46: same-geometry synthetic counterfactual controls

Fixed seed 20261046. Each of 39 source crops receives an artificial dark
centerline, drawn identically in all conditions. Controls use the other
photo warped to the same coordinate grid:

1. Inject only into source photo: registered repeat-risk in **0/39** cases.
2. Inject at same positions into source and registered other photo:
   registered repeat-risk in **37/39** cases.
3. Inject into both, but offset 8 px **along local surface normal** in the
   registered counterpart: registered repeat-risk in **0/39** cases.

**Corrected protocol note:** an initial discarded version shifted the
second line by 8 px globally in X, which is tangent to near-horizontal
curves and not a valid displacement control. We fixed the synthetic
experiment generator to displace along each curve's *local normal* and
reran all scenes. Only corrected V2 figures are valid.

The high separation demonstrates a machine ability to detect coordinate
repeat of *injected pixel strokes*, **not** real-track-vs-scratch identity.
The S45 heuristic was not adjusted based on these results.

## S47: post-hoc registration sensitivity (same scenes as S46)

In fixed-seed exploratory normal displacement sweeps, the number of repeat
flags among 39 injected strokes was:

| Normal offset | Repeat-risk trigger |
| ---: | ---: |
| 0 px | 37/39 |
| 2 px | 37/39 |
| 4 px | 35/39 |
| 6 px | 0/39 |
| 8 px | 0/39 |

This S47 analysis **reuses S46 source crops and synthetic geometry**. It is a
post-hoc robustness analysis, NOT an independent validation set. In
particular, the residual measured at six fiducials is not a guaranteed upper
bound on registration error elsewhere. The observed abrupt change at 6 px
makes a failure mode explicit: uncertainty warnings may disappear under
larger cross-frame misregistration even when a true repeat is present.

## Reproduction using PRIVATE local inputs

```bash
python scripts/research/bubble_chamber/s45_native_repeat_unknown.py \
  --bmp-directory /PRIVATE/H50Z_originals_PRIVATE \
  --s39-private /PRIVATE/S39_MACHINE_HYPOTHESES_PRIVATE.json \
  --s44-private /PRIVATE/S44_PRIVATE_CROSSFRAME_EVIDENCE.json \
  --s39-graph /PRIVATE/S39_GRAPH_RELATIONS_PRIVATE.json \
  --s40-graph /PRIVATE/S40_GRAPH_FIDUCIAL_GUARDS_PRIVATE.json \
  --s42-pseudos /PRIVATE/PRIVATE_geometry_only_teacher_masks \
  --out /PRIVATE/S45
python scripts/research/bubble_chamber/s46_registered_repeat_counterfactuals.py \
  --bmp-directory /PRIVATE/H50Z_originals_PRIVATE \
  --s39-private /PRIVATE/S39_MACHINE_HYPOTHESES_PRIVATE.json \
  --s40-private /PRIVATE/S40_FIDUCIAL_CANDIDATES_PRIVATE.json \
  --out /PRIVATE/S46
python scripts/research/bubble_chamber/s47_registration_sensitivity.py \
  --bmp-directory /PRIVATE/H50Z_originals_PRIVATE \
  --s39-private /PRIVATE/S39_MACHINE_HYPOTHESES_PRIVATE.json \
  --s40-private /PRIVATE/S40_FIDUCIAL_CANDIDATES_PRIVATE.json \
  --out /PRIVATE/S47
python -m pytest -q tests
```

Results: `S45_PUBLIC_AGGREGATES.json`,
`S46_PUBLIC_SYNTHETIC_COUNTERFACTUALS.json` and
`S47_PUBLIC_REGISTRATION_SENSITIVITY.json`. Full private per-candidate
responses, native image views and machine masks are kept in private archives.

## Next unresolved scientific issue

- There is NO verified native dense pixel GT or track-to-track identity.
  Machine labels can support **engineering demonstration** but not clinical/
  physical recognition-accuracy claims.
- The cross-photo repeat cue is correlated with native 2D image structures,
  but no evidence yet distinguishes permanent scanner damage from genuine
  recurring physical patterns.
- To improve physical interpretation without user-provided labels, work on
  machine-only 2D/3D consistency, optical calibration, image-domain
  invariances and hard negatives; **do not reclassify uncertain samples as
  confirmed GT**. Expert review may occur later after the automated demo.
- The original source architectures `AZConv` / `AZUNet` are not modified.

Safety: Never commit the raw H50Z BMPs, visualMetro track coordinates,
private S39/S40/S44 JSONs, masks, per-photo registration matrices, model
weights, thumbnails, or review HTML containing native pixels to public Git.
