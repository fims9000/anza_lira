# Bubble-chamber P1: offline expert review and public-source probe

Date: 2026-10-24. Branch: `research/bubble-chamber-graph-lira`.
Base: Varvara CT28 Graph-LIRA `research/coronary-connectivity-repair` at `80a7521` (2026-10-14). Original CT28 branch remains unchanged.

## What is implemented

1. `bubble_chamber/review.py`: pair-identity human review exporter. It binds labels to original grayscale pixels SHA256, canonical P0 result SHA256, exact fragment endpoints, and pipeline version. Wrong-file labels cannot be loaded or validated. No pair is automatically merged. Overlay is approximated by the existing 14 measurement samples, not full contours.
2. `scripts/research/bubble_chamber/make_review.py`: offline self-contained HTML interface, no server and no LLM. Operator explicitly marks `same_track`, `different_track`, or `uncertain`; saves and can reimport JSON. Validation mode checks exact IDs and provenance.
3. `scripts/research/bubble_chamber/download_public_samples.py`: optional small (two-image) publicly sourced liquid-hydrogen archive probe, with original URLs, rights, dimensions and SHA256. Images are unlabeled and are not an independent benchmark or JINR/propane data.

## Verification, limitations

- 19 focused Python tests passed locally: 12 original + 4 review + 3 public-probe loader tests.
- Generated HTML from the supplied archive's `action.jpg` and initial result: 17 geometry-only proposals, zero auto-merges.
- JS syntax check via Node passed; automated browser screenshot was not completed in this environment. Human UI interaction is not yet acceptance-tested.
- Two real public-domain photo source pages verified; automatic external download attempted but local environment cannot resolve public Internet hosts. No claim that these external photos were downloaded. The original single archive photo is available locally; original photo and derived crops are NOT committed to the public repository.
- No actual expert pair labels have been collected. No neural-network training, physical reconstruction, 3D stereo matching or particle species identification has happened.

## Reproduction commands

```bash
python -m pytest tests/test_bubble_chamber_pilot.py tests/test_bubble_chamber_review.py tests/test_bubble_chamber_public_samples.py -q
python scripts/research/bubble_chamber/run_pilot.py --image '/path/to/action.jpg' --out-dir results/bubble_chamber/action_p0
python scripts/research/bubble_chamber/make_review.py --result results/bubble_chamber/action_p0/result.json --image '/path/to/action.jpg' --out results/bubble_chamber/action_p0/review.html
python scripts/research/bubble_chamber/make_review.py --result results/bubble_chamber/action_p0/result.json --labels '/path/to/bubble_labels_*.json'
python scripts/research/bubble_chamber/download_public_samples.py --out-dir data/bubble_chamber/public_qualitative
```

## Research gate

Obtain legal/approved H2 and C3H8 3-view matched BMP samples; verify scanner calibration and source provenance. Label true/false fragment links including NONE/hard-negative candidates. Freeze splits by experiment/run/film before selection. Next use those annotations to compare strict geometry-only relation scoring vs local image-conditioned Graph-LIRA and record **false bridge risk**, coverage and correction time. The historical CT28 `tau=0.85` and consistency `0.60` have no calibrated meaning for bubble photographs. Do not transfer them.

## Primary public sources

- https://commons.wikimedia.org/wiki/File:Liquid_hydrogen_bubblechamber.jpg — US DOE public-domain photograph, 1955.
- https://commons.wikimedia.org/wiki/File:HD.6D.635_(12000265314).jpg — US DOE photograph, hydrogen event.
- https://cds.cern.ch/record/2307419 — CERN educational bubble-chamber collection, images with original views/highlights; separately check CERN rights.
