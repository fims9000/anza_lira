# Bubble chamber S109–S113: source-backed candidate annotation and full-frame QA

Scope: **only** bubble-chamber research. No modifications to crack, geology, generic ANZA or other research modules. The `bubble_chamber` package contains vector (S109), global association (S110–S111), bead-graph proposals (S112) and offline labeling/QA (S113).

**Scientific honesty.** All S109–S112 track groups are unverified geometric hypotheses, not demonstrated particle IDs. Their spline A/B are parameterization endpoints, **not** proven particle origin or physical final point. Photo-backed samples and inferred gaps remain distinct. Named f1…f8 marks and operator `start` points can only be obtained from correctly matched `labels.json`. X-template matches are unverified candidates. On our original H50Z 005/006, classical full-frame LSD and S112 photometric peak stages ran in tiled mode; full-frame global S111 grouping or trained AZUNet inference **did not** run. Report candidate counts with possible duplicates on overlapping tiles, never as recall or number of true physical tracks.

**Confidentiality.** Do **not** commit original BMP, `labels.json`, film geometry/coordinates, physical track point lists, checkpoints, source image overlays, generated S109–S113 ROI JSONs or private whole-frame reports to public GitHub. Such outputs stay local/private and may contain sensitive research information. Commit only code, synthetic tests and aggregate claims without coordinate lists.

## Setup

```bash
python -m pip install -r requirements_s113.txt
python -m pytest -q tests/test_s113_public_smoke.py
```

## Complete-image baseline on true 8-bit BMP (local/private)

```bash
python scripts/research/bubble_chamber/s113_fullframe_qa.py \
  --bmp /PRIVATE/H50Z_030_01_005.bmp \
  --crosses /PRIVATE/labels.json \
  --private-out /PRIVATE/results/005_lsd
```

This covers the entire full-resolution BMP in overlapping tiles. It uses classical LSD **not** trained AZUNet, and does not recover one unique physical track ID across the whole frame. A/B are geometric ends of LSD segments. The operator marks `f1...f8` are different objects from track endpoints and are only drawn if explicitly present in that frame’s labels.

For the multi-scale LoG bead candidate stage and full-resumable row processing:

```bash
python scripts/research/bubble_chamber/s113_fullframe_s112_peaks.py --help
```

It outputs source-backed candidate peaks, but not expert-verified bubbles. It has row-wise `--start-row`, `--end-row` and `--finalize` controls for resumable work. Save all outputs **outside** Git.

## ROI spline/endpoint/fiducial reviewer (local/private, requires source-derived S111/S112 data)

```bash
python scripts/research/bubble_chamber/physicists_s113_annotation_review.py \
  --root /PRIVATE/S112_export \
  --operator-labels /PRIVATE/labels.json \
  --out /PRIVATE/results/s113_review
```

Open `<ROI>/interactive_annotation_review.html` locally. It has distinct layers for photo-backed sample curves, inferred spans, candidate bead chains, crossings, technical protected masks, geometry endpoints A/B and operator marks. It exports/imports review decisions bound to original image and model SHA256. The review output is not ground truth until validated by a qualified annotator.

## Current limitations

Original AZUNet weights and CUDA environment not available in this local audit. Expert GT (track identities, complete polylines, interaction vertices and physically validated f1…f8) has not been provided. Full-frame graph association is currently missing; S109–S112 original global groups were validated only on selected ROI. Do not claim a complete full-frame reconstruction.