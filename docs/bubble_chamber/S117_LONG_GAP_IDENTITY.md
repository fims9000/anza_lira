# S117: provisional long-gap identity for S109/S115 trajectories

S117 is an **optional, source-constrained post-processing stage** for already
extracted vector curves. It does not modify segmentation masks or fabricate
image-supported pixels. It assigns stable *provisional* group IDs to fragments
whose continuation is geometrically and photometrically supported, including
gaps longer than S110's 80 px limit. Review-only alternatives remain exported.

## Algorithm and invariants

- Uses endpoint tangent consistency, cubic Hermite interpolation, original-photo
  evidence, absolute lateral drift (parallel-track jump guard), nearby rival
  fragments, blocked regions, mutual-best matching and score margin.
- Each endpoint belongs to at most one accepted link. Joining a cycle is
  prohibited. Conflicting inferred bridges require review.
- Inferred connectors are exported separately with
  `connector_is_observed=false`, `observed_connector_pixels=0`.
- A group ID is a stable hash of the member curve IDs, **not** a physical
  particle identity. Expert decisions `same/different/uncertain` are optional
  and remain explicitly distinguishable from algorithmic proposals.
- Operates on cropped ROIs of at most 4 Mi pixels. Whole-frame stitching and
  spiral-specific curvature inference are still open issues.

## Reproduce

From repository root, with OpenCV, NumPy and SciPy installed:

```bash
PYTHONPATH=. pytest -q tests/test_s117_gap_identity.py tests/test_s117_runner.py
PYTHONPATH=. python scripts/research/bubble_chamber/run_s117_long_gap.py \
  --curves /path/to/S115_GEOMETRIC_CURVES_NOT_GT.json \
  --photo /path/to/SOURCE_GRAY.png --out /tmp/s117_output
```

Use `--response /path/to/source_response.npy` when S114's float directional
response is available; a rendered/normalised preview PNG is **not** equivalent.
JSON output includes all alternative links, their review decisions, grouped
members and provenance SHA-256. The image preview uses one color per provisional
group and orange dashed connectors for interpolated geometry.

Manual RLE audits require matching original BMP + annotation JSON and produce
only a partial annotation-stroke ID consistency audit, **not** exhaustive recall:

```bash
PYTHONPATH=. python scripts/research/bubble_chamber/audit_s117_manual_rle.py \
  --annotation /private/H50Z_030_01_007.json \
  --bmp /private/H50Z_030_01_007.bmp --results /private/s117_007 \
  --roi A_top:3600:6000:1200 --roi B_mid:3600:9600:1200 \
  --roi C_dense:3600:14400:1200 --roi D_lower:2400:15600:1200 \
  --roi E_side:4800:7200:1200
```

## Current provisional audit (one frame, five selected 1200 px ROIs)

With S114 directional response, S115 observed curves and the provided manual
H50Z_030_01_007 annotation: 376 S115 geometric curves; 75 S117 provisional
links; 60 consistent with confidently matched manual line IDs, 15 unmatched
or ambiguous, 0 matched-label conflicts after adding the absolute lateral
projection guard. Initial variant had 2 matched-label conflicts, one an obvious
jump between nearby parallel tracks. This is an **in-sample development audit**;
none of the 75 links is a certified physical identity, and the matching does
not cover unlabeled or weakly labeled segments.

Do not commit H50Z BMPs, annotations, derived image crops or private reviewer
artifacts into this public repository. Code/tests are based on synthetic
fixtures. Real image outputs must remain in access-controlled storage.
