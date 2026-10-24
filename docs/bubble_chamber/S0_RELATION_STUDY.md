# S0: Synthetic-only Graph-LIRA relation isolation

Date 2026-10-24. Isolated research branch `research/bubble-chamber-graph-lira`.

## Question

When the exact 2D fragment endpoints are given to a relation model, does the locally observed optical darkness provide additional evidence over geometric endpoint features for deciding whether two fragments belong to one line?

The source is a controlled generator with known, artificially assigned identity. The real-photo P0 segmentation and stereo calibration are **not** part of S0. We do not train ANZA or Graph-LIRA on real data in S0.

## Frozen protocol

- Seed 20261024; 300 generated scenes total, grouped by scene; train 160, validation 70, test 70. Scene seeds never cross splits.
- Scenes include near-parallel, nearly collinear independent traces, curve-dependent decoys, crossing-like arcs, and missing right-side continuations. Underlying paths have dark segments and sparse speckles. This is **not** a calibrated photographic generator for H2/C3H8.
- Candidate generation identical in both arms: conservative endpoint proposal from `bubble_chamber.pilot.propose_joins`.
- Baseline feature vector: gap length, two outgoing endpoint tangent angles, one local quadratic extrapolation discrepancy.
- Expanded vector: exactly the same geometry features plus segment optical darkness difference/min/max from generated pixels. Ground-truth identity is excluded from model features.
- StandardScaler + class-weighted logistic regression for both arms.
- Threshold chosen on validation only to maximize positive recall with candidate-level FPR <= 2%. Same frozen threshold applied to test, no tuning on test.
- Outcome: accepted true vs false *candidate* relations. There is no actual auto-merge in the original real-photo pilot; all output on real photo remains review-only.

## Results: held-out synthetic test

| Metric | Geometry | Geometry + image |
|---|---:|---:|
| AUROC | 0.789670 | 0.871542 |
| AUPRC | 0.793230 | 0.876856 |
| True accepted / candidate positives | 22 / 171 | 28 / 171 |
| False accepted / candidate negatives | 8 / 119 | 4 / 119 |
| Candidate positive recall | 12.8655% | 16.3743% |
| Candidate false-positive rate | 6.7227% | 3.3613% |
| False bridges / scene | 0.114286 | 0.057143 |

Validation FPR was 1.6529% for both arms; transferred test FPR exceeded the frozen 2% risk gate for both. Therefore **S0 does not authorize automatic linking** despite an improvement in ranking and relative false-positive reduction. Numeric results cannot be extrapolated to JINR images, H2-to-C3H8 transfer, bubble contours, magnetic-field calibration or 3D reconstruction.

Candidate recall is artificially 100% for this generator because the oracle trace fragments are directly supplied. This is not a demonstration that the detector segmentation achieves 100% fragment recall. The benchmark isolates relation decision only. Real false bridges could be much higher.

## Verification

- `python -m pytest tests/test_bubble_chamber_pilot.py tests/test_bubble_chamber_review.py tests/test_bubble_chamber_public_samples.py tests/test_bubble_chamber_relation_study.py -q` — 23 passed.
- `python scripts/research/bubble_chamber/run_relation_study.py --out results/bubble_chamber/s0_relation_report.json` — completed.
- Repeated run produced byte-identical report SHA256 `19e7b8c60d47956d60ed2a06dfa8feaff17b3513beaab198ba5b2bd1ea7d4505` (for the exact report JSON in this commit).

## Required next gate

Obtain independent real, expert-labeled original BMP 3-view acquisitions from several H2 and C3H8 runs (including known same-track and no-match decisions). Compare segmenter variants with the same candidate/physics path and use run/film-disjoint splits. **Do not reuse the S0 scores or thresholds on real film without calibrating on real training/validation data.**
