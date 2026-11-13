# V23: autonomous image-to-graph-to-path experiment (separate exploratory protocol)

## Goal
Implement a complete runnable **image-only** CRACKS pipeline: anisotropic dense detector trained on human nonexpert annotations; predicted skeleton fragments and terminal ports; candidate proposal; one-to-one link selection with explicit NO_LINK; constrained image-guided path; evaluation on real expert sections. The task is to establish the actual end-to-end feasibility and errors rather than extrapolating from V22 expert-supplied endpoints.

## Data boundary
CRACKS v2, official Zenodo record 13926822 (verify original MD5 archives). Segmentation training uses sections 340,342,...392 and crowd certain-blue vs certain-orange labels only, ignoring uncertain-green and unlabeled-white; crowd threshold calibration uses sections 311,313,...327, never expert masks. Expert sections <=300 are opened only **after** all image-only predictions for a section have been computed. Earlier CRACKS experiments have inspected some expert masks; all V23 expert evaluation is therefore exploratory and **not an untouched external confirmation**. All data come from one 3D seismic cube.

## Exact scientific distinctions
1. Anisotropic-feature HGB is **not the historical trained ANZA convolutional network**. It is an explicitly named lightweight new front-end, intended as a runnable surrogate.
2. The deterministic geometry/context score is **not a trained historical LIRA P0**. Thresholds are heuristic and are not calibrated geologic fault-membership probabilities.
3. All endpoints, links and paths are predicted automatically from the image-derived mask. Expert mask pixels never enter an inference API.
4. Pixel metrics use only explicitly certain blue and orange pixels; green and white must not be treated as reliable negatives. Gold-connected skeleton paths are used only to measure connection of two evaluation anchors, not to supply source or destination endpoints.
5. Distinct annotated connected components are not confirmed distinct **physical** geologic faults. Never claim natural false-bridge FPR, E2E geological identity recovery, or industrial reliability from these annotations.
6. Compare on exactly identical image-derived dense probabilities: raw mask, nearest linkage, geometric linkage and context linkage, with no-link allowed. Report positive recall, explicit-orange false pixels, and connectivity of sampled expert-positive anchors with section-clustered uncertainty and denominators.
7. A trained real ANZA checkpoint, independent physical LINK/REJECT/AMBIGUOUS expert labels, natural gap count >=75 and an independent second geological volume remain hard gates for the original **full scientific** claim.

## Status
Local prototype modules and seven unit/fixture tests created; local tests pass. A real CRACKS dataset run is **not yet verified**, and no V23 real-image numerical claims are authorized until that run and per-section receipts are inspected.
