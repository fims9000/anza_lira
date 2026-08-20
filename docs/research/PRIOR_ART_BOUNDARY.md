# ANZA-2 prior-art and claim boundary

ANZA-2 does not claim novelty for orientation-aware segmentation, dynamic
sampling, directional connectivity, topology losses, learned affinity,
partial-label learning, consistency training, minimal paths, or max-min/widest
path reachability.

Relevant controls named by the frozen task packet include:

- joint orientation and segmentation for road connectivity;
- Dynamic Snake Convolution;
- DconnNet directional connectivity;
- Path-CNN/minimal-path segmentation;
- WPRF-like learned affinity and max-min reachability;
- decoder/decision repair for curvilinear structures;
- CRACKS crowd-to-expert label-domain adaptation.

The narrow testable hypothesis is:

> A multimodal axial reciprocal-scale fuzzy field provides incremental edge
> evidence beyond a capacity-matched generic learned affinity and a simple
> tangent/distance geometry baseline when backbone, supervision, support graph,
> and reachability are held fixed.

Consequently:

- beating only U-Net is insufficient;
- a diagnostic orientation field is not itself a contribution;
- widest path is downstream algebra, not the proposed novelty;
- `Anosov-inspired reciprocal hyperbolic parameterization` is allowed, while
  `Anosov system`, ergodicity, and uniform hyperbolicity claims are forbidden;
- the strong structural claim is forbidden unless ANZA-conditioned affinity
  beats generic plus simple geometry at the pre-specified low-FPR gate.
