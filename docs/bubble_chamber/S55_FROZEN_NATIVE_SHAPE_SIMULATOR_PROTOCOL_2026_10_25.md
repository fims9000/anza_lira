# S55 frozen protocol | Simulator support for native MACHINE-CANDIDATE geometry

**This is not self-training on physical particles.** The source is S39 machine proposals, with ZERO expert labels. Geometries are sampled from proposal polylines without using any assumed track identity or feature-class truth. Label 1/0 is generated solely by the procedural *renderer* of a **new artificial stroke** on a different native crop.

## Split and leakage policy

- 567 original S39 proposals are partitioned by photo. Train/cal machine candidate IDs belong **only to one source photo** (005 or 006). A deterministic SHA-256 candidate-ID sort allocates 80% train and 20% calibration; no candidate ID is reused across those splits. Because both photographs belong to the same physical run and may overlap, this is NOT an independent run/physics split.
- Test source geometries come only from the opposite photo, never used for fitting or calibration. Artificial stroke renderers use that photo's native grayscale crops. Ensure each artificial stroke geometry is overlaid on a **different crop** from its source, to avoid rendering on top of the original unidentified line.
- Preserve the candidate's image-plane shape. Permitted transformations: select a contiguous measured-pixel path subsegment (no extension to larger length), arc-length interpolation, **translation only**, and identical subpixel/jitter for all renderer variants in a paired scene. No scaling. All artificial renderers share source centerline, background, nominal ink budget.
- Original S53 frozen models (trained on unrelated smooth film-070 measured shapes) are scored as fixed reference on the **same** fresh target-photo synthetic cases; cannot fit or tune them on S55 target.
- Compare newly fitted S43 (14-feature) vs S55 S43+S53 (22-feature) vs the original frozen S53 (22-feature). Identical LogisticRegression specifications for newly fitted arms. Reject any conclusion of real-track precision.
- Train/cal/test scene counts: 40/20/40 per direction and per test seed. Test only: matching-bead negative renderer and previously unseen clustered-bead negative renderer. 99th-percentile calibration novelty gate is calibration-only and never grants physical identity.

Fixed seeds: source photo 005 train 20265501, calibration 20265502, test 20265503/20265504; 006 train 20265511, calibration 20265512, test 20265513/20265514. Freeze before viewing test metrics. No test-set tuning, and no fake positives from native photo annotations.

**Delivery gate:** only machine-hypothesis scores, OOD warnings, and simulations. All 567 native proposal lines and 182 graph candidate relations remain unchanged. None may be automatically accepted as a physical particle track or join.
