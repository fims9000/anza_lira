# S37 preregistered next-step experiment: safer ANZA/Graph-LIRA links

Declared before examining the S37 test injections, 2026-10-25.

## Scientific question

Can an original, weak-label AZUNet provide useful incremental information for
synthetic gap-link REVIEW prioritization over (i) geometry, (ii) local raw
photometry, and (iii) source-locked ordinary U-Net trained on the same S31 weak
mask manifest? More importantly, can a whole-scene false-join calibration gate
abstain on ambiguous proposals without silently treating repeated pairs as
independent observations?

## Protocol lock

- Data: checksum-verified `1mH2Data.7z` full 145 complete visualMetro frames,
  and 39 original H50Z photo crops from **two photos of one physical run**.
- New split: ascending SHA256 of `S37_SOURCE_LOCK_v1|frame_id`; first 36 frames
  synthetic TEST templates; next 30 CAL templates; remaining 79 TRAIN templates.
  Do not filter frame `070_309`, which was missing from the old S35 input folder.
- Photographic source transfer: train/cal on H50Z-005, test on H50Z-006;
  repeat train/cal on 006, test on 005. This is not a run-disjoint split.
- Models: original unchanged AZUNet, S33 quantile input, and original unchanged
  BaselineUNet, S33 local input. Same machine-only mask manifest in pretrained
  checkpoints; no retraining on held-out source photographs.
- Relation arms: distance, geometry, geometry+raw photometry, geometry+local,
  geometry+AZUNet, geometry+UNet, geometry+both models. All have an equal
  synthetic train/cal/test budget, fixed logistic regression architecture.
- Per direction: 160 train scenes, 450 calibration scenes. 80 test scenes for
  each of four new reserved test seeds: forward 20261221/20261222, reverse
  20261223/20261224. Training/calibration seeds forward 20261201/20261202,
  reverse 20261203/20261204. Each generated scene has exactly two synthetic
  correct and four synthetic incorrect candidate relations.
- Compare previous empirical **pairwise** 2% calibration to new **scene-level**
  risk selection. The latter bounds the event 'one or more false suggested
  links in a generated scene', with Bonferroni-adjusted one-sided exact
  Clopper-Pearson upper endpoint over 13 TRAIN-defined thresholds, 7 arms and
  both source directions; nominal family alpha 0.05, target risk 0.02.
- Mutual-best matching margin fixed at 0.04. If no cutoff passes, ABSTAIN ALL.
  Never tune thresholds or choose the best arm using held-out test labels.
- Primary descriptive test outcomes: false-join scenes per 80, number of wrong
  and correct review suggestions, review recall, AUROC/AUPRC on artificially
  drawn pair labels. Avoid pseudoindependent pairwise confidence statements.
- Report per direction and seed; compare conservative and empirical policies.
  No claim of native H50Z tracking sensitivity, 3-D physics or independent-run
  generalization. A synthetic conditional calibration bound is NOT a physical
  safety certificate or permission to automatically join tracks.

## Known limits and safeguards

Data reuse from earlier studies means these are *fresh simulated injections*
under an improved, predetermined S37 protocol, **not independent physical
replications**. Synthetic scenes reuse real backgrounds from a single run and
track geometry from a single different film. Calibration math treats generated
scenes as iid conditional on the fixed simulation; in particular, results
cannot be extrapolated to real trajectories or new apparatus. Original
physical 0/1 track masks and expert pair labels remain unavailable.

Privacy: commit **only source code, mathematical protocol, data-free tests,
and aggregate receipts**. Never upload original BMP/PNG crops, 7z archives,
visualMetro numeric coordinates, model weights or HTML with embedded photos.
