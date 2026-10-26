# S53 frozen protocol | Can local bubble-like texture discriminate track renderings from scratches?

**Scientific scope:** machine-only synthetic-origin study. Not a native physics track classifier, not a dataset of hand-labelled physical particles, not a proof of particle identity. Original AZUNet, U-Net and Belyaev priors unchanged. No manual annotation requested.

## Data and freezing

- Use the already-audited 145 measured film-070 visualMetro frames to generate *unrelated artificial 2D curves*; split by whole frame with the pre-existing S43 frozen 72/37/36 train/cal/test split (the test origin is never used for fitting or calibration).
- Use authentic 512 x 512 crops of H50Z photographs 005/006 as unlabeled backgrounds: train+cal on one, test on the other. The two photos are the **same physical run**, so this is not a run-independent validation.
- Use one identical source background, identical measured 2D curve, same arc length, same nominal darkening budget, and same extracted centerline coordinates for each positive and negative in a paired scene; change only the **renderer of the injected stroke**.
- Synthetic positives: randomly spaced dark dots along the curve (*bubble-like renderings*, not verified bubble physics). Synthetic negatives seen in fitting: continuous hairline, periodically interrupted stroke, scattered dots around the path. Test-only negative controls: same bead renderer as positive (different random seed, intentionally origin-indistinguishable) and unseen clustered dark dots.
- **No negative should be interpreted as a real film scratch**; renderer-origin is the only available truth. The same-bead clone can never support a physical specificity claim even if a classifier scores it differently by finite-sample chance.

## Comparison and selection

Compare three fixed arms, on exactly the same synthetic held-out cases:
1. Local flank contrast alone.
2. Existing 14 S43 features + standardized balanced L2 LogisticRegression (C=1, deterministic).
3. Same 14 features plus predeclared S53 along-path fluctuation/peak/bandpower features, same classifier configuration.

Use 40/20/40 train/cal/test scenes per direction and TWO fixed fresh test seeds per direction. Thresholds are 90th-percentile negative scores on calibration scenes only (strict `>`); no threshold selection or model change using test labels. Report per-renderer positive/negative acceptance, overall and **hard-clone-only** AUROC, scene-blocked paired-bootstrap AUROC differences, and restrictions. All bootstrap intervals are simulator-scene conditional and are not physical-run uncertainty.

Fixed train/cal seeds: 005 (20265301/20265302), 006 (20265311/20265312). Fixed test seeds: 005->006 (20265303, 20265304), 006->005 (20265313, 20265314). Do not alter after results observed. Source program and data checksums retained in the public aggregate (no per-photo pixels).

## Deployment gate

Even if a synthetic discriminator performs well, it may **only attach an advisory to native machine candidates**. Never silently set native GT, erase candidate centerlines, promote `unknown` to positive/negative, auto-join physical traces, or leak native photos/checkpoints to the public GitHub branch. If hard-clone or unseen clustered-dot false acceptance is high, explicitly call the method **not physically discriminative**.
