# S43B post-hoc counterfactual control (not independent confirmation)

This analysis was motivated by the S43 held-out result: pooled synthetic AUROC
was high even though 50-58% of its S35 same-renderer decoys were accepted.
Inspection found a possible simulator shortcut: injected measured-track halves
are much shorter than many simple hand-drawn scratches.

S43B is therefore **a post-hoc sensitivity analysis**, NOT an untouched
independent validation set, even when its random seeds differ.

Fixed diagnostics: use the same original film-070 frame train/cal/test folds;
16 train, 8 calibration and 16 test synthetic scenes in each H50Z photo
transfer direction. Four simulated true fragments, four counterfactual scratch
paths formed by translated measured 2D fragment shapes (same length and same
ink-range rendering), original S35 same-renderer decoy, and one length-matched
straight scratch per scene. Standardized balanced L2 logistic regression is
compared with identical variant withholding the two geometric-length features.
All thresholds are calibration q90 of synthetic negatives; never test-tuned.

Image origin, physics identity, scratch material and true line mask of native
H50Z remain unknown. The sole success condition is diagnostic integrity,
not physical performance or deployment authorization. All native 567 proposals,
S40 fiducial risks and 182 graph edges remain unchanged.
