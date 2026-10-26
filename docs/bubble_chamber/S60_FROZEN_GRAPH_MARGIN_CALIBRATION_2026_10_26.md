# S60 frozen before CAL and new heldout testing: graph-stability margin

26 October 2026. Branch `research/bubble-chamber-graph-lira` ONLY.

## Question
Can a **calibration-only** choice of the S56 global objective-stability margin recover more artificial correct joins/complete paths WITHOUT increasing the number of calibration scenes with a false join? Previous S59 proved expanding the pair retrieval did not change any global selections on fresh synthetic test scenes. No hyperparameters are chosen from S59 test feedback.

## Data and locked features
Same 145 audited visualMetro 070 source-frame set and frozen 79/30/36 whole-frame split. Source H50Z 005 and 006 authentic backgrounds belong to a SINGLE run and remain unlabelled. Same S56 synthetic simulator (two artificial three-piece paths + one decoy, five scenarios). Training and calibration use the exact frozen S56 64/64 scene seeds, the same six geometry features, StandardScaler, balanced LogisticRegression C=.3 and CAL-only score threshold targeting 5% negative pair FPR. Candidate policy remains `strict_s39`; only the global-stability margin is allowed to change. Baseline is S56 fixed margin **0.04**.

Grid predeclared: `(0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08)`. Values are per-photo-direction, never fit to any test target background. CAL-only acceptance of a challenger requires: zero solver failures; number of CAL synthetic scenes with >=1 wrong join <= baseline at 0.04; >= **8 additional correct CAL joins** (out of 256) and >= same CAL fully recovered artificial paths as baseline. If none qualify, use original 0.04. Among qualifiers prefer more correct CAL joins, then fewer wrong-join CAL scenes, then higher full-path count, then smaller margin; reproducible deterministic tie break.

Freeze the CAL decision to JSON before opening any fresh test. New immutable test seeds: photo 005->006 `20266003` and `20266004`, photo 006->005 `20266013` and `20266014`, 64 simulated scenes per seed. All metrics and 95% paired **scene-level** bootstrap CIs are conditioned on the simulator and source-frames, not physical film holdouts. Test both baseline 0.04 and frozen CAL-selected margin, NO test-set retuning or winner picking.

Native H50Z: score S39 pairs with model trained on the opposite photo; selected edges are reversible `UNKNOWN` machine hypotheses and are NOT validated particle joins. Preserve all source nodes, images and older uncertainty warnings; do not upload native photos/coordinates/private records to Git. Native changes are observational advisory counts only.

S60 does NOT test end-to-end image extraction because its seven simulated fragment coordinates are source-derived and perturbed. No physical-precision, mass, charge, momentum, 3D or independent-run claim.
