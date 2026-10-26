# S43 preregistered protocol: machine-only scratch counterexamples

Frozen before inspecting S43 test metrics; implementation must not retune on the held-out photo.

## Question

Can image-profile and geometric features rank synthetic known-origin film-scratch
counterexamples below simulated charged-particle-like curves, without expert
pixel labels? What happens on the deliberately indistinguishable, matched-stroke
decoy from S35 and on new open-set scratch families?

## Data and leakage boundaries

* Real H50Z image crops are used as **unlabeled backgrounds**; the two BMPs
  belong to the same physical run. No native particle label is available.
* Source visualMetro film-070 tracks are real operator-clicked 2D geometry but
  are unrelated to H50Z; artificial injection into H50Z is not native truth.
* Split 145 audited source frames deterministically by SHA-256 frame ID into
  72 development-training / 37 development-calibration / 36 held-out-test
  frames; no measured frame may appear in more than one set.
* Fit classifier on artificial scenes placed on photo 005, calibrate on photo
  005, test on photo 006; then reverse 006 -> 005 independently. This is
  photograph-disjoint test, **not run-independent** test.
* Freeze simulation seeds and counts: 24 train, 12 calibration, 24 test scenes
  per direction; comparison includes all line proposals, not only selected easy
  negatives. Use deterministic source frames and scene-local randomness.

## Candidate origins and controls

* Four synthetic measured-curve segments per scene are positives **only in the
  injection simulator**. No physical particle label is implied.
* Synthetic negative families: the S35 same-renderer matched false continuation,
  a straight scratch, a double rail, a broken/periodic stroke, a wobbly scratch.
  A broad, patchy streak family appears **only in held-out test** as an open-set
  negative control.
* Test negative control remains the S35 matched false continuation, which uses
  the same pixel-generation function as the simulated positives. The model
  must not be represented as physics truth even if it scores easy scratches.

## Frozen comparators and decision constraints

1. S39 raw local center-vs-flank contrast (baseline).
2. S39 multi-scale Hessian ridge support (baseline).
3. Frozen StandardScaler + balanced LogisticRegression(C=1.0, max_iter=1000),
   trained using a preregistered fixed path geometry + 2D intensity profile
   feature vector. No AZUNet used as ground-truth teacher.

Select threshold from **calibration negatives only** to aim at at most 10%
synthetic decoy acceptance. Report held-out TPR and FPR overall and by scratch
family, scene-grouped AUC, paired scene bootstrap uncertainty, and negative
results. The test photo must not be consulted for threshold changes.

## Native H50Z deployment boundary

Evaluate every S39 candidate on original unchanged H50Z crop pixels and attach
simulator-trained ranking as **advisory only**. Never erase native proposals,
create 0/1 pseudo-labels, treat low score as proven artifact, or auto-join
physical tracks. Existing known fiducial risks from S40 remain hard uncertainty.
Graph scores require exact source identity validation. Private model features,
coordinates, photo crops and visual overlays are excluded from public Git.
Only code, tests, preregistration and aggregate metrics may be committed.

## Failure gates

* Matched-decoy FPR exceeding 10% on either held-out direction => NO SCRATCH
  REJECTION GATE; ranking only, with explicit warning.
* AUC improvement on just one photo is not evidence of transfer benefit.
* Confidence intervals from scene resampling describe simulator uncertainty
  only and must not be interpreted as real physical-world performance.
