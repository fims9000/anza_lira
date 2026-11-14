# V45: real Thebe two-stage selective linker with out-of-fold scene gate

**Protocol recorded 14 November 2026 before the first V45 real execution.** This is exploratory posthoc on the previously inspected Thebe REL_CAL VAL1 40..99, not a confirmatory field-independent result.

## Hypothesis and previous evidence
V34 had correct predicted SBPP pair in 65/120 controlled REL_CAL gaps, original matched Logistic top1 36/120, and zero accepted final links under a conservative section-risk policy. V40 same-budget paired-difference ranker improved top1 to 41/120 (+5, descriptive bootstrap interval includes zero), but its high-scoring false LINK or synthetic NO_LINK errors still yielded zero useful accepted cases at all unique CAL score thresholds. V44 naïvely injecting synthetic NO_LINK negatives into **the same pair classifier** collapsed top1 from 36 to 3/120 or 0/120.

Separate **which proposed continuation is best** (candidate ranker) from **whether the first-ranked continuation should be accepted at all** (scene-level gate). The second task has a different supervision signal: a true LINK scene with the wrong top1 must be refused, as must every synthetic candidate-absent NO_LINK scene.

## Frozen design
- Original V33 trained ANZA-v1 checkpoint SHA256 `ba0c3b8060fd0e3cb1c15f6c1598b94a977677c2b075663b5414207df382c25e`, original exact .65 SBPP candidate list <=48. Original source TRAIN1 0..99 and already-inspected relation CAL VAL1 40..99; TEST7 remains unopened.
- Ranker = V40 TRAIN-only StandardScaler + symmetric positive-minus-matched-negative 12-feature LogisticRegression `C=.3, fit_intercept=False`, same original matched training pair labels. No new trained ANZA or P0.
- **Scene-gate training data must not be scored by a ranker fitted on the same source section.** Use deterministic five-fold source-section OOF on TRAIN1, held-out fold = source ID mod 5. Train five fold-specific rankers only on the other 80 sections; extract gate features and correctness labels on the held-out 20 TRAIN sections each, for both LINK and synthetically candidate-absent NO_LINK worlds. Fit the final ranker independently on full 100 TRAIN1 sections and a single scene gate on all OOF TRAIN scene rows.
- The gate targets `1` only when a LINK scene's **predicted** top1 is an 8px-correct candidate; all wrong LINK top1 and synthetic NO_LINK top1 are `0`. The gate receives observed candidate/score/branch features only, **never source gap ID, truth, hidden centerline, physical section ID or intervention endpoints**.
- Predeclared 13 scalar scene features: max rank score; top1−top2; top1−median; mean scores; score std; log1p(candidate count); source-count/4; hard-branch-count/110; mask fraction >=.65; mean model probability; first top candidate's standardized geometry feature (features[0]); top candidate's model probability (features[5]); mean sampled predicted probability along proposed link (features[6]). Missing candidates map to zeros and are fail-closed, not accepted.
- Gate = TRAIN-only StandardScaler + LogisticRegression(`C=.3,class_weight='balanced',max_iter=1400,random_state=20261114`). No hyperparameter tuning or CAL-based feature selection.
- Compare (i) V40 rank score raw and (ii) OOF-trained gate probability for **the same final full-TRAIN ranker top1** on all 120 LINK +120 synthetic NO_LINK original relation-CAL worlds. Exact V40 65/120 candidate availability and 41/120 top1 must reproduce or STOP.
- **Safety:** fixed CAL threshold grid `(0.1,0.2,...,0.9,0.95,1.01)` = 11 thresholds, and conservative Bonferroni simultaneous 95% one-sided binomial bound for `11 thresholds × 2 error types`. Source section = 60 units, two gaps/section. Also report raw best at zero observed wrong source errors and descriptive <=10% empirical section error rates across unique scores; these are NOT deployment thresholds and may expose selection overfit.
- Report complete method-scored source case rows, gate TRAIN OOF positive counts and performance, risk-coverage by method, worst-case count, original source provenance, and the no-TEST restriction.

## Limitations and stop conditions
- V45 VAL1 REL_CAL has been inspected repeatedly. Its new metrics cannot establish independent out-of-sample architectural superiority.
- All source sections come from the same Thebe field; section independence is not proven, so exact binomial intervals are conditional, not real-world geological safety bounds.
- Synthetic NO_LINK is obtained by deleting target areas using expert annotations, **not** an independently labeled natural geological non-connection cohort.
- If the ranker/candidate baseline drifts, any source identity overlaps OOF train and heldout fold, any variant ingests expert endpoints or TEST7, or the confidence/risk grid changes, STOP.
- A publishable field-independent ANZA–LIRA end-to-end superiority claim additionally requires new reference geology, independent fields/volumes, per-pair false-link annotations, and full path completion metrics. V45 alone can at most demonstrate controlled feasibility.
