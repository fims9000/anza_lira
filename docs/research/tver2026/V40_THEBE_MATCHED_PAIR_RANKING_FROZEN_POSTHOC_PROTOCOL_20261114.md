# Tver V40: same-label-budget paired-difference ranking on frozen Thebe ANZA

**Status:** posthoc exploratory hypothesis fixed 14 Nov 2026, after V34 REL-CAL results, before V40 new model training/evaluation. No natural geological instance identities, no new untouched held-out inference.

## Objective
V34 on real Thebe VAL1 40..99: correct candidate 65/120, best matched per-pair logistic top1 36/120, native P0 top1 13/120 and 17/120, final accepted zero. The V39 paired audit found P0 only contributes 1 or 4 unique correct top1 cases not found by the matched logistic. Test whether learning a *pairwise rank* instead of independently classifying pairs improves top1 on the same frozen 12-dimensional observed SBPP features.

## Immutable experimental conditions
- Use V33 exact original trained ANZA checkpoint SHA256 `ba0c3b8060fd0e3cb1c15f6c1598b94a977677c2b075663b5414207df382c25e`, original normalization and threshold 0.65.
- Original TRAIN1 all 100 numeric source sections and VAL1 relation-CAL 40..99 only; official Thebe external TEST7 remains unopened.
- Recompute original predicted SBPP candidates in the same gap interventions. Baseline must reproduce 116 correct candidate TRAIN cases, 65/120 VAL1 correct-candidate availability, and matched logistic 36/120 VAL1 top1. Otherwise STOP.
- Use exactly the original `paired_training_examples`: every true predicted TRAIN pair with one uniquely matched real predicted negative (same scene preferred). No generated true endpoints, oracle pairs, or additional positive labels.
- Baseline matched LogisticRegression uses those same 12-dimensional features, labels, `C=0.3` and TRAIN-only StandardScaler.
- New ranker uses the same exact matched pairs but forms standardized feature differences `d=z_positive-z_negative` with its sign-flipped counterpart and minimizes paired LogisticRegression loss, `fit_intercept=False`, `C=0.3`. Paired/symmetric objective duplicates optimization rows, not independently labeled observations.
- Compare raw top1 rankings (not calibrated probabilities) on all original 120 artificial VAL1 gaps. Paired whole-source-section bootstrap 10,000 replicates describes delta uncertainty; no independent-volume confidence claims.
- Report complementarity and all zero-coverage or negative outcomes. No acceptance threshold selection or assertions of natural geological false-link control. P0 remains frozen and excluded from new training.
- VAL1 already inspected in V34/V38: this is **not** prospective independent confirmation. Do NOT open TEST7 in this branch. Only after the architecture and ranking/risk protocol are frozen can a separately audited one-shot TEST be discussed.

## Decision rule
Keep a new ranker only if it improves top1 against the *reproduced* same-budget matched logistic, with full paired case receipts. Improvement here is exploration only. A final publication superiority claim requires a new independent field/volume and verifiable risk-coverage on genuine negative relations.