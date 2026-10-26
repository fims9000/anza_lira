# S59 frozen before benchmark: real-candidate retrieval bottleneck

Date: 26 October 2026. Branch: `research/bubble-chamber-graph-lira` only.

**Question:** Does expanding the candidate graph (not the learned score) increase the number of complete artificial three-piece paths while respecting a fixed calibration-only threshold and the existing S56 global abstention rule?

**Locked evidence:** visualMetro film 070 145 real measured source frames, S37 whole-frame 79/30/36 train/cal/test. H50Z 005/006 is one physical run, used only as authentic image backgrounds. The S56 synthetic simulator supplies the artificial true/false relationship. No native physics GT exists.

**Source model and splits:** exactly S56 geometry six-field StandardScaler and balanced LogisticRegression C=.3, fixed S56 train/cal seeds, S56 per-pair 5% CAL-negative FPR cutoff. Test data use *fresh* locked generator seeds `20265903/20265904` for 005->006 and `20265913/20265914` for 006->005, 64 scenes each; frame split remains the same test-frame partition as S56, so this is NOT an independent physical/film confirmation. Use frozen five S56 scenario types, seven candidates each.

**Candidate policies, no label access in retrieval:**

- `strict_s39`: `3 < nearest_endpoint_gap <= 36`, minimum outward tangent-to-gap >.25, antiparallel outward-tangent cosine >=0. These are S39 geometric gates.
- `relaxed_s59`: `3 < gap <= 48`, minimum outward tangent-to-gap >.05, mutual cosine >=-.20. This is deliberately predeclared and must be evaluated negatively if unsafe.
- `all_pairs_oracle_upper_bound`: every unordered pair of the seven provided *machine-perturbed but source-derived* synthetic polylines; this is an upper bound on retrieval opportunities, not a deployable native method, and corresponds to S56 all-pair input.

A policy never sees candidate labels, generator identity, test scores or physical truth. Evaluate retrieval true-positive link coverage, false candidate burden, and full-path *upper bound* (both true links of a three-piece path available). Next run the IDENTICAL frozen S56 `global_forest` optimizer with margin .04 and pairwise cutoff calibrated only on CAL. Evaluate correct/wrong selected links, complete artificial trajectories, scenes with any false join and scene-level paired bootstrap uncertainty. Count solver failures as zero proposals, never silently omit.

**CAL-only selection for deployment:** choose `relaxed_s59` only if it recovers at least 10 percentage points more CAL synthetic true links than `strict_s39`, with false candidate count no greater than 4 times strict, and the selected global policy does NOT exceed strict's number of CAL scenes with any false join. Otherwise remain with strict. This rule MUST run before viewing fresh test results. Test all three policies regardless, but report whether selected policy was strict or relaxed and do NOT post-hoc choose a test winner.

**Real H50Z diagnostic, conditional on CAL-only approval:** score newly retrieved candidate pairs on the 39 existing S39 graph node lists, using synthetic training/calibration only from the *opposite* original H50Z background. Mark every new pair as `UNKNOWN`, without established S40-S52 risk checks; never automatically approve physical joins or label pixels. Save per-tile pairs/coordinates solely in private JSON, aggregated counts only in public results. No private photos, coordinates, predictions or checkpoints in Git.

**No claims:** This still receives generator-derived fragment coordinates; it is NOT an end-to-end image detector test and cannot establish native particle recognition, physical graph precision, 3D paths or independent multi-run generalization.
