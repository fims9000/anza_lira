# S80 frozen multiscale ridge-instance proposals: CAL then FRESH TEST

**2026-10-26** — branch `research/bubble-chamber-graph-lira`.
This protocol was written BEFORE any S80 CAL or TEST outcomes were inspected. Prior S76–S79 reported results are historical motivation, NOT S80 tuning examples.

## Objective and claim boundary

S76–S79 show that blurring eliminates image-derived trace instances and that reconnecting fragments or adding geometric links can worsen graph results. S80 asks whether adding *separate, pixel-only, scale-specific ridge-instance hypotheses* restores artificial tracks without destroying original line instances or increasing wrong graph joins.

**Native H50Z physics ground truth = 0.** Both H50Z backgrounds (`005`,`006`) derive from ONE physical run, while artificial curve templates are measured from DIFFERENT visualMetro film 070. The evaluator's synthetic labels are never shown to pixel inference. No 3D/mass/particle identity claim. No physical auto-join authorization.

## Frozen variants (pixel input only)

- `baseline`: unchanged S72 CAL-selected per-direction graph and pixel extractor, original S56 geometry scorer, 5% negative-pair threshold and global forest stability margin `.04`, with S59 strict pair gate.
- `s80_medium_add4`: preserve all original lines; separately run S39 ridge extraction on the existing **sigma 2.2** Hessian response image with the SAME darkness map/3.5% mask/3×3 closing/junction guards; append up to **4** non-duplicate new ridge instances without deleting originals, max total 32.
- `s80_consensus_add6`: preserve all originals; detect candidate lines on **sigma 2.2** and **sigma 3.5** Hessian maps, but only accept a sigma 2.2 line supported by at least 60% of its samples within 3 px of some sigma 3.5 line **and vice versa** (coarse line has at least 40% sample coverage within 3 px); append up to **6** such non-duplicates, max 32.
- Duplicate exclusion: reject an appended instance when its points and any original/accepted instance have at least 75% bidirectional sample coverage within radius 3 px. This is an unlabeled geometry guard, not a physics judgment. Original line objects/coordinates remain exact.
- Original S39 scoring, selection, S69 orientation-invariant synthetic evaluator and S76 shape/endpoints audit remain unchanged. All new arms are evaluated using same frozen fitted scorer and cutoff; MILP failure reports failure and clears selected edges (not fake success).

## Frozen split, seeds, conditions

- Reuse audited 145 measured visualMetro film-070 frames, deterministic **79 TRAIN /30 CAL /36 TEST** whole-frame split with fingerprint `448f21ca3a709010ed972a596ed046cadff0643112905fd1d029c91f8b14faef`.
- Fit source-photo S56 model only from frozen TRAIN seeds; freeze original S56 source-CAL 5% simulated negative-pair threshold. Per-direction S72 graph branch is locked to `docs/bubble_chamber/S72_PUBLIC_FROZEN_CAL_2026_10_26.json` with SHA `1ed7e58667aaab1a7d5ba91252db64b38bf8d3da517396b9e18e9e6bb484a952`.
- S80 **CAL** uses fresh source-photo overlays: seed `005:20268001`, `006:20268011`, **16 scenes per direction**. Both clean and blur sigma 1.1, paired on same scenes. All three arms evaluated (no manual cherry-picking).
- S80 **FRESH TEST** uses new heldout visualMetro TEST frames and opposite H50Z photo: source `005` to target `006` seeds `20268003,20268004`, reverse seeds `20268013,20268014`, **24 scenes per seed (96 total)**. Report clean, blur sigma 1.1, blur sigma 1.6. No retuning after CAL or TEST inspection.
- For every arm/condition, report: synthetic-correct joins, noncorrect/unknown joins, full synthetic 3-piece paths, incorrect-proposal scenes, matched true fragments by **shape-only** and those with both endpoints close, fragment-endpoint failures, proposal and candidate counts, additional lines and solver failures. Include paired 1200-bootstrap scene confidence intervals separately per direction/seed, not over pixels or across physical camera runs.

## CAL-only selection and future deployment gate

S80 arm is CAL eligible only if all are true: on blurred (sigma 1.1) scenes, its synthetic-correct links increase by **at least +2**, its synthetic-noncorrect links **do not increase**, its whole three-piece paths **do not decline** and at least **2 more true artificial fragments are matched by shape**; on clean scenes, correct synthetic joins decline by no more than **1**, synthetic-noncorrect joins do not increase, full three-piece paths do not decline; MILP solver has zero failures. If multiple arms qualify, select most blurred true links, fewest blurred wrong links, fewer added fragments, lexical arm. If none, retain `baseline`. Save CAL decision and immutable CAL/protocol/operator hashes before test. TEST cannot alter selection.

Do not promote automatically even with a passing synthetic TEST: lack of independently annotated physical runs prohibits native validation. Native outputs stay `UNKNOWN`, and all input photographs and measured coordinates remain private.
