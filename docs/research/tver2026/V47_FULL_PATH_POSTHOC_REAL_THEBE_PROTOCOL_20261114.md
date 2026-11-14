# V47: complete controlled path readout for coherent V46 ANZA→SBPP→ranker→gate cascade

**Registered 14 Nov 2026 before real V47 inference; previously inspected Thebe VAL1 REL_CAL, exploratory only.** V46 frozen ANZA V33 model, q=.70 density-only fallback, TRAIN-matched pair ranker and source-section OOF-trained scene gate stay unchanged. Original 120 controlled LINK and 120 synthetic candidate-absent NO_LINK worlds from 60 Thebe VAL1 sections 40–99. No TEST7.

## Question
Candidate availability, rank top1 and selective pair acceptance are not full geometric path recovery. After the V46 gate, can the repository's existing deterministic **path-readout stage** actually bridge the deliberately erased middle segment of the correct pair? Preserve both successful and unsuccessful geometries; do not convert a top1 classification score into path success.

## Frozen methods, source and metrics
- Use the original repository `scripts.tver_v26_predicted_cracks_e2e._path_success`, not a new path-selection algorithm, with `method='straight'` and `method='maximin'`, on the **actual** top1 pair in each of the same frozen LINK scenes. Maximin uses a predicted-probability and seismic-gradient corridor, 8-neighbor maximum bottleneck graph path; no hidden expert line in path computation.
- Expert hidden centerline may be accessed **only after a path is constructed**, to score: within the original erased rectangle, 2px-dilated precision >=0.80 AND recall >=0.80. Pair correctness is still the original 8px endpoint tolerance, separate from path success.
- Evaluate path success for all 120 positive controlled gaps, and for every fixed predeclared V46 gate threshold `(.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,1.01)`, record accepted correct pairs, accepted correct paths for straight/maximin, wrong LINK-source and synthetic NO_LINK-source counts. Choose no deployment threshold on REL_CAL.
- Report a separate *descriptive* best path count among thresholds with <=6 wrong LINK source sections and <=6 accepted synthetic NO_LINK source sections; this is posthoc on already viewed CAL and **not safety guarantee**. Independently, if any of the V46 simultaneous Bonferroni 95% CP risk-admissible thresholds exists, report the count of successfully reconstructed paths at those thresholds, else zero.
- Original V46 frozen source controls still must reproduce 116 TRAIN/65 CAL initial candidate, 125 TRAIN/75 CAL with q70 before allowing path readout; full case receipts and source checksum are required. Natural line identity, uncertainty over independent geological volumes, autonomous gap localization, true fault topology and original TEST7 remain untested.

## Scientific boundaries
Even a recovered controlled missing segment is not an autonomous map-level geophysical reconstruction. Both the gap rectangle and expert hidden centerline originate from field annotations; only the former is passed to candidate generation and the latter is scoring-only. All CAL conclusions posthoc and clustered in one field, not independent geological evidence. New statistical significance claims are not justified without independent natural fault-instance truth and separate held-out volumes.
