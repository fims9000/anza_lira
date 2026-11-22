# V54: Diagnose missing Thebe true candidates without changing V53

## Scope and freeze

V54 starts from V53 commit `50aceec462403877e796ca5909110383d0b05cfe` on a new research branch. All V33/V43/V46/V50/V52/V53 saved baselines, thresholds, trained checkpoints, and their earlier conclusions remain immutable. This is exploratory candidate-generation research, **not** a new independent external test. TEST7 must remain unopened.

The target is to explain **45/120 positive controlled Thebe gap cases without an 8-pixel-matched source/target candidate** before changing the complete reconstruction pipeline. This study does **not** measure full-slice fault-instance recall or recovery of every geological track. The 120 artificially erased gaps come from 60 previously inspected REL_CAL sections of one real seismic field; the synthetic NO_LINK interventions are not verified real geological negatives.

## Frozen prior result and observed receipt distribution

Original frozen V53 q70 dense-only fallback on positive REL_CAL:
- correct candidate present in 75/120; absent in 45/120;
- first-ranked correct 45/120, conditional successful V52 decoded routes 33/45;
- accepted successful paths under the fixed simultaneously CP-constrained risk scan: 8/120.

From `V46_CASE_ROWS.csv` in successful [V53 CI run 43868877436](https://github.com/fims9000/anza_lira/actions/runs/43868877436):
- 61/120 positive scenes have exactly 48 candidates;
- among 45 missed cases, 26 have 48 candidates and 19 have fewer than 48;
- the 48-proposal cap **cannot by itself explain all 45 misses**;
- a 48-candidate count is not proof that a correct candidate was discarded. It may never have been generated, may have lost a source port, or may be outside an orientation/radius filter.

Candidate generation presently restricts its input to a **known expert-selected intervention rectangle**. No claim of autonomous whole-image fault detection is supported.

## Changes under study

Make the existing `sbpp_proposals` configurable, with original defaults exactly unchanged:
`max_sources=4, per_source_k=12, max_pairs=48, max_source_radius=29, min_alignment=.10`.

Record truth-free stage counts: available ports, ports after radius and direction filters, ports used, proposals before per-source K, and proposals before final cap. The model must never receive `true_ends`, expert track pixels, `hidden_centerline`, case labels, or correctness indicators.

Before looking at V54 outcomes, compare exactly five fixed policies:

| Policy | Sources | K per source | Final cap | Source radius | Minimum alignment |
|---|---:|---:|---:|---:|---:|
| V53 frozen | 4 | 12 | 48 | 29 | .10 |
| K only | 4 | 24 | 96 | 29 | .10 |
| Sources only | 8 | 12 | 96 | 29 | .10 |
| Sources + K | 8 | 24 | 192 | 29 | .10 |
| Relaxed radius/direction | 8 | 24 | 192 | 40 | -.10 |

A final cap of 96 with 4 sources × 12 candidates would be a no-op; expanding final cap must be accompanied by changing upstream source/K restrictions.

## Evaluation

On the same TRAIN1 and previously viewed REL_CAL controlled examples, report for each policy:
- positive gap candidate recall, rescued and lost case IDs relative to V53;
- median/P95 and capped candidate burden;
- source/filter diagnostics and number of complete available pairs;
- candidate burden in synthetic NO_LINK scenes;
- simple geometric top-1 correctness **only as a descriptive fixed comparator**.

Use expert endpoints only after all candidates are generated, with the original fixed 8-pixel criterion. Verify exact V53 case IDs, counts, candidate sizes, and 75/120 correct availability from its archived receipt. Fail closed on drift; retain negative outcomes.

**Do not** label extra NO_LINK candidate proposals as false bridges: a generated possibility is not an accepted connection. Do not compare the learned V53 ranker with wider untrained candidate sets and claim a fair paired model improvement. A model trained or recalibrated on changed proposals needs its own frozen training and independent confirm.

If V54 improves the available-pair denominator, next investigate held-out ranking and path selection. If there is no substantial improvement, diagnose missed source endpoints and ANZA segmentation on whole slices before changing another classifier or searching acceptance thresholds. Never use the previously inspected REL_CAL results to tune a proposed final operating point.

## Acceptance

1. Existing source call defaults and legacy unit tests remain unchanged.
2. The expanded generator produces only candidate geometry/features, never expert labels.
3. V53 receipt counts are reproduced exactly on the identical scenes.
4. Five policy comparisons and stage-level case receipts are persisted.
5. Independent TEST7 is not accessed; no whole-image or geological-instance success claim is made.
