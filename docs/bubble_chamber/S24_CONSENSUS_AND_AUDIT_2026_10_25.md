# S24 — Conservative consensus and human-review rescue on film-070 geometry

**Status:** exploratory, same-film, *expert-measured coordinates with artificially omitted clicks*; **NOT actual photographed H50Z segmentation or automatically trusted physical joins.**

## Scientific objective

Given very little initial expert supervision, can Graph-LIRA increase the fraction of correctly joined trajectory fragments while reducing wrong merges? S21/S22 found a trade-off between long-gap recall and wrong joins. S24 tests two distinct candidate classifiers, a conservative intersection of their one-to-one proposals, and an explicit human review path for disagreements.

## Data and frozen protocol

- Original source is the audited `1mH2Data.7z`, **145 frames, 1,073 expert-measured tracks**, source SHA-256 `bd8e28247c5d178ae4f2f5a86215cbf5626f3ad86560928f4521093ce9188dea`.
- The same original film `070` supplies all three disjoint-within-split partitions; this is **not independent filming/run validation**.
- New split seeds: `20261030, 20261031, 20261032`; each holds out **whole source frames**, including all stereo views.
- Training uses artificial missing-click gaps `1, 3, 5`; testing uses `5` withheld measured clicks. Missing regions are not confirmed real photographic occlusions.
- Expert-review simulation consumes **640 training relation answers** per split; separately **80 calibration reviews** from some validation source frames and **80 independently selected audit reviews** from other validation frames (total = 800 before test-only simulated review). The complete candidate graph remains visible; labels of unreviewed candidates are never used in fitting or calibration.
- Both classifiers use **identical training labels and optimization budget**: ordinary 14-feature geometry HistGradientBoosting and 24-feature geometry + graph-neighborhood-context HistGradientBoosting. Neither uses original AZConv pixel feature tensors. The latter is a Graph-LIRA *geometry proxy*.
- Probability thresholds are chosen from reviewed calibration labels only, allowing **zero observed false joins among the 80 reviewed calibration pairs**. Remaining unreviewed candidates still compete in mutual-best matching. The second validation subset is **audit only**, not used to tune thresholds, models, or margin.
- A cautious consensus proposes an edge only when **both complete-scene matchers** select the same edge. All other proposed edges remain human-review questions. No physical automatic merge is authorized.

## Held-out (new frame splits) test results

| Predictor | Correct pairs (TP) | Wrong joins (FP) | Correct continuations possible | TP / possible |
|---|---:|---:|---:|---:|
| Ordinary geometry HGB | 882 | 5 | 993 | 88.82% |
| Graph-context HGB | **900** | 9 | 993 | **90.63%** |
| **Conservative consensus** | 834 | **2** | 993 | 83.99% |

Exactly 992 correct continuations are represented in candidate lists; one valid continuation is missing from the generator's top-5 candidates. The aggregate candidate count is 5,149. Consensus reduced wrong joins versus graph-context HGB at the expense of 66 fewer correct joins. The same-film result is **not** a safety or statistical guarantee for new photographs.

Per split (geometry TP/FP, graph TP/FP, consensus TP/FP):

- `20261030`: 286/2, 280/7, **258/1**
- `20261031`: 307/1, 316/0, **289/0**
- `20261032`: 289/2, 304/2, **287/1**

## Small human-review budget applied *after* the automatic consensus

Using the *withheld test track identities as an offline proxy for later human decisions* (not actually performed human reviews), the queue examines model-disagreement pairs. Reviewed positive pairs may be added only if their endpoints are unused. Negative and conflicting pairs are never accepted. These are additional review decisions per split and **do not modify the trained models**.

| Extra simulated reviews **per split** | Verified correct pairs added across three splits | Final correct total | Wrong auto joins left |
|---|---:|---:|---:|
| 20 | 56 | 890 | 2 |
| 40 | 96 | 930 | 2 |
| 80 | **114** | **948** | 2 |

The deterministic hash-order control added **55, 94, 114** respectively; the particular confidence/graph-disagreement ranking did **not meaningfully outperform** this control at the tested review budgets. Do not claim that acquisition strategy is proven superior.

The 114 additional correct links are **not** autonomous: they are a counterfactual result assuming an expert confirms those relations individually. The original two false consensus joins are *not* eliminated merely by checking a different set of edges.

## Why a small zero-error validation set is not certification

The separate 80-candidate audit samples contained only 10–12 *accepted* candidates per model for each split. Hence, even observing zero wrong joins gives a one-sided **iid-only illustrative 95% upper error rate around 22–26%** with such few accepted checks. One wrong among ten gives around 39%. These values cannot establish real physical trustworthiness; samples are same-film and frame-correlated.

For ideal **independent random accepted joins**, even **zero** observed errors would require at least **299** reviewed accepted joins to obtain an iid one-sided 95% upper bound below 1%. This is a theoretical sample-size illustration, **not** a proposed physical release gate or valid confidence interval under our same-film source. Real photos and separate physical runs are required.

## Accept/reject decision

1. **Retain** both calibrated graph and ordinary geometry classifiers. The ordinary model remains competitive and sometimes more accurate.
2. **Retain consensus** as a safe *review-first* option because it cut false joins from nine to two in the fixed held-out benchmark, while keeping track that it misses 66 correct continuations.
3. **Do not approve automatic physical joining.** Two wrong unions are still present, and independent real run/medium/fiducial calibration is missing.
4. **Do not claim uncertainty queue wins** when the random/hash control performs virtually the same.
5. Next genuine scientific gate: get expert-reviewed true BMP line/artifact scribbles **and track identities/pair reviews** on different source photos/runs, train original AZUNet on real sparse masks, and evaluate Graph-LIRA end-to-end on photographs rather than simulating missing clicks.

## Reproduce

```bash
python -m pytest -q tests/test_s24_consensus_review.py tests/test_s21_selective_graph.py tests/test_s22_curriculum.py
python scripts/research/bubble_chamber/s24_consensus_audit.py \
  --measurements /private/1mH2Data/S1 \
  --out results/bubble_chamber/s24_consensus_study.json
```

Code explicitly refuses changing the source film ID, invalid review budgets, malformed matchings and duplicate source candidate identities; `unattended_release` and physical automation remain disabled.
