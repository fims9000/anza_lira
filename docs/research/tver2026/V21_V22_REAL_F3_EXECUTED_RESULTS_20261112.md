# V21/V22 CRACKS v2 on real F3 pixels (12 Nov 2026)

**Status: executed exploratory diagnostic, NOT full ANZA–LIRA E2E.**

- [V21 successful workflow](https://github.com/fims9000/anza_lira/actions/runs/43297677785), case artifact 13978679810.
- [V22 successful workflow / final test fixture](https://github.com/fims9000/anza_lira/actions/runs/43299046857), case artifact 13980162525.
- [Independent successful V22 repeat](https://github.com/fims9000/anza_lira/actions/runs/43298906903), artifact 13980596647.
- Pre-execution protocols: `V21_GEOMETRY_MATCHED_NOLINK_FROZEN_PROTOCOL_20261112.md`, `V22_MIRRORED_EQUAL_GEOMETRY_FROZEN_PROTOCOL_20261112.md`.

## V21: matched observed expert endpoints and synthetic NO_LINK

From the fixed V19 pool (71 close expert-connected-component pairs, 132 directed potential cases) construct 58 two-candidate positive cases on 16 expert sections, requiring source-to-target distance ratio [0.85,1.15], heading-cosine gap <=0.06, endpoint-tangent compatibility gap <=0.10, target separation >=5 px. 34 of 58 support an additional synthetic NO_LINK world in which neither proposed endpoint belongs to the source's annotated connected component. Those are **not verified geological negatives**.

Separate by section parity, with thresholds chosen only on even CAL: 21 positive + 12 NO_LINK. Odd TEST: **37 positive + 22 NO_LINK**.

| Policy | Positive branch chosen correctly /37 | Correct accepted /37 | False accepted NO_LINK /22 |
|---|---:|---:|---:|
| Visible heading | 36 | 34 | **0** |
| Geometry | 36 | 35 | 2 |
| Geometry + true central section | 37 | 36 | 1 |
| Geometry + genuine immediate neighbors | 37 | 36 | 2 |
| Geometry + distant sham | 36 | 35 | 2 |

Visible heading still correctly orders 56/58 full positive cases, revealing **selection bias and insufficient difficult examples** even after near-geometry matching. The CAL cap <=10% empirical false acceptance is NOT a prospective statistical risk guarantee. With 0/22 and independent-binomial assumptions, one-sided 95% upper bound is **12.73%**, and image cases sharing seismic sections are correlated.

## V22: exact geometry-tied synthetic mirrored continuation

The real V21 *positive* source/true endpoint rows are reused. A false endpoint and its tangent are exactly mirrored about the visible tangent through the source. Original expert line and synthetic reflection have equal distance, heading cosine and abs tangent compatibility to numerical error <=1.42e-14. Exclude reflected endpoints closer than 5 pixels to the true endpoint; **40/58 eligible cases on 15 sections**. Original F3 seismic pixels remain visible. Models score **only raw image gradients** along the two straight candidate paths; labels and hidden expert centerline are not passed to scoring. Candidate order is SHA256-flipped.

| Scorer | Correct /40 |
|---|---:|
| Geometry tie, deterministic candidate 0 | 17 |
| **Real central F3 image gradient** | **38** |
| Genuine adjacent F3 gradients | 37 |
| Real central + adjacent | 37 |
| Distant-section sham gradients | 18 |
| Gradient similarity to 12 visible source samples | 13 |

Paired section-bootstrap (10,000 resamples across 15 sections, seed 20261112):
- Central minus distant sham: **+50.0 pp [95% +30.6,+68.8]**.
- Real adjacent minus sham: **+47.5 pp [95% +25.8,+67.6]**.
- Central minus equal geometry: **+52.5 pp [95% +35.0,+68.6]**.
- Central minus real adjacent: **+2.5 pp [0,+8.8]**; not a demonstrated advantage.

Repeated run on identical verified CRACKS archives reproduced all 40 case decisions and aggregated metrics. V21 raw CSV may differ in ~1e-16 floating last bits across runners, but a case-level SHA256 of canonical *fully label-preserving* serialized numeric rows rounded to 1e-10 remains **ac4b588f865a9422c8c6d304f6bb49c770b24cfa52b3ca0aab6067f24204d309**. All 92 V21 class labels, candidate coordinates, choices and risk-threshold metrics are identical across both runs. V22 source and tests are in `scripts/cracks_real_mirrored_v22.py` and `tests/test_cracks_real_mirrored_v22.py`.

## Scientific admissibility

**What is supported:** under artificially equal geometry, actual observed seismic pixel evidence identifies the original expert-annotated continuation more often than a distant-seismic sham or deterministic geometric tie policy. This is evidence of **available image information** for *conditional* path verification.

**Not supported:** geological physical fault-instance correspondence of the mirrored endpoint; general false-bridge risk; an ANZA, SBPP or learned LIRA end-to-end score; resilience to missing seismic-image pixels; inter-volume generalization; that 2.5D neighbors beat the central image (they do not). V21/V22 were designed after inspecting earlier V17–V20 outcomes and are explicitly exploratory, with correlated cases and unadjusted CIs.

**Publication recommendation:** discuss V21 briefly to establish the need for genuine hard negatives and an adequately sized risk calibration; use V22 as a controlled observational argument for consulting image evidence under geometric non-identifiability. Keep the missing-checkpoint and naturally labeled LINK/REJECT/AMBIGUOUS constraints prominent. No new model superiority or natural E2E claim.
