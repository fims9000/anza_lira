# S21–S23. Low-label Graph-LIRA for bubble-chamber track continuation

**Date:** 25 October 2026. **Scope:** expert measured visualMetro coordinates from film `070` (1,073 tracks, 145 complete frames), artificially hidden interior expert measurements, plus **unlabelled** real H50Z image proposals. **Do not interpret these results as accuracy on real images.**

## Scientific question

Can a small number of verified `same_track` / `different_track` *pair* decisions teach Graph-LIRA to rank continuation candidates through long missing stretches, while preventing unsafe false joins? Does explicitly training on multiple synthetic gap lengths help more than adding graph-context features alone?

We distinguish: (a) ranking/recall on artificially interrupted **real human measured trajectories**, (b) original-image segmentation requiring expert pixel masks (currently unknown), and (c) physical three-view/B-field reconstruction (unavailable). Neither (b) nor (c) has been measured here.

## Inputs and withheld information

- `1mH2Data.7z` was safely extracted and parsed into 145 complete visualMetro frame manifests; verified source SHA-256 of audited manifest: `bd8e28247c5d178ae4f2f5a86215cbf5626f3ad86560928f4521093ce9188dea`.
- Source track identities *only* supply simulated expert answers to selected training or calibration **pair queries**. Unreviewed candidate labels remain unknown to both training and calibration, even though offline tests later reveal them to compute metrics.
- All three projections of any source frame are assigned to **one train, calibration, or held-out test partition**. The three test partitions are different shuffles of **the same underlying film**, not independent physical replicates.
- Frangi proposals on 7 H50Z crops (two real BMPs in one run) have **zero** confirmed true line/artifact or same/different expert responses. They cannot be treated as annotated test data.

## S21: 160 / 320 / 640 pair reviews, three withheld-frame seeds

Graph-context features are computed from candidate competitor scores, candidate degree and relative geometric ranking **without using track truth**. Models: basic 14D axial-geometric logistic regression; 24D context logistic regression; and a 24D context hist-gradient-boosting classifier. **This is a Graph-LIRA-like geometry classifier, not a newly trained original AZConv feature extractor.** All three share the SAME queried training pairs for fair comparisons.

- Training gap lengths: 1 and 3 hidden measured clicks; calibration: 3; held-out test: 1, 3 and 5.
- Label review simulation uses 160, 320 or 640 queried train pairs, **plus 160 separate queried calibration pairs per split**. Validation geometry includes unreviewed candidates in the competition graph, but calibration sees track truth ONLY for the 160 reviewed pairs.
- Probability acceptance uses mutually best left/right links and a fixed 0.05 margin plus train-observed scale support. Threshold is selected for **zero observed false joins on the 160 calibration reviews**. This limited sample cannot guarantee a small true physical error rate.

**Test with 5 hidden expert clicks:**

| Human train pair reviews | Context HGB true joins / available | Context HGB false joins | Basic geometric logistic true joins | Basic false joins |
|---:|---:|---:|---:|---:|
| 160 | 4 / 847 | 0 | 213 / 847 | 1 |
| 320 | 267 / 847 | 5 | 3 / 847 | 0 |
| 640 | 603 / 847 | 26 | 0 / 847 | 1 |

The S21 result is fragile. Increasing the training budget from 320 to 640 helped recall but allowed **26 erroneous links** on the same three-split test. It is **not safe for unattended physical merging**. Do not select a model based on its S21 test outcomes and then report those outcomes as independent confirmation.

## S22: a new preregistered multi-gap curriculum, new frame partitions

We froze split seeds `20261027/28/29`, model hyperparameters, train label counts (320 or 640), an additional 160 calibration pair reviews, validation lengths 3/5, and test lengths 3/5 **before viewing these S22 test outcomes**. Compare training on short gaps (1,3) with training on a mixture (1,3,5). All hold-out constraints remain.

| Human training reviews | HGB graph-context, short-gap train: true/false joins on 5-gap test | HGB graph-context, mixed-gap train: true/false joins | HGB ordinary geometric features, mixed-gap train: true/false joins |
|---:|---:|---:|---:|
| 320 | 117 / 1 | **323 / 1** | **456 / 1** |
| 640 | 369 / 4 | **554 / 3** | 421 / 0 |

Each row aggregates **876 available true candidate continuations** across the three test splits, with nearly complete true-candidate generation. Mixed-gap training increases S22 graph-context recall from 13.4% to 36.9% at 320 training queries, and from 42.1% to 63.2% at 640, but the model still makes nonzero wrong links. Conventional geometry-only HGB remains a **strong comparator**: at 320 reviews it recovered **456 of 876** with 1 false join, outperforming the graph-context variant. At 640 the graph-context model recovered more but also made more wrong joins (3 vs 0).

**No statistically independent physics run was available.** Confidence intervals within one frame partition or across overlapping splits do not provide independent-run validation. The S21 and S22 numbers must not be directly subtracted as if they were matched holdouts; only S22's within-seed short vs mixed comparisons use the same held-out scenes.

## S23: optional HUMAN review priority for real H50Z

- The original H50Z `S19_review_package` is loaded with original image/photo/proposal SHA checks; no new real-photo truth is created.
- For 23 candidate pairs across 7 source crops, observed 2D polyline geometry is mapped to the visualMetro-trained context HGB **only for alternate review ordering**. The original S20 expert workbench export contract, candidate IDs, canonical JSON structure and all `UNKNOWN` decisions remain unchanged.
- All 23 pair scores are **uncalibrated cross-domain ranks**; none may authorize a merge or be presented as a particle probability. Local fit geometry differs from sparse visualMetro clicks, and a scale-support pass is NOT evidence of matched imaging conditions.
- The private `S23_optional_workbench.html` includes real H50Z crop bytes, so it is distributed **only to the user**, not committed to GitHub. `S23_graph_ranker_geometry_only.joblib` is an exploratory serialized scikit-learn model; only load trusted, locally generated joblib artifacts.
- The original S20 workbench remains a useful non-learned review baseline and is **not** overwritten.

## Reproduction (private user data outside repository)

```bash
python scripts/research/bubble_chamber/s21_run_safe_graph.py --measurements /private/1mH2Data/S1 --out /tmp/s21.json --seeds 20261024 20261025 20261026 --budgets 160 320 640 --cal-reviews 160
python scripts/research/bubble_chamber/s22_gap_curriculum.py --measurements /private/1mH2Data/S1 --out /tmp/s22.json
python -m pytest -q tests/test_s21_selective_graph.py tests/test_s22_curriculum.py tests/test_s23_review_rank.py
```

## Next scientific acceptance gate

1. Expert confirms a few true track/artifact scribbles and true same/different fragments on at least **two distinct real photographs**. This is already supported by S19/S20. **Never generate missing human labels with a model.**
2. Use exactly the same reviewed pixels/pairs and photo-disjoint splits for ANZA and U-Net, with and without any optional unsupervised pretraining. Measure real label-conditioned segmentation quality and **false physical track merges per image**.
3. Collect an **independent physical run or filling**, with calibration for all three stereo views if physical momentum/trajectory parameters are to be inferred. Until such data exist, every proposed physical join stays `PROVISIONAL_REVIEW_REQUIRED`.
