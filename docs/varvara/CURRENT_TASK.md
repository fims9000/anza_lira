# CURRENT TASK — JUNCTION-LIRA -> CT-conditioned Graph-LIRA -> ANZA-LIRA

Date: 2026-10-06

This file supersedes the older JUNCTION generator instructions in this branch.

## Current status

The controlled degree-3 JUNCTION train/validation benchmark is now frozen for development.

Generator facts:

- train/val only;
- 143 degree-3 positive junctions;
- 1,666 retained negatives;
- candidate recall remains 100%;
- `segment_label` is audit metadata only and is not used to select negatives;
- junctions are separated into `competitive` and `isolated`;
- held-out test has not been generated/evaluated in the current JUNCTION pipeline.

The JUNCTION CT baseline is also implemented.

Candidate representation:

- raw CT tensor: `3 x 17 x 25`;
- 38 CT summary features per arm;
- symmetric arm aggregation by mean/min/max/std;
- total CT features: 152;
- geometry features: 9;
- geometry+CT features: 161;
- CT alignment checks pass for all 22 train/val patients.

Validation baseline headline:

| model | AUROC | top-1 all | top-1 competitive |
|---|---:|---:|---:|
| geometry | 0.9949 | 35/36 | 27/28 |
| CT | 0.9405 | 28/36 | 20/28 |
| geometry+CT | 0.9936 | 36/36 | 28/28 |

The important qualitative case is `966:left:58`: geometry ranks a false candidate above the true candidate, while CT supplies complementary evidence and the combined baseline restores the true candidate to rank 1.

Do not over-interpret this yet: it is one validation failure corrected by CT, not a final claim of general JUNCTION superiority.

## Why raw geometry+CT is not the final LIRA module

The 161-feature concatenation is a useful baseline, but its thresholded validation operating point is less safe than geometry-only:

- geometry: recall 0.9722, FPR 0.016, precision 0.814;
- raw geometry+CT: recall 0.9722, FPR 0.040, precision 0.636.

So ranking improves, but calibration/acceptance gets worse.

For this project that distinction matters: a false structural repair is more costly than abstention.

## Task A — JUNCTION-LIRA v0

Keep geometry and CT as separate evidence streams:

```text
geometry features -> geometry model -> score_g
CT summaries       -> CT model       -> score_ct

[score_g, score_ct]
        |
small fusion / relation head
        |
P(real JUNCTION candidate)
```

Requirements:

- train base scores for the fusion head must be patient-level OOF;
- validation must not be used to fit the fusion head;
- threshold selection is validation-only;
- test remains closed;
- keep the current raw geometry+CT model as a baseline, not as the canonical LIRA definition.

A reproducible development implementation is now in:

`scripts/research/ccta_graph_lira_safe_repair/train_junction_lira_fusion.py`

Diagnostic train/val result from the current feature table:

- validation AUROC about 0.99394;
- top-1 36/36;
- competitive top-1 28/28;
- at the selected low-FPR operating point: 35 TP / 4 FP;
- recall 0.9722;
- FPR 0.008;
- precision 0.8974.

This is development evidence only. It was designed after looking at validation and must not be presented as held-out evidence.

## Ambiguity groups

Keep the existing generator-distance subgroup, but rename it conceptually to:

`geometry_near`

because it is defined by `geometry_match_distance`, not by model uncertainty.

Add a second subgroup:

`geometry_model_ambiguous`

defined from patient-OOF geometry ranking margin:

```text
margin_g = score_g(true) - max(score_g(false))
```

The cutoff is derived from train OOF margins only.

This subgroup is the main place to inspect whether image evidence helps where the geometry model itself is uncertain.

## Task B — ANZA-LIRA image evidence ablation

After JUNCTION-LIRA v0 is reproducible, move directly into the clean local image-evidence experiment.

Use the same candidate tensor and do not change the surrounding protocol:

```text
hand-crafted radial CT
vs
compact CNN encoder
vs
compact ANZA encoder
```

Freeze across the comparison:

- patients/split;
- frozen JUNCTION generator;
- candidate IDs;
- geometry branch;
- fusion head interface;
- evaluation;
- threshold policy.

Only the image encoder changes.

The purpose is not to replace geometry. The question is whether a learned local image encoder, and specifically ANZA's directed/fuzzy local aggregation, produces better complementary evidence than radial summaries or a conventional compact CNN.

ANZA is therefore part of the intended research line, but it must enter as a controlled encoder ablation, not by changing the whole graph pipeline at once.

## Task C — scene relation type and Graph-LIRA

Once PAIR and JUNCTION image evidence are stable:

```text
PAIR_GEOMETRY + PAIR_CT/encoder
JUNCTION_GEOMETRY + JUNCTION_CT/encoder
                  |
       NONE / PAIR / JUNCTION / BOTH
                  |
            canonical Graph-LIRA
                  |
          tau = 0.85
     consistency = 0.60
                  |
           repair / abstain
                  |
            max-min path
```

Initially keep graph compatibility, tau and perturbation consistency frozen.

The repository still does not contain a clean canonical replacement for the lost historical `run_graph_lira_large_scale.py`. Do not rebuild hidden pickle state. The final integration should become a new explicit canonical runner/module with saved inputs, protocol and outputs.

## Test discipline

Do not open the six held-out patients while choosing:

- fusion architecture;
- CNN vs ANZA architecture;
- ambiguity definition;
- thresholds;
- calibration.

When train/validation choices are frozen, run the held-out test once.

## Immediate next deliverable

1. Reproduce `train_junction_lira_fusion.py` on the current feature table.
2. Save OOF train scores and validation source rankings.
3. Audit the geometry failure `966:left:58` plus the new `geometry_model_ambiguous` group.
4. Implement the compact CNN and compact ANZA encoders against the existing `3 x 17 x 25` tensors.
5. Compare radial / CNN / ANZA using the same LIRA fusion interface.
6. Freeze the selected image-evidence branch before any held-out test.
7. Then build the PAIR+JUNCTION relation-type head and canonical CT-conditioned Graph-LIRA.

Generator tuning is no longer the research target.
