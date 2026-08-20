# ANZA operator forensics

Status: `STOP_OPERATOR_DEFINITION_MISMATCH`

## Exact frozen operator

| Item | Frozen legacy implementation | Verdict |
|---|---|---|
| Membership | `softmax(logits / temperature, dim=mode)`, four modes | Material mismatch with independent fuzzy-degree contract |
| Orientation | per-mode raw angle; center and neighbor combined with doubled-angle axial mean | Axial, but not the directed center-only packet formula |
| Scales | positive local base/hyper fields; pair-averaged before `sigma_parallel/perpendicular` | Actual code documented |
| Geometry | `exp(-d_parallel^2/sigma_parallel^2-d_perpendicular^2/sigma_perpendicular^2)` | Missing literal `1/2`; scale-equivalent, not literal |
| Raw interaction | `mu_center * mu_neighbor * G * valid` | Matches pair-product structure |
| Normalization | global over four modes and valid 3x3 offsets per destination | Matches current code |
| Mode fusion | aggregate per mode, concatenate, learned 1x1 pointwise mix | `W=sum_r w_r` is reconstructable but not the standalone tensor consumed by output |

Runtime reconstruction error: `0.000e+00`.  
Membership sum error: `1.192e-07`.  
Normalized interaction sum error: `2.384e-07`.

The source remains unchanged at `d0a5e9ac03d01ffa8b98e802921a5d876b48e91da8e6d582235b92abecb76197` and the frozen T1 checkpoints load this legacy operator. CleanANZA is a separate sigmoid-membership implementation and is not substituted here.

## Stop decision

The packet requires an immediate stop on a material paper/code definition mismatch. Therefore read-only instrumentation, S0-S4 confirm scoring, learned affinity, and training were not run.

There is also no legal confirm split: the only segmentation-unseen images are sections `49`, `73`, and `385`, and each has zero crowd annotation files. Statistical independence cannot be manufactured from edges inside already trained-on sections.

- Training performed: no
- Expert accessed: no
- Confirm performed: no
- Next phase allowed: no
