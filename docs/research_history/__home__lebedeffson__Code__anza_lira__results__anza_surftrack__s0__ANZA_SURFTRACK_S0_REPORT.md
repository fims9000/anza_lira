# ANZA-LIRA SurfTrack V1 — S0 causal geometry

## Status

`STOP_ANOSOV_SURFTRACK_NO_CAUSAL_VALUE`

Zero-training geometry only. No seismic rendering, CNN, Thebe, CRACKS, or confirm data were opened.

## Observability

- Center-only AUROC: `0.497201` (required 0.45–0.55).
- Adjacent-history oracle Top1: `1.000000` (required >=0.85).

| Method | IID Top1 | IID switch | OOD Top1 | OOD switch | Survival@7 OOD |
|---|---:|---:|---:|---:|---:|
| G0_euclidean | 0.7662 | 0.6944 | 0.7102 | 0.7641 | 0.6440 |
| G1_local_reset | 0.7603 | 0.7051 | 0.7030 | 0.7907 | 0.6397 |
| G2_shear_compose | 0.7668 | 0.6991 | 0.7099 | 0.7795 | 0.6456 |
| G3_free_compose | 0.7560 | 0.7094 | 0.6971 | 0.7993 | 0.6322 |
| G4_anza_cocycle | 0.7660 | 0.6962 | 0.7095 | 0.7693 | 0.6440 |

## Train-only fitted transport

- `G1_local_reset`: `{'sigma_u': 0.25, 'sigma_s': 0.27807076907664324}`; `sigma_u` hit its frozen lower bound.
- `G2_shear_compose`: `{'sigma0': 0.25, 'q': 0.0011481992401772623, 'alpha': -0.003938769644930752}`; `sigma0` hit its lower bound and shear fitted near zero.
- `G3_free_compose`: `{'sigma0': 0.25, 'q': 0.025001262647653958, 'a': -0.3431541670185126, 'b': -0.17195625696994668}`; `sigma0` hit its lower bound.
- `G4_anza_cocycle`: `{'sigma0': 0.25, 'q': 0.001154986042855668, 'lambda': 0.0}`; both `sigma0` and `lambda` hit their lower bounds. The train-only optimum therefore disabled hyperbolicity before dev was opened.

## Paired OOD bootstrap

- G4−G1 Top1: `+0.006583`, 95% CI `[+0.004638, +0.008547]`; below the required `+0.08`.
- G1−G4 switch benefit: `+0.021400`, 95% CI `[+0.017700, +0.025100]`; G4/G1 switch ratio `0.972935`, above the required `0.70`.
- G4−G2 Top1: `−0.000326`, 95% CI `[−0.001181, +0.000564]`.
- G2−G4 switch benefit: `+0.010200`, 95% CI `[+0.007000, +0.013500]`; G4/G2 switch ratio `0.986915`, above the required `0.80`.
- G4−G3 Top1: `+0.012410`, 95% CI `[+0.010186, +0.014693]`; below the required OOD `+0.03`.

## Frozen gates

- `composition_G4_vs_G1`: `False`
- `hyperbolic_G4_vs_G2`: `False`
- `flexible_control_G4_vs_G3`: `False`
- `per_stratum_3_of_5`: `False`
- `top1_delta_G4_G1_ood`: `0.006583066239316149`
- `switch_ratio_G4_G1_ood`: `0.9729353737194891`
- `top1_delta_G4_G2_ood`: `-0.00032628205128204524`
- `switch_ratio_G4_G2_ood`: `0.986914688903143`
- `top1_delta_G4_G3_iid`: `0.009956196581196619`
- `top1_delta_G4_G3_ood`: `0.012410470085470027`
- `switch_ratio_G4_G3_ood`: `0.9624671587639184`
- `full_pass`: `False`

## Claim boundary

Only a full causal PASS could open learned SurfTrack. Every other status keeps S1, confirm, real data, and all Anosov-specific repairs locked.
