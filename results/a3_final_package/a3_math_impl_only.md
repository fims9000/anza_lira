# Article 3 (Math + Implementation Only)

This version is prepared for public sharing without experimental scores.

## Scope

The work studies anisotropic fuzzy local convolution as a local aggregation mechanism for thin-structure segmentation, evaluated in a cross-domain setting (medical vessels and road networks).

## Mathematical core

For a center pixel `p`, neighbor `q` and local mode `r`, the local compatibility is:

`w_r(p, q) = mu_r(p) * mu_r(q) * kappa_r(q - p)`

where:
- `mu_r(.)` is fuzzy local membership,
- `kappa_r(.)` is anisotropic directional kernel.

Normalized local weights:

`w_tilde_r(p, q) = w_r(p, q) / (sum_{q in N(p)} sum_{m=1..R} w_m(p, q) + eps)`

Local aggregation:

`z(p) = sum_{q in N(p)} sum_{r=1..R} w_tilde_r(p, q) * V(q)`

Binary prediction is obtained by thresholding a probability map:

`M_hat(p) = 1 if y_hat(p) >= tau else 0`

## Geometric parameterization

The directional part is implemented with a fixed anisotropic metric inspired by hyperbolic expansion/contraction behavior:
- separate along/across sensitivity,
- finite metric conditioning,
- non-zero anisotropy gap.

This is an operator-level geometric prior, not a full dynamical-system construction.

## Model integration

The operator is integrated into a U-Net style encoder-decoder:
- baseline: standard local convolution blocks,
- proposed: AZ-enhanced local blocks in selected encoder stages,
- optional residual branch to keep stable optimization.

## Training protocol (implementation-level)

- binary segmentation objective with overlap-aware loss,
- validation-based threshold selection,
- matched baseline/proposed protocol per dataset,
- same split and preprocessing policy inside each comparison pair.

## Repository pointers

- Main draft with full results: `results/a3_final_package/a3.md`
- Article-ready draft text: `results/stdh2026_paper_draft_en.md`
