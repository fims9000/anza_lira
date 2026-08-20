# ANZA-2 mathematical specification

Status: frozen Phase-1 operator contract. This document defines `ANZA2-HFR`;
it does not reinterpret or modify LegacyANZA in `models/azconv.py`.

## Field

For each local mode `r`, the field predicts independent membership, doubled
axis, base scale, and hyperbolicity:

```text
mu_r = sigmoid(a_r)
(c_r, s_r) = normalize(u_r, v_r) = (cos(2 theta_r), sin(2 theta_r))
ell_r = ell_min + softplus(b_r)
h_r = h_max sigmoid(g_r)
sigma_parallel = ell_r exp(h_r)
sigma_perpendicular = ell_r exp(-h_r)
```

The memberships are not normalized across modes. The doubled-angle
representation makes `theta` and `theta + pi` identical. The reciprocal shape
transform has eigenvalues `exp(h)` and `exp(-h)`, hence determinant one. This is
an Anosov-inspired local parameterization, not a claim that the layer is an
Anosov dynamical system.

## Directed geometry

For `d=(dx,dy)=q-p`:

```text
lambda_parallel      = ell^-2 exp(-2h)
lambda_perpendicular = ell^-2 exp(+2h)
m0 = (lambda_parallel + lambda_perpendicular) / 2
m1 = (lambda_parallel - lambda_perpendicular) / 2

Q_r(p,d) = m0 (dx^2 + dy^2)
         + m1 [c_r (dx^2 - dy^2) + 2 s_r dx dy]

G_r(p -> q) = exp(-Q_r(p,q-p) / 2)
```

No angle is reconstructed and no center/neighbor axis is averaged.

## Mode-permutation-invariant step support

Mode indices are local hypotheses, not global instance identities:

```text
D(p -> q) = max_r [mu_r(p) G_r(p -> q)]
```

The max is a fuzzy OR: any active local mode may support the actual
displacement.

## Absolute structural affinity

```text
A_ANZA(p,q) = sqrt(D(p -> q) D(q -> p))
```

This relation is symmetric, lies in `[0,1]`, and is not normalized over a
neighborhood. It therefore retains absolute edge strength for downstream graph
algebra.

## Local aggregation

For center mode `r`:

```text
T_r(p,q) = G_r(p -> q) D(q -> p)
Z_r(p) = tau0 + sum_q T_r(p,q)
alpha_r0 = tau0 / Z_r
alpha_rq = T_r(p,q) / Z_r

h_r(p) = mu_r(p) [alpha_r0 V(p) + sum_q alpha_rq V(q)]
```

Normalization is within each mode and includes explicit self mass. One mode
never renormalizes another. If all neighbors are unsupported, the mode falls
back to `mu_r V(p)`.

Mode outputs are concatenated and projected. The block output is:

```text
y = x_proj + gamma P(concat(h_1,...,h_R)),  gamma_init = 0.
```

## Learned affinity causal control

The generic image head remains common to both variants:

```text
s_comb = s_generic + beta logit(A_ANZA)
A_comb = sigmoid(s_comb)
beta = softplus(beta_raw) >= 0
```

`beta` is initialized numerically at zero. At that operating point, the model
is the generic affinity baseline. Incremental value must be established against
generic plus simple geometry with the same reachability algorithm.

## Downstream algebra and claim boundary

Widest-path connectivity is:

```text
C(u,v) = max_path min_edge A(edge)
```

It runs only in a fixed support domain. Max-min/widest path is prior art and is
not an ANZA-2 novelty claim. The testable ANZA contribution is the structural
edge evidence supplied to an otherwise identical graph/reachability system.

## Frozen Phase-1 defaults

- modes: `R=4` (`R=2` is the only mode-count ablation);
- `ell_min=0.25`;
- `h_max=1.25` (`0.75` is the bounded development ablation);
- `tau0=1.0` (`0.5` is the bounded development ablation);
- local graph: 8-neighborhood;
- standard Gaussian factor: literal `1/2`;
- expert supervision: forbidden.
