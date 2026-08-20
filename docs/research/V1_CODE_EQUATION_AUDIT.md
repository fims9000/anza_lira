# AZConv2d v1 code-equation audit

Status: `V1_PARTIAL_EQUATION_MATCH_C1_REQUIRED`

The audited source is the unchanged `models/azconv.py` with SHA256 `d0a5e9ac03d01ffa8b98e802921a5d876b48e91da8e6d582235b92abecb76197`.

## Findings

| Contract | Code finding | Verdict |
|---|---|---|
| `mu_r` is an independent fuzzy degree | `F.softmax(..., dim=1)` forces `sum_r mu_r = 1` | **MISMATCH** |
| `w_r = mu_r(p) mu_r(q) G_r` | Code uses `mu_center * mu_un * kern` | MATCH |
| normalize over neighbor and rule | Default global path sums dimensions `(rule, neighbor)` | MATCH |
| no extra membership attenuation | Membership occurs only at the two pair endpoints | MATCH |
| axial `theta == theta + pi` | Local pair geometry uses `cos(2 theta), sin(2 theta)` | MATCH |
| Gaussian exponent includes literal `1/2` | Code uses `exp(-du^2/sigma_u^2-ds^2/sigma_s^2)` | PARAMETERIZATION MISMATCH |
| positive finite scales | softplus base and bounded hyperbolicity produce positive finite scales for finite parameters | MATCH |
| isotropic limit ignores direction | `use_anisotropy=False` uses radial squared distance | MATCH |

The missing Gaussian `1/2` is scale-equivalent after `sigma -> sigma/sqrt(2)`, but the parameterization is not literally the supplied equation. The scientifically material mismatch is categorical softmax membership. Therefore C1 (`v1_fuzzy_independent`) is required as a separate ablation; the frozen v1 code is not edited.
