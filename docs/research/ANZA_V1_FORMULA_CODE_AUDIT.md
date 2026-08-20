# ANZA v1 formula/code audit

Status: `V1_FORMULA_CODE_AUDIT_PASS_CLEAN_ANZA_REQUIRED`

Published contract: `w_r(p,q)=mu_r(p) mu_r(q) G_r(p,q)` followed by
normalization of positive interaction weights over rules and valid neighbors.

| Property | Legacy v1 | CleanANZA | Verdict |
|---|---|---|---|
| Membership activation | softmax across modes | independent sigmoid | legacy mismatch; clean match |
| Multiple memberships may exceed 0.5 | no, except degenerate two-mode boundary | yes | clean match |
| Pair weight | center mu x neighbor mu x geometry | unchanged | match |
| Normalization | global over rule and neighbor | unchanged | match |
| Axial theta equivalence | doubled-angle local geometry | inherited unchanged | match |
| Repeated membership attenuation | absent beyond pair endpoints | absent | match |
| Gaussian literal 1/2 | absent | inherited | scale-equivalent, not literal |

The legacy source remains unchanged at SHA256 `d0a5e9ac03d01ffa8b98e802921a5d876b48e91da8e6d582235b92abecb76197`. Its
scientifically material mismatch is categorical softmax competition. CleanANZA
is isolated in `/home/lebedeffson/Code/anza_lira/models/azconv_clean.py` and only changes membership activation; the
positive normalized ANZA aggregation is otherwise reused.

The missing literal Gaussian factor `1/2` can be absorbed into learned sigma,
so it is recorded as a parameterization mismatch rather than silently called
equation-identical.
