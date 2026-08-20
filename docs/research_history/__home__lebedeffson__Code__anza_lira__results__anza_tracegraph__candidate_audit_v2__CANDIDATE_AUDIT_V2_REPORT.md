# ANZA-TraceGraph Candidate Audit V2

Status: `CANDIDATE_AUDIT_V2_COMPLETE`

This is a zero-training forensic analysis of the exact frozen TG1 predictions. Radius 8/10 and aligned-gap values are diagnostics, not revised gates.

## Reproduction

- V1 Recall@6: `0.798828`
- V1 misses: `206`
- exact parent miss set: `True`
- distance bins: `{"gt_10_or_missing": 30, "gt_6_le_8": 120, "gt_8_le_10": 56, "le_6": 818}`
- misses with K=8 full: `143`

## Cause taxonomy

| Cause | Count |
|---|---:|
| B_correct_branch_eligible_but_dropped_by_topK | 1 |
| D_skeleton_connected_with_confidence_valley | 1 |
| A_correct_branch_port_in_topK_but_endpoint_shifted | 148 |
| C_branch_support_or_junction_but_no_admissible_port | 21 |
| E_correct_branch_absent_in_dense_prediction | 35 |

Operational A--D cases account for `171/206` (`83.010%`); E accounts for `35/206`. The taxonomy is defined by the frozen 3 px branch tube and 0.60 overlap rule, not claimed as annotation-independent ground truth.

## Coverage

| Rule | K | Branch recall | Endpoint@6 | Endpoint@8 | Endpoint@10 |
|---|---:|---:|---:|---:|---:|
| axial_v1 | 4 | 0.850586 | 0.798828 | 0.915039 | 0.970703 |
| axial_v1 | 8 | 0.909180 | 0.798828 | 0.916016 | 0.970703 |
| axial_v1 | 12 | 0.915039 | 0.798828 | 0.916016 | 0.970703 |
| axial_v1 | 16 | 0.915039 | 0.798828 | 0.916016 | 0.970703 |
| axial_v1 | 24 | 0.915039 | 0.798828 | 0.916016 | 0.970703 |
| axial_v1 | 32 | 0.915039 | 0.798828 | 0.916016 | 0.970703 |
| directed | 4 | 0.865234 | 0.792969 | 0.911133 | 0.967773 |
| directed | 8 | 0.913086 | 0.792969 | 0.911133 | 0.967773 |
| directed | 12 | 0.913086 | 0.792969 | 0.911133 | 0.967773 |
| directed | 16 | 0.913086 | 0.792969 | 0.911133 | 0.967773 |
| directed | 24 | 0.913086 | 0.792969 | 0.911133 | 0.967773 |
| directed | 32 | 0.913086 | 0.792969 | 0.911133 | 0.967773 |

## Port geometry

- mean axial eligible pool: `10.550`
- mean directed eligible pool: `5.350`
- mean away-facing ports removed: `5.200`
- smallest K reaching branch recall 0.95: `None`
- nearest correct-port error quantiles: `{"abs_longitudinal_error": {"0.5": 1.1244570956373248, "0.9": 12.410021022543575, "0.95": 17.597690634464936}, "abs_transverse_error": {"0.5": 0.45841980015120704, "0.9": 2.1801155791151903, "0.95": 3.113772182686034}, "total_error": {"0.5": 1.4078484161184182, "0.9": 12.535559773630014, "0.95": 17.668251252226444}}`

K expansion is not the main repair: axial branch recall rises only from 0.909180 at K=8 to 0.915039 at K=12 and then saturates. Directed ports halve the pool and slightly improve branch recall at small K, but reduce endpoint-radius recall; they are a useful pruning diagnostic, not a passed replacement.

Localization error is predominantly longitudinal along the correct branch rather than transverse to a neighboring branch, especially in the upper tail.

## Protocol mismatches

- aligned forced-gap Recall@6: `0.771484` versus V1 `0.798828`
- aligned branch recall@8: `0.908203`
- curvature split declared but implemented: `False`
- scene names without specialized construction: `curved, weak_branch, y_junction, t_junction, none, multiple_plausible`
- `none` positives / negatives: `64 / 64`

Aligning the forced cut to the generator endpoint does not rescue V1: Recall@6 falls by 0.027344 and branch recall is unchanged within 0.001. The mismatch is real, but it is not the dominant measured cause under this audit.

## Frozen conclusion

The evidence supports a port-localization/front-end repair, not a larger Transformer and not a blind K increase. Any next protocol should test soft or branch-aware ports while keeping P0 frozen, and must rebuild true X/T/Y/weak/multiple-plausible generators before attributing stratum-specific effects.

## Boundary

No prediction, threshold, model, split, training, confirm, CRACKS, expert, or path result was changed or opened. This audit ends at the A/B/C/D/E table.
