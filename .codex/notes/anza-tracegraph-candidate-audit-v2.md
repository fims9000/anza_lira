# ANZA-TraceGraph Candidate Audit V2

The zero-training audit completed with status `CANDIDATE_AUDIT_V2_COMPLETE`.
It reused the exact frozen ANZA-KIR R0 predictions, threshold 0.35, development
split, and parent TG1 candidate definition. The parent miss set was reproduced
exactly: Recall@6 0.798828, 206 misses, distance bins 818/120/56/30, and 143
misses with K=8 full.

The frozen operational A/B/C/D/E taxonomy of the 206 misses is:

- A, correct-branch port in top K but endpoint shifted: 148;
- B, correct branch eligible but removed by top K: 1;
- C, branch support or junction but no admissible port: 21;
- D, connected skeleton with confidence valley: 1;
- E, correct branch absent under the frozen support rule: 35.

Thus 171/206 misses fall in A--D under the declared 3 px tube / 0.60 overlap
rule, but the result does not support a blind K increase. Branch recall is
0.909180 at K=8, 0.915039 at K=12, and does not improve through K=32. Directed
ports reduce the mean pool from 10.55 to 5.35 and slightly improve branch recall
at small K, while slightly reducing endpoint-radius recall.

Nearest correct-port errors are primarily longitudinal: median absolute
longitudinal error 1.12 px versus transverse 0.46 px; at the 90th percentile
they are 12.41 versus 2.18 px. Aligning the forced cut with the generator
destination does not rescue V1: endpoint Recall@6 falls to 0.771484 and branch
recall@8 remains about 0.908.

The implementation audit also confirms that `curvature_split_radians=0.70` is
declared but unused. `curved`, `weak_branch`, `y_junction`, `t_junction`,
`none`, and `multiple_plausible` have no dedicated construction branch; the
`none` name contains 64 positive and 64 negative examples because polarity is
independent of scene name.

No model, prediction, threshold, split, checkpoint, training, confirm, CRACKS,
expert, or path result changed. Any later protocol should focus on branch-aware
or soft port localization while keeping P0 frozen, and should correct generator
semantics before making X/T/Y/weak/multiple-plausible claims.

