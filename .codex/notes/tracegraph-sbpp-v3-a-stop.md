# TraceGraph SBPP V3-A calibration stop

`TRACEGRAPH_RELATION_V2` passed its semantic validator with 20 explicit strata,
name-defined relation polarity, disjoint calibration/development/confirm seeds,
and no latent target fields in the public model input.  The frozen ANZA-KIR R0
checkpoint and threshold 0.35 were unchanged.

The zero-training SBPP calibration sweep used 3840 independent scenes: 2688
positive and 1152 NONE.  Every allowed `tau_micro` value (0.20, 0.25, 0.30,
0.35) produced the same BranchCandidateRecall@12, 2593/2688 = 0.964658.  This
missed the frozen 0.97 calibration gate by 15 positive cases.  Candidate burden
was safe on calibration (median 1, p95 5, mean 1.887), and the Wilson 95%
interval for recall was 0.956988--0.971001.  The identical sweep is expected
from the implemented V3-A constraint: micro-branches are candidate-only short
segments of the frozen hard graph, whose pixels already exceed 0.35.

The fail-closed status is `STOP_SBPP_CALIBRATION_COVERAGE_FAIL`.  Development
inference and metrics, P0/P1/P2, Transformer, path, confirm evaluation, CRACKS,
and expert data remained unopened.  V3-B soft-ridge extraction was explicitly
not authorized by V3-A and requires a separate frozen protocol if pursued.
