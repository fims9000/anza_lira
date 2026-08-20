# ANZA-LIRA Graph-Cut Intervention V2 final STOP

Graph-Cut V2 is a separate treatment-validity protocol and does not rewrite
either `STOP_LIRA_REAL_GAP_DATA_INSUFFICIENT` or the V1 fixed-3-pixel
`STOP_LIRA_INTERVENTION_CANDIDATE`.  Its fresh split and placement namespace
were frozen before manipulation counts.  Confirm sections 347--400 remained
hash-only; expert annotations were not accessed.

The benchmark searched only radii 3/5/7/9/11/13/15 and required a minimal
8-connected cut at frozen support threshold 0.12, surviving source/destination
context, and at most 5% overlap with another local trace from the same
annotator/section.

Calibration had 1,391 pre-treatment eligible traces and development had 1,740.
No case survived all manipulation-validity rules, so retention was 0 in both
splits versus the frozen 0.50 gate.  In development, 1,606 cases disconnected
only while destroying required context and 134 exceeded the collateral-trace
limit.  Minimal disconnect radii before exclusions were predominantly 9
(726 cases) and 11 (1,007 cases); there were only 5 at radius 7 and 2 at radius
13, with none at 3/5/15.

`TreatmentValidity` is undefined (`N/A`), not zero, because there are no
accepted benchmark cases.  The correct status is
`STOP_GRAPH_CUT_BENCH_TOO_SELECTIVE`.  Frozen SBPP candidate scoring was not
opened, so this is not an SBPP or P0 performance result.

P0, AUTO-LINK/REVIEW calibration, path, relation seeds, confirm, expert, and
new architectures were not run.  Per the frozen V2 boundary, this closes the
real-image continuation-development line.  Do not alter anchor bands, clip
tube end caps, relax collateral/context/retention gates, or create a V3 under
this research sequence.
