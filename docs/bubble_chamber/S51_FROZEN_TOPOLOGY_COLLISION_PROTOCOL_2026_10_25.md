# S51 preregistration — 2D bridge/fragment collision diagnostic

Frozen before accessing S51 synthetic/native collision outcomes. Branch:
`research/bubble-chamber-graph-lira`, no manual annotation.

## Hypothesis

A Hermite continuation that overlaps a *third* detected line in its central
60% is topologically ambiguous: may cross another particle, form a vertex,
follow a scratch or merely run alongside a duplicate machine candidate.
This is an **advisory**, NEVER a hard false-join classifier or independent GT.

## Fixed method

For each S39 original candidate bridge sample 101 linearly interpolated points
along its original Hermite proposal, retain fractional indices 20..80, and
compute exact point-to-segment Euclidean distance to all remaining candidate
centerlines, excluding both endpoint nodes. Flag if distance <=3 pixels. No
change to registered S40/S45/S48 guards or to source graph topology. All 182
native edges must remain in the output; all physical join authorizations false.

Synthetic check: S35 injected-trace scenes with original source-frame
train/cal/test split and the already frozen S50 length-matched 31-point
candidates. Use only SOURCE TEST frames from S37 (36), H50Z backgrounds, 40
scenes each direction and two seeds (20265101, 20265102 on photo 006;
20265111, 20265112 on photo 005). For all 2x3 synthetic candidates, compare
advisory incidence for known *injection-only* correct/false pairs. No labels,
thresholds or policy may be selected on the results.

## Decision and provenance

If collision incidence does not separate correct and false pairs reliably,
report negative result and keep warning solely for visual triage. 2-D line
crossing is permitted physical geometry and cannot prove a false join; no
spatial track identity or mass/3D relation may be claimed. Private source
coordinates, pixels and per-edge paths remain outside the public GitHub.
Keep the viewer and input JSON unchanged except for a reversible advisory
layer. Preserve original native 567 lines, 182 relations and S49 warnings.
