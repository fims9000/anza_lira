# ANZA-2 Phase 3C-A membership root cause

The frozen no-training F0-F9 component replacement audit passed with
`RC1 ROOT_CAUSE_MEMBERSHIP_LEARNING`. Phase 2B was reproduced exactly. Supplying
oracle memberships with learned orientation/scale/h restores zero parallel
false bridges and TPR `0.5430` at FPR <= `0.05`; supplying learned memberships
with oracle geometry leaves false bridge at `1.0` and TPR `0.0111`.

The learned field has active-mode recall `0.0030`, leaves `0.9948` of target
pixels with every membership inactive, and collapses to at most one active mode
at `0.9986` of crossing pixels. Orientation q90 error remains `0.1068` radians
and derived along/perpendicular geometry ratio remains `6.156`, so do not
replace orientation/scale/h or fusion before repairing membership learning.

Exactly one bounded RC1 membership repair on synthetic development data is
authorized next. Phase 3C-A itself performed no training and did not open
confirm, CRACKS, or expert data. Freeze the repair before any new confirm.
