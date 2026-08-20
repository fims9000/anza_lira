# ANZA-2 Phase 3C-B RC1 bounded stop

The exact M-A (`lambda_bg=0.25`) and M-B (`lambda_bg=0.50`) membership-only
repairs ran for seed 41, five epochs each, from the frozen Phase-3B checkpoint.
Only `field.membership_head` changed; encoder, generic head, orientation,
scale, hyperbolicity, ANZA affinity, and beta remained bitwise unchanged.

Neither variant passed development. M-A/M-B membership recall was
`0.7562/0.7433`, X two-mode fraction was effectively zero, raw ANZA TPR at
FPR <= 0.05 was `0.0853/0.0872`, and parallel false bridge was `1.0` for both.
The train monitor improved, but the frozen train[0:256] inventory contains
only 128 positive gaps and 128 negative gaps and no X/T/Y/context cases. Record
that as an interpretation, not authorization to change the frozen protocol.

Status is `STOP_RC1_MEMBERSHIP_REPAIR_FAILED`. Do not add a third background
weight, more epochs, context samples, three-seed runs, beta fitting, confirm,
CRACKS, or expert access under this packet. The Phase-2B oracle-field positive
result remains valid but separate from learned-image evidence.
