# Original ANZA Phase-0 stop

The independent Original-ANZA forensic cycle ended at Phase 0 with
`STOP_OPERATOR_DEFINITION_MISMATCH`. The frozen legacy checkpoints use the
unchanged `models/azconv.py` implementation: categorical softmax memberships,
pair-symmetric doubled-angle local geometry, pair-averaged scales, raw
`mu_center * mu_neighbor * G * valid` interactions, global normalization over
modes and valid offsets, and a learned pointwise mix of per-mode aggregates.
This is not the literal independent-fuzzy directed interaction required by the
Research Task Packet.

The secondary status is `STOP_NO_INDEPENDENT_CONFIRM_SPLIT`: 393 of 396 CRACKS
images were used for segmentation training, while unseen sections 49, 73, and
385 have no crowd annotation files. Do not manufacture independence from edges
inside trained-on sections.

Under this packet, do not add read-only instrumentation, run S0-S4 confirm,
train an affinity model, or access expert data after the mismatch. The exact
evidence and validator are under `results/original_anza_forensics/phase0/`.
