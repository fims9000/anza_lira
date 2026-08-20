# ANZA-2 Phase-3 learned-field negative result

Phase 3 trained a common local edge task on three seeds. Its first frozen
development comparison failed and revealed a causal-confounding issue plus an
uncoupled orientation/membership loss. One bounded Phase-3B repair used a frozen
generic checkpoint, trained only the ANZA field and beta, coupled target-axis
coverage to active membership, and evaluated affinity OFF/ON in the same
checkpoint.

After correcting inclusive threshold ties without retraining, both conditions
have FPR `0.049973`. The three-seed TPR delta is `+0.00027406`, 95% CI
`[+0.00011431, +0.00043699]`, versus the frozen practical gate `+0.08`.
Therefore the result is `STOP_PHASE3B_LEARNED_AFFINITY_NO_GAIN`.

Do not open synthetic confirm, CRACKS Phase 4, or expert data under this
protocol. Phase-2B remains valid positive oracle-field evidence, but it must not
be described as learned-image or real-data improvement. Any continuation must
define a materially new hypothesis and output root before new evaluation.
