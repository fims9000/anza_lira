# ANZA-LIRA CRACKS Structural Stability V1.1 SS1.5 freeze

V1.1 is a pre-result amendment to the separate Structural Stability V1 line.
It does not rewrite parent SS0/SS1 or any historical ANZA/LIRA STOP. Parent and
old-STOP hashes remained byte-identical throughout SS1.5.

Normalization was recomputed from only the 220 frozen SS_TRAIN image sections:
mean `[0.8508687842, 0.7009342679, 0.8396038140]`, std
`[0.2396369329, 0.2645439530, 0.2419134349]`, and normalization payload SHA-256
`013b16cc61ee8e1bc34a3221c5e7c26576e7dde8b4955e51adc65cc45f008630`.
No calibration/development/confirm image contributes to these values.

Train-only nonexpert crowd geometry supervision exists on 3,432,186 of
39,326,100 train pixels (`0.08727502`) and on all 220 train sections. It uses
the frozen agreement field, structure-tensor tangent, coherence gates, and
identical `d*=0.35*kappa` for B2 and B3. Expert and evaluation labels were not
decoded.

The metric implementation uses SPD congruence, not raw-J similarity. B2 has
free `m` and determinant `exp(4m)`; B3 fixes `m=0` and determinant one. The
parent warp is output-to-input, so transport uses its inverse forward Jacobian,
area-normalized before `C'=Abar C Abar^T`. Numerical tests cover SPD,
determinant, axial invariance, rotation/scaling/shear transport, Jacobian
direction, log reconstruction, and finite gradients.

Fresh backbone initializations are frozen for seeds 41/42/43; B0-B3 load the
same backbone state within each seed. Historical H0 provenance is rejected.
Each seed has one shared 7,920-row pair manifest, 36 epochs, 1,980 planned
optimizer updates, and only severities 1/2. All 23,760 rows retain explicit
labels for the selected annotators. B2/B3 total parameter counts differ by
`0.00008490` relative, below the frozen 1% limit.

All 3,300 train-normalized perturbation cases were finite and deterministic;
warp/palette constraints passed and no performance metric was used. Final
validation: 29 targeted tests and the full repository suite (`874 passed, 1
skipped`), compileall, JSON parsing, and `git diff --check` all passed.

Status is `SS1_5_PRETRAINING_FREEZE_PASS`. No B0/B1/B2/B3 training,
development, confirm, LIRA, or expert evaluation was opened. The 12 planned
training jobs are authorized only as the next separate phase.
