# ANZA-LIRA SurfTrack S0 final Anosov-specific STOP

SurfTrack S0 was a zero-training causal geometry study. It froze five controls
and 50k/10k/10k/10k train/calibration/IID/OOD scene streams before results;
the 20k confirm stream remained hash-only. No seismic rendering, CNN, Thebe,
CRACKS, or confirm data were opened.

The benchmark observability gate passed: center-only matched geometry AUROC was
0.497201 and the adjacent-history oracle Top1 was 1.0. The tracking problem was
therefore context-observable without being identifiable from the center slice.

Train-only maximum likelihood selected the following key parameters:

- LocalReset: sigma_u=0.25, sigma_s=0.278071.
- ShearCompose: sigma0=0.25, q=0.001148, alpha=-0.003939.
- FreeCompose: sigma0=0.25, q=0.025001, a=-0.343154, b=-0.171956.
- ANZA-Cocycle: sigma0=0.25, q=0.001155, lambda=0.0.

Thus the fitted ANZA prior disabled hyperbolicity before dev was opened. On OOD
scenes, ANZA Top1/switch were 0.709544/0.7693 versus LocalReset
0.702960/0.7907. The small Top1 delta +0.006583 had paired 95% CI
[+0.004638,+0.008547], but missed the predeclared +0.08 practical gate; the
switch ratio 0.972935 missed the <=0.70 gate. Against ShearCompose, Top1 delta
was -0.000326 with CI crossing zero. All five named per-stratum gates failed.

Frozen status: `STOP_ANOSOV_SURFTRACK_NO_CAUSAL_VALUE`. Per the strict protocol,
this closes the Anosov-specific seismic mechanism line. Do not add another
Anosov kernel, cocycle, entropy, Koopman, neural rescue, or reopen S1. A generic
sequential SurfTrack product would require a separate non-Anosov authorization.

Artifacts are under `results/anza_surftrack/s0/`; validator is PASS. Fifteen
targeted tests and the full 830-pass regression suite passed.
