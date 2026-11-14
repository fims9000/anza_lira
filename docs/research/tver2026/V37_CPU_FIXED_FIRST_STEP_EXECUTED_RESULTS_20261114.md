# V37 executed: CPU first-operation fingerprint DOES reproduce across different AMD processor families

**14 November 2026 — GitHub Actions [43612792459](https://github.com/fims9000/anza_lira/actions/runs/43612792459), SUCCESS.** Frozen protocol: [V37 CPU probe](./V37_CPU_PROBE_PREDECLARED_PROTOCOL_20261114.md). The official Thebe TEST7 remained unopened; V37 does **not download any Thebe data**, only a deterministic 64×64 synthetic tensor.

Two independent runner jobs executed the original ANZA-v1 (14,194 parameters) and U-Net (14,481 parameters) code with source seeds 20263311/20263312, torch 2.5.1+cpu, one CPU thread, MKLDNN disabled and deterministic algorithms enabled. Runners are **not identical hardware**:
- Runner A: AMD EPYC 9V45, family 26, AVX2/AVX512F/AVX512BW/FMA available.
- Runner B: AMD EPYC 7763, family 25, AVX2/FMA available, **no AVX512F** in reported flags.

The independent source auditor reports **V37_CPU_PROBE_BITWISE_CROSS_RUN_PASS** with **no mismatches**:

| Stage of exact model arithmetic | ANZA-v1 | U-Net |
|---|---|---|
| seeded original initial parameter tensor state SHA256 | identical | identical |
| first forward logits SHA256 | identical | identical |
| first backward parameter gradients SHA256 | identical | identical |
| first AdamW update parameter tensor state SHA256 | identical | identical |
| first loss scalar | identical | identical |

Synthetic fixed inputs, labels, torch/NumPy and deterministic flags also have equal SHA/values across runners. Thus two different GitHub CPU families **CAN** reproduce this particular initialization/forward/backward/single-update sequence. This **rules out a universal claim** that every pair of runners differs at first step because of AVX512 or initialization.

However V36 full **600 real Thebe TRAIN patches** showed two later separate jobs (43612287488 and 43612301607) agreeing with each other but disagreeing with original V35 job (43603004021), despite matching recorded patch raw SHA and numerical settings. V37 does NOT explain that difference: it uses one different synthetic 64×64 input and one training step instead of 600 distinct real patches, and the 2D mask loss is simpler (single BCE, not real weighted BCE+Dice+grad clip). Need isolate first divergent real TRAIN step/state, fully recorded CPU and runtime dispatch and the exact original real weighted objective on the *same real TRAIN patch bytes* before any causal claim about hardware kernels or compiled runtime. Neither model architecture nor optimizer has been proven the source of the mismatch.

**Scientific boundary**: no evidence of stable ANZA superiority over U-Net on heldout geology, no successful accepted LIRA relation under V34 CAL, no historical ANZA/P0 checkpoint recovery and NO external Thebe TEST7. V37 validates only a fixed small CPU arithmetic sequence.
