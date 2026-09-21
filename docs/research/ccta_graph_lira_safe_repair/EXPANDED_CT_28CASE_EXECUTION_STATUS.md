# Execution status — expanded matched CCTA runner

Date: 2026-09-21

Branch: `research/ccta-graph-lira-safe-repair`

## Verification completed in the current execution environment

The prepared `GraphLIRA_expand_radial_runner.zip` was unpacked and checked.

- `extract_radial_features_z04.py` passes `python -m py_compile`.
- frozen relation plan rows: **1360**;
- patients: **28**;
- train patients: **17**;
- validation patients: **5**;
- test patients: **6**;
- split rows: train **758**, validation **268**, test **334**;
- positive/negative counts are exactly balanced within every split;
- maximum number of split labels per patient = **1**.

The runner was then launched with the mounted data available to ChatGPT.

It correctly initialized:

```text
graphlira-z04-radial-expansion-v1
patients: 28 [953, 954, 955, 956, 957, 958, 959, 960, 961, 963, 964, 965, 966, 967, 969, 970, 971, 972, 973, 974, 975, 976, 977, 979, 980, 982, 983, 984]
```

and stopped at the raw archive gate:

```text
FileNotFoundError: /mnt/data/801-1000.z04
```

This environment currently contains the runner, plans, ImageCAS-X research bundle and earlier image-only CT files, but **not** the original Kaggle `801-1000.z04` + final `801-1000.zip` split archive required for the 28-patient matched-CCTA extraction.

No scientific metric was generated or inferred beyond the available data.

The next executable action is therefore exact and singular: make those two raw split-archive files available, then rerun the already-frozen runner without changing thresholds, patient IDs or model settings.
