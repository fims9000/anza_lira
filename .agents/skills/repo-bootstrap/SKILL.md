---
name: repo-bootstrap
description: Capture the GeoCrack repo, environment, runtime, RTK, and reproducibility baseline before experiments.
---

# Repository Bootstrap

Extend, do not replace, `repo-bringup`. Record Git commit/branch/remote, Python,
PyTorch/torchvision, CUDA/GPU, OS, installed packages, and RTK status under
`results/geocrack_study/`. Reuse existing environments. Do not delete user
changes or data, create an environment, or reinstall PyTorch without a proven
need. Record runtime incompatibilities explicitly.
