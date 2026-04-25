## Project Rules For Agents

This repository is `anza_lira`: a research codebase for AZConv / geometry-aware
segmentation models compared against U-Net-style baselines.

The project is currently used for:
- medical vessel segmentation: `DRIVE`, `CHASE_DB1`, `FIVES`;
- coronary angiography segmentation: `ARCADE syntax`, `ARCADE stenosis`;
- GIS / remote-sensing road segmentation: `gis_roads`, `global_roads`;
- paper/result analysis around AZ geometry, baseline trade-offs, and reproducibility.

## Coordination Rules

For non-trivial coding tasks, combine:
- `.agents/skills/karpathy-coding-rules/SKILL.md`
- the most specific local skill below.

When repository/runtime state is unclear, or before larger edits, also load:
- `.agents/skills/repo-bringup/SKILL.md`

When task touches metrics, result tables, model-vs-baseline comparisons, sweeps,
or manuscript claims, also load:
- `.agents/skills/benchmark-tradeoff-eval/SKILL.md`

When task touches CUDA, GPU speed, long training, background downloads, or
environment issues, also load:
- `.agents/skills/gpu-runtime/SKILL.md`

If multiple skills apply:
- follow the most specific skill for local decisions;
- keep the coding rules for code quality and validation;
- prefer small targeted changes over broad refactors.

## Execution Commitments

- Truthfulness first: never invent run results, metrics, downloads, or validations.
- Fix-to-done loop: localize issue, apply minimal fix, validate, document residual risk.
- Keep experiment settings explicit and rerunnable.
- Keep result claims conservative when AZ does not clearly beat baseline.
- Do not hide precision/recall trade-offs behind Dice-only language.
- Keep repository clean: remove transient debug/cache files after conclusions are captured.

## Current Runtime Facts

- Windows PowerShell workspace: `C:\Users\Comp1\SASHA\anza_lira`.
- Local CPU environment: `.venv\Scripts\python.exe`.
- GPU environment: `C:\ProgramData\anaconda3\envs\mcda-xai\python.exe`.
- Known good GPU: NVIDIA GeForce RTX 5070 Ti with CUDA available in `mcda-xai`.
- The local `.venv` is CPU-only; do not expect it to train on CUDA.

Useful checks:

```powershell
.\.venv\Scripts\python.exe -m pytest
& 'C:\ProgramData\anaconda3\envs\mcda-xai\python.exe' -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Engineering Checklist

Before finishing non-trivial work, verify:
- touched scope has a targeted test or smoke run;
- full tests are run when shared loaders/model code changed;
- metric/table/docs claims point to actual `metrics.json` artifacts;
- temporary files, partial downloads, and caches are cleaned when appropriate;
- background processes needed by the user are either completed or explicitly reported with PID/log path.

## Important Local Artifacts

- main training entrypoint: `train.py`
- shared loaders/models/metrics: `utils.py`
- AZConv implementation: `models/azconv.py`
- segmentation architectures: `models/segmentation.py`
- configs: `configs/*.yaml`
- saved metrics and reports: `results/**/metrics.json`, `results/*.md`
- GIS downloader: `scripts/fetch_global_roads_from_hf.py`
- external GIS reference repo: `external/samroadplus`
