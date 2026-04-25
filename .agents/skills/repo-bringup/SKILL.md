---
name: repo-bringup
description: Use when repository state is unclear, after copied files arrive from another project, or before feature work needs a reliable baseline.
---

# Repository Bring-Up Rules For `anza_lira`

Use this skill when:
- a new session starts and repo state is unclear;
- files were copied from another project;
- imports, configs, data paths, or entrypoints may be stale;
- a clean working baseline is needed before model/data changes.

## Read First

1. Project overview:
- `README.md`
- `results/latest_results_summary_ru.md` when present
- `results/architecture_baseline_analysis_ru.md` when discussing AZ vs baseline

2. Main code paths:
- `train.py`
- `utils.py`
- `models/azconv.py`
- `models/segmentation.py`

3. Relevant configs:
- medical: `configs/drive*.yaml`, `configs/chase*.yaml`, `configs/fives*.yaml`
- angiography: `configs/arcade_*`
- GIS: `configs/gis_roads_smoke.yaml`, `configs/global_roads_*.yaml`

## Data Reality Check

Expected local datasets:
- `data/DRIVE`
- `data/CHASE_DB1`
- `data/FIVES`
- `data/ARCADE`
- `data/Roads_HF`
- `data/GlobalScaleRoad` while the full GIS dataset is downloading

Do not claim a dataset is complete without checking file counts.

Useful check:

```powershell
Get-ChildItem data -Directory
Get-ChildItem data\GlobalScaleRoad -Recurse -File | Group-Object { $_.FullName -replace '^.*GlobalScaleRoad\\([^\\]+).*$','$1' }
```

## Verification Ladder

Cheap to expensive:

1. Syntax/import scope through targeted tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_drive_pipeline.py::test_gis_roads_dataloaders_build_and_batch
```

2. Full CPU test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

3. Small GPU smoke:

```powershell
& 'C:\ProgramData\anaconda3\envs\mcda-xai\python.exe' train.py --config configs\drive_smoke.yaml --run-name debug_drive_gpu_smoke
```

4. Dataset-specific GPU smoke:

```powershell
& 'C:\ProgramData\anaconda3\envs\mcda-xai\python.exe' train.py --config configs\gis_roads_smoke.yaml --run-name gis_roads_gpu_smoke
& 'C:\ProgramData\anaconda3\envs\mcda-xai\python.exe' train.py --config configs\global_roads_smoke.yaml --run-name global_roads_gpu_smoke
```

5. Longer multiseed/benchmark runs only after targeted checks pass.

## Working Rules

- Do not start with broad refactors.
- If validation fails, localize before stacking fixes.
- Keep docs and result summaries aligned with executed `metrics.json`.
- Report reality only: no "green" status without command evidence.
- Keep generated artifacts in `results/`, `logs/`, or explicit data/download folders.

## Hygiene Rules

- Avoid temporary files in repository root.
- Remove `.pytest_cache` and `__pycache__` outside `.venv` after test-heavy work.
- Do not delete user data or downloaded datasets unless explicitly asked.
- Do not kill background downloads unless they are broken or the user asks.

## Done Criteria

Bring-up is done when:
- relevant tests/smoke runs are green;
- changed commands and paths are valid for this repo;
- remaining issues are documented clearly;
- background tasks are reported with PID/log path when still running.
