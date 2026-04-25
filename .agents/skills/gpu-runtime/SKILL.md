---
name: gpu-runtime
description: Use for Windows/CUDA runtime behavior, GPU utilization/debugging, long training, background downloads, and performance checks in this repo.
---

# GPU Runtime Rules For `anza_lira`

Use this skill when:
- GPU is not utilized or runs are unexpectedly slow;
- CUDA/device behavior needs debugging;
- planning local GPU smoke runs or longer training;
- managing background dataset downloads or long-running jobs.

## Known Local Environments

CPU/test environment:

```powershell
.\.venv\Scripts\python.exe
```

GPU/training environment:

```powershell
& 'C:\ProgramData\anaconda3\envs\mcda-xai\python.exe'
```

Known good GPU environment:
- torch reports CUDA available in `mcda-xai`;
- GPU: NVIDIA GeForce RTX 5070 Ti.

Do not use the local `.venv` for CUDA training unless CUDA availability is re-verified.

## Quick Environment Checks

```powershell
& 'C:\ProgramData\anaconda3\envs\mcda-xai\python.exe' -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
& 'C:\ProgramData\anaconda3\envs\mcda-xai\python.exe' -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
nvidia-smi
```

## Fast Validation Ladder

1. CPU tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

2. Small GPU segmentation smoke:

```powershell
& 'C:\ProgramData\anaconda3\envs\mcda-xai\python.exe' train.py --config configs\drive_smoke.yaml --run-name debug_drive_gpu_smoke
```

3. GIS smoke:

```powershell
& 'C:\ProgramData\anaconda3\envs\mcda-xai\python.exe' train.py --config configs\gis_roads_smoke.yaml --run-name gis_roads_gpu_smoke
```

4. Full or benchmark configs only after smoke passes:

```powershell
& 'C:\ProgramData\anaconda3\envs\mcda-xai\python.exe' train.py --config configs\global_roads_benchmark.yaml --run-name global_roads_benchmark_s42
```

## Runtime Guardrails

- Start with small configs before long runs.
- Keep run outputs in dedicated `results/<run_name>` directories.
- If GPU OOM appears, reduce image size, patch size, batch size, widths, or rule count before broad code edits.
- If GPU is unexpectedly idle, verify CUDA availability and the actual Python interpreter.
- Report measured timing from `metrics.json`; do not assume speedups.
- AZ variants can be much slower than baseline. Compare speed and GMACs along with Dice.

## Background Jobs

For long downloads/runs:
- redirect logs to `logs/`;
- report PID and log path;
- check liveness with `Get-Process -Id <PID>`;
- check dataset growth by file count and size;
- do not kill a live user-needed process unless it is clearly stuck/broken or the user asks.

Global-Scale road download command:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_global_roads_from_hf.py --splits train val in-domain-test out_of_domain
```

Expected segmentation-only target:
- 8892 files;
- 4431 tiles;
- tens of GB.
