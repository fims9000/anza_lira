---
name: karpathy-coding-rules
description: Conservative coding behavior for non-trivial tasks in `anza_lira`. Use when modifying code, adding features, refactoring, debugging, or changing experiments.
---

# Coding Rules For `anza_lira`

## Core Behavior

1. Understand existing code first.
- Read neighboring files and reuse current abstractions.
- For loaders, check `utils.py` before adding new dataset code.
- For model changes, check `models/azconv.py` and `models/segmentation.py`.
- For training behavior, check `train.py`.

2. Prefer minimal changes.
- Make the smallest correct readable diff.
- Avoid broad refactors unless they materially improve correctness or maintainability.
- Keep medical, ARCADE, and GIS loaders compatible unless the task explicitly narrows scope.

3. Avoid unnecessary complexity.
- No new dependencies without strong need.
- Prefer straightforward code over clever code.
- For image/mask datasets, prefer structured loaders and explicit split handling over ad hoc path hacks.

4. Match project style.
- Use existing config-key patterns.
- Keep dataset aliases centralized near existing dataset constants.
- Keep metrics in `metrics.json` and summaries in `results/*.md`.

5. Validate before declaring success.
- Run targeted tests for touched scope.
- Run full `pytest` when shared model/loader/training code changes.
- Run a small GPU smoke when CUDA/training behavior changes.
- Re-read changed code for accidental breakage.

6. Tell the truth about status.
- Never claim a run, download, test, or metric that was not actually produced.
- If a background process is still running, report PID/log/progress.
- If a result is smoke-only, label it smoke-only.

7. Keep research code reproducible.
- Keep experiment settings explicit in configs.
- Do not silently change old result artifacts.
- Keep paper claims aligned with saved `metrics.json`.
- Preserve precision/recall/speed trade-offs in summaries.

## Bugs

- Reproduce/localize first.
- Fix root cause with minimal surface area.
- Do not mix unrelated fixes silently.
- If fix does not hold under validation, iterate until root cause is addressed or residual risk is documented.

## Features

- Integrate at the narrowest stable point.
- Extend existing flows before creating parallel ones.
- Add tests for new dataset loaders or config-driven behavior.
- Keep new scripts under `scripts/` and new results under `results/`.

## ML And Research Code

- Prefer reproducibility over shortcuts.
- Keep model variants comparable by dataset, seed, epoch policy, and threshold policy.
- Do not report Dice without checking precision and recall for thin-structure segmentation.
- Keep AZ interpretation separate from benchmark claims:
  - geometry may be mathematically coherent;
  - implementation may still lose on optimization, recall, or speed.

## Hygiene

- Remove transient debug scripts/files after conclusions are captured.
- Clean `.pytest_cache` and `__pycache__` outside `.venv` when appropriate.
- Do not delete downloaded datasets or user artifacts unless explicitly asked.
