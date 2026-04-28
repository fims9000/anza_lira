# ANZA-LIRA

ANZA-LIRA is a research codebase for segmentation of thin elongated structures
(retinal vessels, roads, line-like objects) using anisotropic fuzzy local
aggregation in U-Net style pipelines.

This README is project-level and publication-safe: method, implementation,
reproducibility, and usage workflow.

## 1) What this project does

The repository provides:

1. baseline segmentation models,
2. AZ-enhanced models (`az_cat`, `az_thesis`, related variants),
3. training + validation threshold sweep,
4. geometry diagnostics and figure export scripts.

Primary task type is binary segmentation.

## 2) Core method (compact)

For center pixel `p`, neighbor `q`, local rule `r`:

`w_r(p, q) = mu_r(p) * mu_r(q) * kappa_r(q - p)`

where:
- `mu_r` is fuzzy rule membership,
- `kappa_r` is directional anisotropic compatibility.

Normalized aggregation:

`w_tilde_r(p, q) = w_r(p, q) / (sum_{q in N(p)} sum_{m=1..R} w_m(p, q) + eps)`

`z(p) = sum_{q in N(p)} sum_{r=1..R} w_tilde_r(p, q) * V(q)`

In practice this is implemented as an AZ block inserted into encoder/decoder
stages of segmentation architectures.

## 3) Datasets supported

Medical vessel segmentation:
- `DRIVE`
- `CHASE_DB1`
- `FIVES`
- `HRF_SegPlus`

GIS road segmentation:
- `Roads_HF`
- `global_roads` (SpaceNet3 prepared split)

Dataset routing and canonical names are handled in `utils.py`.

## 4) Repo layout

- `models/` — AZ layers and segmentation networks
- `train.py` — training + evaluation entry point
- `utils.py` — data loading, losses, metrics, threshold policy, reporting
- `configs/` — ready experiment configs
- `scripts/` — utility scripts (figure export, run helpers, diagnostics)
- `results/` — experiment outputs and prepared assets
- `tests/` — lightweight checks

## 5) Environment setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 6) Run training

Example:

```powershell
python train.py --config configs/fives_benchmark.yaml --variants baseline,az_thesis
```

Important:
- `metrics.json` is written per run in `results/<run_name>/`.
- threshold for test metrics is selected by validation sweep.

## 7) Geometry visualization (model-native)

Main export scripts:
- `scripts/export_geometry_attention_story.py`
- `scripts/export_geometry_clean_article_figure.py`

Example:

```powershell
python scripts/export_geometry_clean_article_figure.py `
  --results-dir results `
  --run article3_spacenet_sprint_v3_recover `
  --baseline-run article3_spacenet_sprint_v3_baseline `
  --sample-index 30 `
  --output-dir results/a3_final_package/final_article3/figures `
  --device cpu
```

The generated 2x2 figure includes:
1. input + GT,
2. error-centric baseline-vs-AZ map,
3. AZ orientation axis (model-derived),
4. anisotropy strength map.

## 8) Reproducibility notes

- Config-driven runs (`configs/*.yaml`)
- Fixed seeds in configs when required
- Per-run saved artifacts:
  - `metrics.json`
  - `checkpoint_best.pt`
  - `history.json` (if enabled)
- Scripted figure generation from checkpoints/results

## 9) Current status of defaults

AZ experiments support both legacy and newer geometry modes.
Checkpoint loading includes backward compatibility for older AZ snapshots.

## 10) Public repo policy

This repository is kept in public-safe form:
- no private manuscript binaries in tracked files,
- generated heavy artifacts are mostly ignored via `.gitignore`,
- project-level technical documentation is preferred.

Repository:
`https://github.com/fims9000/anza_lira`
