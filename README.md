# ANZA-LIRA

ANZA-LIRA is a research codebase for thin-structure segmentation with a focus on
anisotropic fuzzy local aggregation.

This README is intentionally project-level (open-repo version): method,
mathematical idea, datasets, and implementation workflow, without reporting
paper-specific metric tables.

## Core idea

Standard local convolution is isotropic. For elongated structures (vessels,
roads), local evidence is directional and uncertain.  
ANZA-LIRA uses a local operator that combines:

1. directional geometric sensitivity (anisotropy),
2. fuzzy local agreement (soft membership),
3. normalized neighborhood aggregation.

## Mathematical formulation (compact)

For center point `p`, neighbor `q`, and local mode `r`:

`w_r(p, q) = mu_r(p) * mu_r(q) * kappa_r(q - p)`

where:
- `mu_r(.)` is fuzzy membership,
- `kappa_r(.)` is anisotropic directional kernel.

Normalized local weight:

`w_tilde_r(p, q) = w_r(p, q) / (sum_{q in N(p)} sum_{m=1..R} w_m(p, q) + eps)`

Local aggregation:

`z(p) = sum_{q in N(p)} sum_{r=1..R} w_tilde_r(p, q) * V(q)`

The operator is integrated into a U-Net style segmentation pipeline
(`baseline`, `az_cat`, `az_thesis`, and related variants).

## Geometry implementation notes

- Fixed directional metric (`fixed_cat_map`) and learned/hybrid modes are
  supported.
- The code logs geometry diagnostics (anisotropy gap, metric conditioning,
  rule-usage entropy) for interpretability and stability checks.
- The operator is used as a finite feature-space local block (practical neural
  implementation), not as a full dynamical-system solver.

## Supported dataset families

- Retinal/medical vessel segmentation:
  - `DRIVE`
  - `CHASE_DB1`
  - `FIVES`
  - `HRF_SegPlus`
- GIS road segmentation:
  - `Roads_HF`
  - `GlobalScaleRoad / SpaceNet3_prepared`
- Synthetic/auxiliary domain-specific sets (when present in `data/`).

See dataset/config routing in:
- `utils.py` (`build_dataloaders`, dataset resolvers)
- `configs/` (task-specific experiment setups)

## Repository structure

- `models/` — AZ blocks and segmentation architectures
- `train.py` — main training/evaluation entry point
- `utils.py` — data loading, losses, metrics, threshold sweep, reporting
- `configs/` — experiment configs
- `scripts/` — helper pipelines (training queues, evaluation, asset prep)
- `results/` — run outputs and paper materials

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --config configs/fives_benchmark.yaml --variants baseline,az_thesis
```

## Public repository note

This repository is maintained in public-safe form:
- no private manuscripts in tracked files,
- generated run artifacts are partially ignored,
- project-level docs are preferred over conference-specific internal notes.

Public URL: `https://github.com/fims9000/anza_lira`
