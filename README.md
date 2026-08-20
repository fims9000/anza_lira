# ANZA-LIRA

ANZA-LIRA is a research codebase for segmentation and structural continuation
of thin seismic fault traces. The final repository release keeps the code,
tests, configs, research notes, and small final result tables, but excludes
datasets, checkpoints, raw experiment folders, and archive packages.

## Main Idea

ANZA explored local anisotropic fuzzy geometry and multiple Anosov-inspired
constraints for thin-structure segmentation. The final CRACKS V1.1 stability
study tested a reciprocal determinant-one structural-stability prior against a
more flexible generic anisotropic control.

LIRA separates dense fault evidence from structural reasoning. Its strongest
validated role is controlled structural continuation: candidate relations are
verified in context and accepted links can be reconstructed by a bounded
max-min path rule.

## Main Result

The strongest positive result is the historical controlled continuation result:

- AUROC: `0.9923`
- Recovery: `67.2%`
- False bridges: `0.78%`

The final CRACKS Structural Stability V1.1 multiseed run ended with:

`STOP_ANZA_STABILITY_NO_INCREMENTAL_VALUE`

The determinant-one ANZA stability prior did not pass the frozen incremental
development gate over the matched free-anisotropy control. LIRA development,
confirm evaluation, and expert descriptive evaluation were therefore not
authorized in that protocol.

## Repository Layout

- `models/` - segmentation models and geometry-aware layers
- `anza_*`, `anza2/`, `structural_stability_v1_1/` - bounded research modules
- `lira_*`, `anza_tracegraph/`, `path_completion/` - structural reasoning and continuation modules
- `configs/` - experiment configuration files
- `scripts/` - runnable experiment, validation, and reporting entrypoints
- `tests/` - regression and protocol tests
- `docs/` - final research documentation and claim boundaries
- `docs/results/` - small final metric tables and frozen manifests
- `.codex/notes/` - research history, negative results, and stop boundaries

`results/`, `data/`, checkpoints, and zip archives are local/generated artifacts
and are not part of the git release.

## Reproducibility

Use the existing Python environment when available:

```bash
/home/lebedeffson/Code/venv/bin/python -m pytest
```

The final V1.1 execution entrypoints are:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/run_anza_lira_ss_v1_1_pretrain.py
/home/lebedeffson/Code/venv/bin/python scripts/run_anza_lira_ss_v1_1_endgame.py
/home/lebedeffson/Code/venv/bin/python scripts/validate_anza_lira_ss_v1_1_endgame.py
```

Datasets and checkpoints are not included. Generated artifacts should stay in
ignored local folders such as `results/` or `_wip_backups/`.

## Final Documents

- [Final research report](docs/ANZA_LIRA_FINAL_RESEARCH.md)
- [Final results](docs/ANZA_LIRA_FINAL_RESULTS.md)
- [Claims and limitations](docs/CLAIMS_AND_LIMITATIONS.md)
- [Experiment ledger](docs/EXPERIMENT_LEDGER.md)
- [Final research status](docs/RESEARCH_STATUS_FINAL.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
