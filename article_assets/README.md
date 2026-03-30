# Article Assets

This directory stores article-ready visual materials for the segmentation paper.

Files included by default:

- `segmentation_pipeline.svg` - main inference pipeline for the segmentation article.
- `segmentation_training.svg` - training and loss pipeline.
- `exports/` - generated qualitative examples from `scripts/export_drive_article_assets.py`.

Recommended workflow:

1. Run training and keep the target checkpoint under `results/`.
2. Export qualitative panels with:
   `python scripts/export_drive_article_assets.py --run <run_name> --samples 0,3,7 --device cuda`
3. Use the generated `article_grid.png` panels and `metrics.json` in the paper draft.
