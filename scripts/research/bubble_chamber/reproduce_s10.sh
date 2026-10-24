#!/usr/bin/env bash
# S10 matched-source original AZUNet versus U-Net under independent model seeds.
set -euo pipefail
ARCHIVE="${1:?Supply the original 1mH2Data.7z}"
python scripts/research/bubble_chamber/extract_7z_safe.py "$ARCHIVE" data/s10_measurements
for seed in 20261024 20261025 20261026; do
  python scripts/research/bubble_chamber/s7_original_retrain.py \
    --source visualmetro --measurements data/s10_measurements/1mH2Data/S1 \
    --out-dir "results/s10_fixed_seed${seed}" \
    --seed "$seed" --data-seed 20261024 --size 64 --epochs 8 --batch-size 4 \
    --train-limit 120 --val-limit 26 --test-limit 25 \
    --widths 4 6 8 12 --rules 2 --cpu-threads 2 --robust-train --az-residual
done
python scripts/research/bubble_chamber/s9_frangi_baseline.py \
  --measurements data/s10_measurements/1mH2Data/S1 \
  --seed 20261024 --data-seed 20261024 --out results/s10_fixed_frangi.json
python scripts/research/bubble_chamber/s10_collect_fixed.py \
  --root . --out docs/bubble_chamber/s10_fixed_split_metrics.json
