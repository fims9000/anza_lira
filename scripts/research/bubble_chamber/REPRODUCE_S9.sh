#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/research/bubble_chamber/REPRODUCE_S9.sh /path/to/1mH2Data.7z" >&2
  exit 2
fi
python scripts/research/bubble_chamber/extract_7z_safe.py "$1" data/s9_measurements
for seed in 20261024 20261025 20261026; do
  python scripts/research/bubble_chamber/s7_original_retrain.py \
    --source visualmetro --measurements data/s9_measurements/1mH2Data/S1 \
    --out-dir "results/s9_replication_seed$seed" \
    --size 64 --epochs 8 --batch-size 4 --train-limit 120 \
    --val-limit 26 --test-limit 25 --widths 4 6 8 12 \
    --rules 2 --cpu-threads 2 --robust-train --az-residual --seed "$seed"
  python scripts/research/bubble_chamber/s9_frangi_baseline.py \
    --measurements data/s9_measurements/1mH2Data/S1 \
    --seed "$seed" --out "results/s9_frangi_seed$seed.json"
done
python scripts/research/bubble_chamber/s9_verify_checkpoints.py \
  --measurements data/s9_measurements/1mH2Data/S1 --results results \
  --out results/s9_checkpoint_replay.json
