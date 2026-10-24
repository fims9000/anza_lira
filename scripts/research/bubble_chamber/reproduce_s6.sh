#!/bin/sh
set -eu
if [ "$#" -ne 2 ];then echo "Usage: $0 /private/path/1mH2Data/S1 /path/to/results" >&2;exit 2;fi
DATA=$1
OUT=$2
[ -d "$DATA" ] || { echo "Missing measurements" >&2; exit 2; }
mkdir -p "$OUT"
python scripts/research/bubble_chamber/run_synthetic_anza_pretrain.py --measurements "$DATA" --out-dir "$OUT/train" --epochs 4 --batch-size 16 --size 64 --seed 20261024 --device auto
python scripts/research/bubble_chamber/evaluate_s6.py --measurements "$DATA" --training-dir "$OUT/train" --out-dir "$OUT/eval"
