#!/usr/bin/env python3
"""Run leave-one-frame-out relation experiment on raw visualMetro film070 exports."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bubble_chamber.visualmetro_import import read_measurement_tree
from bubble_chamber.real_sparse_lira import run_sparse_study


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--measurements', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    manifest = read_measurement_tree(a.measurements)
    result, pred = run_sparse_study(manifest)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir/'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    if pred:
        with (a.out_dir/'test_predictions.csv').open('w', newline='') as f:
            wr=csv.DictWriter(f, fieldnames=list(pred[0]));wr.writeheader();wr.writerows(pred)
    print(json.dumps({'status':result['status'],'aggregate':result['aggregate'],'candidates':result['dataset_summary']['candidates']},indent=2))

if __name__ == '__main__': main()
