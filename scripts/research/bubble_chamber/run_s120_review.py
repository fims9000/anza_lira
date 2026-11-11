"""Render S120 uniform-green full-frame review from original BMP and S118."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from bubble_chamber.s120_review import run

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bmp', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True,
                   help='Trust only your own locally trained joblib file')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.bmp, a.run_dir, a.features, a.model, a.out), indent=2))
