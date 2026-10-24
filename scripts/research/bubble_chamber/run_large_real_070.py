#!/usr/bin/env python3
"""Run audited large-film visualMetro LIRA relation study without BMP images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from bubble_chamber.large_archive import save_audit
from bubble_chamber.large_relation_study import LargeRelationProtocol, save_study


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--measurements', type=Path, required=True, help='Extracted 1mH2Data/S1 folder containing 070_* subfolders')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--cuts', nargs='+', type=float, default=[0.35,0.50,0.65])
    parser.add_argument('--missing-clicks', type=int, default=1, help='Expert clicks synthetically hidden between observed fragments')
    args=parser.parse_args()
    manifest=save_audit(args.measurements,args.out_dir/'audit.json')
    cfg=LargeRelationProtocol(seed=args.seed,cuts=tuple(args.cuts),missing_clicks=args.missing_clicks)
    metrics=save_study(manifest,args.out_dir,cfg)
    print(json.dumps({'experiment':metrics['experiment_id'],'archive_summary':manifest['summary'],
                      'split':metrics['split_counts'],
                      'test_metrics':{n:m['test'] for n,m in metrics['variants'].items()}},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
