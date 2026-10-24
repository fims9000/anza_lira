#!/usr/bin/env python3
"""Ingest visualMetro sparse track clicks; never invent background masks."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bubble_chamber.visualmetro_import import write_manifest
from bubble_chamber.weak_labels import export_pairs_csv
from bubble_chamber.geometry_gap_eval import evaluate_geometry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-root',type=Path,required=True,help='extracted film/frame tree (e.g. 070_008.7z contents)')
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest=write_manifest(args.measurement_root,args.out_dir/'sparse_manifest.json')
    export_pairs_csv(manifest,args.out_dir/'stereo_correspondences.csv')
    geometry=evaluate_geometry(manifest)
    (args.out_dir/'coordinate_gap_diagnostic.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summary':manifest['summary'],
                      'segmentation_ground_truth_available':False,
                      'independent_film_count':geometry['independent_film_count'],
                      'coordinate_only_geometry_mean_px':{k:v['mean_px'] for k,v in geometry['per_method'].items()}},
                     ensure_ascii=False,indent=2))

if __name__=='__main__':main()
