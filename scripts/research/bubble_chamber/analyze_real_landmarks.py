#!/usr/bin/env python3
"""Extract real annotated H50Z keypoint patches and fixed baseline diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from bubble_chamber.h50z_seed_analysis import make_keypoint_pilot

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--labels',type=Path,required=True)
 p.add_argument('--bmps',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--tile-size',type=int,default=384)
 a=p.parse_args()
 result=make_keypoint_pilot(a.labels,a.bmps,a.out,tile_size=a.tile_size)
 print(json.dumps({'status':result['status'],'audit':result['audit'],
                   'diagnostics':result['keypoint_proximity_diagnostics']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
