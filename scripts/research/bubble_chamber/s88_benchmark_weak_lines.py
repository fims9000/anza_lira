#!/usr/bin/env python3
"""Evaluate S88 using PUBLIC GENERATED scenes only (no private H50Z)."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from bubble_chamber.s88_generated_benchmark import write_benchmark

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cal-count',type=int,default=8)
    parser.add_argument('--test-count',type=int,default=12)
    a=parser.parse_args()
    d=write_benchmark(a.output,a.cal_count,a.test_count)
    print(json.dumps({'output':str(a.output),'calibration':d['selected_using_CAL_only'],
                      'generated_TEST':{k:v['summary'] for k,v in d['independent_generated_TEST'].items()},
                      'native_physical_track_recall_measured':False},ensure_ascii=False,indent=2))
