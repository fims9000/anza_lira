#!/usr/bin/env python3
"""S1: evaluate global safety gating on disjoint NEW synthetic scene seeds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.graph_safety import GraphSafetyProtocol,run_graph_safety_study


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    result=run_graph_safety_study(GraphSafetyProtocol())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'selected_margin':result['frozen_margin'],'selection_status':result['selection_status'], 'test':result['test']},indent=2))

if __name__=='__main__':main()
