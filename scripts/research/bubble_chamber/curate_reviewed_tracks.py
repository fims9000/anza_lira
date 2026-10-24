#!/usr/bin/env python3
"""Turn audited human pair labels into ordered, conservative 2D track groups."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.curation import curate_tracks


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=json.loads(a.result.read_text(encoding='utf-8'))
    labels=json.loads(a.labels.read_text(encoding='utf-8'))
    c=curate_tracks(result,labels)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(c,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({"track_groups":c['track_group_count'],"expert_accepted_bridges":c['approved_gap_count'],"out":str(a.out)}))

if __name__=='__main__':main()
