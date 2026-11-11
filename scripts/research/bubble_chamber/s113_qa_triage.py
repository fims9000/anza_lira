#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s113_qa_triage import analyze_saved_summary
if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--private-summary',type=Path,required=True)
 p.add_argument('--private-out',type=Path,required=True)
 a=p.parse_args();print(json.dumps(analyze_saved_summary(a.private_summary,a.private_out),ensure_ascii=False,indent=2))