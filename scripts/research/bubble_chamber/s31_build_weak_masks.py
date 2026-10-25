#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s31_weak_supervision import build_weak_masks

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--drafts',required=True,type=Path)
 p.add_argument('--tiles',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
 p.add_argument('--corrections',type=Path,default=None)
 a=p.parse_args();r=build_weak_masks(json.loads(a.drafts.read_text()),a.tiles,a.out,corrections=json.loads(a.corrections.read_text()) if a.corrections else None)
 print(json.dumps(r['counts'],ensure_ascii=False))
