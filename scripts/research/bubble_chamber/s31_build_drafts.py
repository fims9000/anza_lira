#!/usr/bin/env python3
"""Produce provisional machine annotations using REAL H50Z images and S28 hypotheses."""
import argparse, json, re, hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s31_machine_first import source_locked_machine_drafts,validate_draft_pack

def run(source_html,proposals_json,tiles,out):
    source_html=Path(source_html);proposals_json=Path(proposals_json)
    m=re.search(r'<script type="application/json" id="payload">(.*?)</script>',source_html.read_text(encoding='utf-8'),re.S)
    if not m:raise ValueError('S29 reference photo HTML missing')
    src=json.loads(m.group(1))
    if src.get('schema')!='s26_review_visual_payload_v1' or not src.get('no_expert_labels_in_predictions'):
        raise ValueError('Source is not original S29 unreviewed photo payload')
    hypotheses=json.loads(proposals_json.read_text(encoding='utf8'))
    pack=source_locked_machine_drafts(hypotheses,src['photos'],tiles)
    validate_draft_pack(pack)
    pack['source_S28_proposals_sha256']=hashlib.sha256(proposals_json.read_bytes()).hexdigest()
    pack['source_S29_reviewer_sha256']=hashlib.sha256(source_html.read_bytes()).hexdigest()
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(pack,ensure_ascii=False,indent=2)+'\n')
    print('CREATED',pack['counts'])
    return pack

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source-html',type=Path,required=True)
    p.add_argument('--proposals-json',type=Path,required=True)
    p.add_argument('--tiles',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.source_html,a.proposals_json,a.tiles,a.out)
