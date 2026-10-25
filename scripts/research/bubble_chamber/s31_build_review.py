#!/usr/bin/env python3
import argparse,json,hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s31_review_workbench import embed_machine_drafts
from bubble_chamber.s31_machine_first import validate_draft_pack

def run(s29_html:Path,drafts:Path,out:Path):
    data=json.loads(drafts.read_text());validate_draft_pack(data)
    html=embed_machine_drafts(s29_html.read_text(encoding='utf-8'),data)
    out.parent.mkdir(exist_ok=True,parents=True);out.write_text(html,encoding='utf-8')
    return {'schema':'s31_private_machine_first_reviewer_receipt_v1','S29_source_sha256':hashlib.sha256(s29_html.read_bytes()).hexdigest(),
            'S31_machine_drafts_sha256':hashlib.sha256(drafts.read_bytes()).hexdigest(),
            'HTML_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'crops':39,
            'machine_drafts':data['counts']['candidate_drafts'],'human_approved_physical_links':0}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--s29',required=True,type=Path);p.add_argument('--drafts',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
 a=p.parse_args();receipt=run(a.s29,a.drafts,a.out)
 print(json.dumps(receipt,ensure_ascii=False,indent=2));(a.out.parent/'S31_private_reviewer_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
