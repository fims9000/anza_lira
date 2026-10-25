#!/usr/bin/env python3
from pathlib import Path
import argparse,json,hashlib,sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s32_review_queue import build_review_queue,inject_review_queue
if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--drafts',type=Path,required=True);p.add_argument('--s31-html',type=Path,required=True)
 p.add_argument('--out-dir',type=Path,required=True)
 a=p.parse_args();drafts=json.loads(a.drafts.read_text())
 q=build_review_queue(drafts,hashlib.sha256(a.drafts.read_bytes()).hexdigest())
 html=inject_review_queue(a.s31_html.read_text(encoding='utf-8'),drafts,q)
 a.out_dir.mkdir(parents=True,exist_ok=True)
 (a.out_dir/'S32_review_priority.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
 (a.out_dir/'S32_MACHINE_REVIEWER.html').write_text(html)
 print({'source_crops':q['original_photos'],'machine_drafts':q['machine_drafts'],
        'possible_pairs':q['possible_pairs'],'true_human_confirmations':0})
