#!/usr/bin/env python3
"""Build private S27 expert-review priority UI from 39 source-locked H50Z crops.

Every exported annotation remains MANUAL. An image-model proposal is never
silently promoted into expert track, background, or same-track truth.
"""
from __future__ import annotations
import argparse,base64,json,re,sys,hashlib
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s27_review_priority import rank_questions,embed_priorities


def audit_s26_html(img_html:str)->dict:
    m=re.search(r'<script type="application/json" id="payload">(.*?)</script>',img_html,re.S)
    if not m:raise ValueError('missing original S26 labels/preview payload')
    data=json.loads(m.group(1))
    if data.get('schema')!='s26_review_visual_payload_v1' or not data.get('no_expert_labels_in_predictions'):
        raise ValueError('legacy reviewer source not safe')
    return data


def make_contact_sheet(payload:dict,queue:dict,path:Path,max_rows=12):
    lookup={p['crop_id']:p for p in payload['photos']}
    cards=[]
    for i,q in enumerate(queue['questions'][:max_rows]):
        p=lookup[q['crop_id']]
        tiles=[]
        for key in ('original','unet','guided_az','consensus'):
            data=base64.b64decode(p['image_modes'][key].split(',',1)[1])
            raw=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            if raw is None or raw.shape[:2]!=(512,512):raise ValueError('invalid source/overlay in HTML')
            tiles.append(cv2.resize(raw,(196,196),interpolation=cv2.INTER_AREA))
        row=np.hstack(tiles)
        # LABEL IS META ONLY; not a prediction probability or physical particle identity.
        cv2.rectangle(row,(0,0),(row.shape[1]-1,24),(15,15,20),-1)
        cv2.putText(row,f'{i+1:02d} {q["photo"]} {q["stratum"]}  (not verified)',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
        cards.append(row)
    sheet=np.vstack(cards)
    if not cv2.imwrite(str(path),sheet):raise IOError('cannot write review-contact JPG/PNG')
    return path


def run(args):
    source=json.loads(args.metrics.read_text(encoding='utf-8'))
    if source.get('schema')!='s27_h50z_no_label_overlap_robustness_v1':
        raise ValueError('provenance-bound real-photo S27 metrics required')
    q=rank_questions(source['per_crop'])
    s=args.source_html.read_text(encoding='utf-8')
    pack=audit_s26_html(s)
    new=embed_priorities(s,q)
    args.out.mkdir(parents=True,exist_ok=True)
    html=args.out/'S27_prioritized_real_H50Z_expert_review.html'
    html.write_text(new,encoding='utf-8')
    (args.out/'S27_source_sha_locked_review_queue.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    make_contact_sheet(pack,q,args.out/'S27_expert_review_first12_contact.jpg')
    manifest={'schema':'s27_private_expert_review_ranking_receipt_v1',
       'original_S26_reviewer_sha256':hashlib.sha256(args.source_html.read_bytes()).hexdigest(),
       'new_S27_reviewer_sha256':hashlib.sha256(html.read_bytes()).hexdigest(),
       'source_crops':len(q['questions']),'selected_first12_ids':[r['crop_id'] for r in q['questions'][:12]],
       'two_source_photo_ids':['005','006'],'native_expert_track_pixel_truth':0,
       'newly_created_manual_review_strokes':0,'automatic_track_merges':0,
       'source_locked_original_review_JSON_schema_unchanged':True,
       'private_photo_content_embedded':True}
    (args.out/'S27_review_receipt.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'first12':manifest['selected_first12_ids'],'photos':len(pack['photos']),'html_bytes':html.stat().st_size},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--source-html',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())

if __name__=='__main__':main()
