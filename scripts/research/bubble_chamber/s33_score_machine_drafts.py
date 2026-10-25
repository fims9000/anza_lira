#!/usr/bin/env python3
"""Add review-only AZUNet normalized evidence to existing S31 hypotheses.

Saves no certified segmentation masks, never changes S31 reviewer JSON.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,cv2,torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s33_evaluate_photometric import load_model
from bubble_chamber.s33_normalized_inference import score_s31_drafts,patchwise_probabilities


def run(draft_file,tiles,ck_dir,out,*,seed=20261025,start=0,limit=None):
    torch.set_num_threads(2)
    pack=json.loads(Path(draft_file).read_text(encoding='utf8'))
    models={mode:load_model(ck_dir,'azunet',mode,seed) for mode in ('quantile','local')}
    selected=pack['photos'][start:] if limit is None else pack['photos'][start:start+limit]
    if not selected:raise ValueError('Empty source crop chunk')
    record=score_s31_drafts(pack,tiles,models,stride=64,crop_subset={x['crop_id'] for x in selected})
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    # Only candidate evidence visualization, not gold/photo mask.
    example=next((p for p in selected if p['crop_id'].endswith('start00') and p['photo_id']=='006'),None)
    if example is not None:
        name=example['crop_id'];gray=cv2.imread(str(Path(tiles)/(name+'.png')),cv2.IMREAD_GRAYSCALE)
        x=patchwise_probabilities(models['quantile'],gray,mode='quantile')
        heat=cv2.applyColorMap(np.uint8(np.clip(x*255.,0,255)),cv2.COLORMAP_TURBO)
        base=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        overlay=np.hstack((base,cv2.addWeighted(base,.55,heat,.45,0)))
        cv2.putText(overlay,'ORIGINAL H50Z / AZUNet probability UNVERIFIED',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,245,245),1,cv2.LINE_AA)
        cv2.imwrite(str(out.parent/'S33_AZUNet_quantile_heatmap_UNVERIFIED.png'),overlay)
    print('S33 draft evidence:',record['source_crops'],'real crops',len(record['rows']),'uncalibrated proposals',flush=True)
    return record

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--drafts',type=Path,required=True)
 p.add_argument('--tiles',type=Path,required=True)
 p.add_argument('--models',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--start',type=int,default=0)
 p.add_argument('--limit',type=int,default=None)
 a=p.parse_args();run(a.drafts,a.tiles,a.models,a.out,start=a.start,limit=a.limit)
