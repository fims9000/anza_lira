#!/usr/bin/env python3
"""S78/S79 on 39 real H50Z center patches, no expert GT; PRIVATE overlays."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75
from scripts.research.bubble_chamber import s78_calibrate_then_test_exclusive_graph as s78
from scripts.research.bubble_chamber import s79_calibrate_then_test_gate_rescue as s79

SCHEMA='S78_S79_REAL_H50Z_UNLABELLED_READONLY_SMOKE_V1'


def center(image):
    if image.shape!=(512,512) or image.dtype!=np.uint8:
        raise ValueError('Expected real 512x512 crop')
    return image[160:352,160:352].copy()


def preview(im,arms,out):
    import numpy as np
    montage=[]
    for name,pred in arms.items():
        x=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        for curve in pred['polylines']:
            cv2.polylines(x,[np.rint(curve).astype('int32')],False,(60,200,225),1,cv2.LINE_AA)
        for e in pred['chosen']:
            a=pred['polylines'][e.i][e.end_i];b=pred['polylines'][e.j][e.end_j]
            cv2.line(x,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),(220,90,180),2,cv2.LINE_AA)
        x=cv2.resize(x,(576,576),interpolation=cv2.INTER_NEAREST)
        cv2.putText(x,name[:27],(10,26),cv2.FONT_HERSHEY_SIMPLEX,.57,(255,255,255),2)
        cv2.putText(x,'NO EXPERT GT / UNKNOWN',(8,556),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),2)
        montage.append(x)
    out.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(out),np.hstack(montage)):
        raise RuntimeError('Failed private PNG write')


def run(visualmetro,tiles,preview_dir=None):
    s75.frozen_s72(s78.SOURCE_CAL)
    templates,split=s72._source_data(visualmetro)
    if split!='448f21ca3a709010ed972a596ed046cadff0643112905fd1d029c91f8b14faef':
        raise ValueError('Changed film 070 split')
    photos={p:sorted(tiles.glob(f'H50Z_030_01_{p}_*.png')) for p in ('005','006')}
    if (len(photos['005']),len(photos['006']))!=(21,18):
        raise ValueError('Original H50Z photo bank count differs from 39')
    cal=json.loads(s78.SOURCE_CAL.read_text())
    arms=('baseline','s78_alternative_strict','s78_alternative_relaxed',
          's79_rescue_strict','s79_rescue_relaxed')
    metrics=('machine_curves','advisory_joins_UNKNOWN','raw_gap_hypotheses',
             'novel_alternative_candidates','selected_pixel_alternatives')
    totals={arm:{k:0 for k in metrics} for arm in arms}
    for p in ('005','006'):
        source='006' if p=='005' else '005'
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        graph=cal['directions'][f'{source}_to_{p}']['CAL_selected']
        # Six fixed previews selected *before* any prediction; never cherry-picked.
        fixed={0,len(photos[p])//2,len(photos[p])-1}
        for idx,file in enumerate(photos[p]):
            image=cv2.imread(str(file),cv2.IMREAD_GRAYSCALE)
            if image is None:raise ValueError('Unreadable real photograph crop')
            pixels=center(image)
            a=s78.infer_arms_from_pixels(pixels,scorer,cutoff,graph)
            b=s79.infer_arms_from_pixels(pixels,scorer,cutoff,graph)
            if a['baseline']['chosen']!=b['baseline']['chosen']:
                raise AssertionError('Native baseline changed during new hypothesis layering')
            preds={'baseline':a['baseline'],
                's78_alternative_strict':a['alternative_strict'],
                's78_alternative_relaxed':a['alternative_relaxed'],
                's79_rescue_strict':b['rescue_strict'],
                's79_rescue_relaxed':b['rescue_relaxed']}
            for arm,pred in preds.items():
                summary=totals[arm]
                summary['machine_curves']+=len(pred['polylines'])
                summary['advisory_joins_UNKNOWN']+=len(pred['chosen'])
                summary['raw_gap_hypotheses']+=int(pred.get('raw_local_gap_proposals',pred.get('raw_S77_gap_hypotheses',0)))
                summary['novel_alternative_candidates']+=int(pred.get('candidate_extra_alternatives',pred.get('pixel_alternatives_added',0)))
                summary['selected_pixel_alternatives']+=int(pred.get('selected_pixel_alternatives',0))
            if preview_dir is not None and idx in fixed:
                preview(pixels,{'Original S72':preds['baseline'],
                     'S78 short-gap alternatives':preds['s78_alternative_relaxed'],
                     'S79 pixel-backed rescue':preds['s79_rescue_relaxed']},
                     Path(preview_dir)/f'S78_S79_NATIVE_{p}_{idx:02}.png')
    return {'schema':SCHEMA,'experiments':['ANZA-LIRA-BC-S78-20261026','ANZA-LIRA-BC-S79-20261026'],
         'source_S72_CAL_SHA256':s78.S72_SHA256,'source_frame_split_sha256':split,
         'original_H50Z_crop_count':39,'fixed_window':'center_192_of_512',
         'actual_independent_photographic_runs':1,'native_H50Z_GT_count':0,
         'physical_join_authorization':False,'all_native_proposals_are_UNKNOWN':True,
         'aggregated_machine_counts':totals}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--visualmetro',type=Path,required=True)
    ap.add_argument('--photo-tiles',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--private-preview-dir',type=Path)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Do not overwrite native aggregate')
    result=run(args.visualmetro,args.photo_tiles,args.private_preview_dir)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['aggregated_machine_counts'],ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
