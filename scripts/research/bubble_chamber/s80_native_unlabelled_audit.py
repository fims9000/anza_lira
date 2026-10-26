#!/usr/bin/env python3
"""S80 private 39-crop center read-only inspection. NO native ground truth."""
from __future__ import annotations
import argparse
import json
from hashlib import sha256
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75
from scripts.research.bubble_chamber.s80_calibrate_then_test_multiscale import (
   ARMS,S72_CAL,LOCKED_S72_SHA256,SPLIT_SHA,EXPERIMENT_ID,infer_from_pixels)


def centre_crop(gray):
    if gray.shape!=(512,512) or gray.dtype!=np.uint8:
        raise ValueError('Need the original 512x512 grayscale private photograph tile')
    return gray[160:352,160:352].copy()


def run(visualmetro:Path,tiles:Path,*,preview_dir:Path|None=None)->dict:
    if sha256(S72_CAL.read_bytes()).hexdigest()!=LOCKED_S72_SHA256:
        raise ValueError('Original S72 CAL SHA changed')
    frozen=s75.frozen_s72(S72_CAL)
    templates,split=s72._source_data(visualmetro)
    if split!=SPLIT_SHA:raise ValueError('S37 source-film whole-frame split changed')
    files={p:sorted(tiles.glob(f'H50Z_030_01_{p}_*.png')) for p in ('005','006')}
    if len(files['005'])!=21 or len(files['006'])!=18:
        raise ValueError('39 original private tiles required')
    aggregate={arm:{'image_only_line_candidates':0,'graph_advisory_links_UNKNOWN':0,
             'new_multiscale_line_instances':0,'rejected_duplicate_candidates':0,
             'native_physics_GT':0} for arm in ARMS}
    examples=[]
    for photo in ('005','006'):
        source='006' if photo=='005' else '005'
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        direction=f'{source}_to_{photo}'
        method=frozen['directions'][direction]['CAL_selected']
        for f in files[photo]:
            im=cv2.imread(str(f),cv2.IMREAD_GRAYSCALE)
            if im is None:raise ValueError('Failed to open a source private pixel tile')
            gray=centre_crop(im)
            result=infer_from_pixels(gray,scorer,cutoff,method)
            for arm,val in result.items():
                x=aggregate[arm]
                x['image_only_line_candidates']+=len(val['polylines'])
                x['graph_advisory_links_UNKNOWN']+=len(val['chosen'])
                x['new_multiscale_line_instances']+=int(val['added_pixel_instances'])
                x['rejected_duplicate_candidates']+=int(val['duplicate_suppressed'])
            if preview_dir is not None:
                # Real pixels are retained only in the private preview, never JSON/Git.
                priority=result['s80_consensus_add6']['added_pixel_instances']
                examples.append((priority,gray,result))
    if preview_dir is not None:
        preview_dir.mkdir(parents=True,exist_ok=True)
        for i,(_,pixels,rows) in enumerate(sorted(examples,key=lambda z:-z[0])[:6],1):
            imgs=[]
            for arm in ARMS:
                pred=rows[arm]
                canvas=cv2.cvtColor(pixels,cv2.COLOR_GRAY2BGR)
                originals=pred['original_polylines_retained']
                for ci,pts in enumerate(pred['polylines']):
                    col=(0,210,200) if ci<originals else (30,70,230)
                    cv2.polylines(canvas,[np.rint(pts).astype(np.int32)],False,col,1,cv2.LINE_AA)
                for edge in pred['chosen']:
                    a=pred['polylines'][edge.i][edge.end_i]
                    b=pred['polylines'][edge.j][edge.end_j]
                    cv2.line(canvas,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),
                             (210,40,230),1,cv2.LINE_AA)
                canvas=cv2.resize(canvas,(576,576),interpolation=cv2.INTER_NEAREST)
                cv2.putText(canvas,arm,(12,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
                cv2.putText(canvas,'MACHINE UNKNOWN / NO PHYSICS GT',
                       (12,555),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),2)
                imgs.append(canvas)
            cv2.imwrite(str(preview_dir/f'S80_PRIVATE_H50Z_{i:02d}.png'),np.hstack(imgs))
    return {'schema':'S80_NATIVE_UNLABELLED_CENTRE_39_MACHINE_ONLY_V1',
            'experiment_id':EXPERIMENT_ID,'genuine_private_tiles':39,
            'central_192px_of_512px':True,'independent_physics_runs':False,
            'native_particle_GT':0,'all_proposals_status':'UNKNOWN',
            'native_physical_joins_authorized':False,'source_s72_cal_sha256':LOCKED_S72_SHA256,
            'aggregates':aggregate}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--visualmetro',type=Path,required=True)
    ap.add_argument('--photo-tiles',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--private-preview-dir',type=Path)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Never overwrite native read-only audit')
    report=run(a.visualmetro,a.photo_tiles,preview_dir=a.private_preview_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('S80 genuine H50Z WITHOUT physics labels:',json.dumps(report['aggregates']),flush=True)
if __name__=='__main__':main()
