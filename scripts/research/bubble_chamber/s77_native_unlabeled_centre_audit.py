#!/usr/bin/env python3
"""S77 offline genuine-H50Z machine-only smoke, never physics accuracy.

Exactly one fixed central 192px patch is used per each of the 39 private 512px
source tiles. No pixel is published; no reference/synthetic GT is used.
"""
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
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75
from scripts.research.bubble_chamber import s77_calibrate_then_test_short_gaps as s77


def centre_crop_192(gray):
    if gray.shape!=(512,512) or gray.dtype!=np.uint8:
        raise ValueError('Private source must be a 512x512 grayscale PNG')
    crop=gray[160:352,160:352]
    assert crop.shape==(192,192)
    return crop.copy()


def run(visualmetro,tiles,*,private_preview_dir=None):
    s75.frozen_s72(s77.SOURCE_CAL)
    templates,split_sha=s72._source_data(visualmetro)
    if split_sha!='448f21ca3a709010ed972a596ed046cadff0643112905fd1d029c91f8b14faef':
        raise ValueError('Unexpected source film split')
    files={p:sorted(tiles.glob(f'H50Z_030_01_{p}_*.png')) for p in ('005','006')}
    if [len(files[p]) for p in ('005','006')]!=[21,18]:
        raise ValueError('Expected original 39 real unlabelled H50Z tiles')
    cal=json.loads(s77.SOURCE_CAL.read_text())
    aggregated={arm:{'pixel_candidate_curves':0,'graph_advisory_joins_UNKNOWN':0,
                    'pixel_supported_short_gap_repairs':0,'native_physics_GT_count':0}
                for arm in s77.ARMS}
    private=[]
    for photo in ('005','006'):
        source='006' if photo=='005' else '005'
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        arm=cal['directions'][f'{source}_to_{photo}']['CAL_selected']
        for file in files[photo]:
            im=cv2.imread(str(file),cv2.IMREAD_GRAYSCALE)
            if im is None:raise ValueError('Unreadable private native crop')
            pixels=centre_crop_192(im)
            variants=s77.infer_arms_from_pixels(pixels,scorer,cutoff,arm)
            for method,pred in variants.items():
                s=aggregated[method]
                s['pixel_candidate_curves']+=len(pred['polylines'])
                s['graph_advisory_joins_UNKNOWN']+=len(pred['chosen'])
                s['pixel_supported_short_gap_repairs']+=int(pred['attempted_short_gap_merged_pairs'])
            if private_preview_dir is not None:
                quality=sum(variants[m]['attempted_short_gap_merged_pairs'] for m in s77.ARMS[1:])
                private.append((quality,pixels,variants))
    if private_preview_dir is not None:
        out=Path(private_preview_dir);out.mkdir(parents=True,exist_ok=True)
        for old in out.glob('S77_PRIVATE_NATIVE_*.png'):old.unlink()
        for n,(_,pixels,variants) in enumerate(sorted(private,key=lambda z:-z[0])[:6],1):
            imgs=[]
            for method in s77.ARMS:
                vis=cv2.cvtColor(pixels,cv2.COLOR_GRAY2BGR)
                for curve in variants[method]['polylines']:
                    cv2.polylines(vis,[np.rint(curve).astype('int32')],False,(0,210,225),1,cv2.LINE_AA)
                for edge in variants[method]['chosen']:
                    a=variants[method]['polylines'][edge.i][edge.end_i]
                    b=variants[method]['polylines'][edge.j][edge.end_j]
                    cv2.line(vis,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),
                             (210,60,230),1,cv2.LINE_AA)
                vis=cv2.resize(vis,(576,576),interpolation=cv2.INTER_NEAREST)
                cv2.putText(vis,method,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                cv2.putText(vis,'MACHINE UNKNOWN / NO EXPERT GT',(12,555),
                            cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
                imgs.append(vis)
            cv2.imwrite(str(out/f'S77_PRIVATE_NATIVE_{n:02d}.png'),np.hstack(imgs))
    return {'schema':'S77_NATIVE_UNLABELED_MACHINE_ONLY_SMOKE_v1',
            'experiment_id':s77.EXPERIMENT_ID,'fixed_crop':'center_192_of_original_512',
            'input_native_unlabelled_crops':39,'photos_from_independent_runs':False,
            'native_physics_GT_count':0,'physical_auto_join_authorized':False,
            'graph_proposals_are_unknown':True,'aggregates':aggregated,
            'source_S72_cal_SHA256':s77.LOCKED_S72_CAL_SHA256}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--private-preview-dir',type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Do not overwrite historical native smoke')
    x=run(a.visualmetro,a.photo_tiles,private_preview_dir=a.private_preview_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Native S77 machine-only smoke (no native truth):',json.dumps(x['aggregates']),flush=True)
if __name__=='__main__':main()
