#!/usr/bin/env python3
"""S62: image-only candidate availability on controlled injected 3-piece scenes.

No reference trajectory is supplied to detector functions.  Generator proxy
curves enter solely the *evaluation* routine and private synthetic QA previews.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_simulated_gaps import extract_track_templates
from bubble_chamber.s56_three_piece_scenes import make_three_piece_scene,SCENARIOS
from bubble_chamber.s62_image_candidate_diagnostics import (
    image_only_predictions,proxy_mask_fragment_support,
    proxy_extracted_piece_support,scene_upper_bound_from_proxy_support)
from scripts.research.bubble_chamber.s37_benchmark_review_risk import split_source_frames
from scripts.research.bubble_chamber.s59_benchmark_candidate_retrieval import sha256

SCHEMA='s62_image_only_candidate_upper_bound_SYNTHETIC_ONLY_v1'
PROTOCOL=ROOT/'docs/bubble_chamber/S62_FROZEN_IMAGE_ONLY_CANDIDATE_AUDIT_2026_10_26.md'
SEEDS={'005':(20266203,20266204),'006':(20266213,20266214)}
N_SCENES_PER_SEED=24
ARMS=('multiscale_ridge','local_darkness','sobel_edge','image_only_ridge_polylines')


def aggregate(rows:list[dict],*,seed:int)->dict:
    if not rows:raise ValueError('No synthetic scenes')
    values=defaultdict(list)
    for r in rows:
        for k,v in r.items():
            if k!='image_only_proposal_count':values[k].append(float(v))
    rng=np.random.default_rng(seed)
    output={}
    for k,v in values.items():
        a=np.asarray(v,float)
        ix=rng.integers(0,len(a),size=(1000,len(a)))
        output[k]={'mean_per_synthetic_scene':float(a.mean()),
                   'scene_bootstrap_95pct':[float(q) for q in np.quantile(a[ix].mean(1),[.025,.975])],
                   'raw_total':int(a.sum()) if np.all(a==np.round(a)) else None}
    output['scene_count']=len(rows)
    if 'image_only_proposal_count' in rows[0]:
        output['image_only_proposals_mean']=float(np.mean([x['image_only_proposal_count'] for x in rows]))
    return output


def evaluate_one(gray:np.ndarray,scene_nodes:list[np.ndarray])->dict:
    """Prediction FIRST; labels/proxies only in evaluation second."""
    prediction=image_only_predictions(gray)
    results={}
    for arm,mask in prediction['masks'].items():
        proxy=[proxy_mask_fragment_support(mask,x) for x in scene_nodes]
        result=scene_upper_bound_from_proxy_support(proxy)
        result['activated_pixels_fraction']=float(mask.mean())
        results[arm]=result
    pieces=prediction['image_only_ridge_polylines']
    proxy=[proxy_extracted_piece_support(pieces,x) for x in scene_nodes]
    results['image_only_ridge_polylines']={**scene_upper_bound_from_proxy_support(proxy),
                                           'image_only_proposal_count':len(pieces)}
    return results,prediction


def private_preview(scene,prediction:dict,path:Path)->None:
    # Only screenshot-like diagnostic of a synthetic overlay on a real H50Z
    # background. This is never native physics GT.
    pic=cv2.cvtColor(scene.gray,cv2.COLOR_GRAY2BGR)
    for i,points in enumerate(scene.nodes):
        color=(75,215,70) if i<6 else (60,85,250)
        cv2.polylines(pic,[np.rint(points).astype(np.int32)],False,color,1,cv2.LINE_AA)
    for pts in prediction['image_only_ridge_polylines']:
        cv2.polylines(pic,[np.rint(pts).astype(np.int32)],False,(240,110,50),1,cv2.LINE_AA)
    vis=cv2.resize(pic,(768,768),interpolation=cv2.INTER_NEAREST)
    cv2.putText(vis,'INJECTED ARTIFICIAL TRACKS - NOT NATIVE GT',(14,26),
                cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    path.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(path),vis)


def run(visualmetro:Path,tiles:Path,out:Path,private_dir:Path|None=None,*,smoke:bool=False):
    manifest=audited_large_manifest(visualmetro)
    train,cal,test,split_hash=split_source_frames(manifest)
    assert len(train)==79 and len(cal)==30 and len(test)==36
    templates=extract_track_templates(manifest,chosen_frames=test)
    public={'schema':SCHEMA,'protocol_sha256':sha256(PROTOCOL),
            'source_frame_split_sha256':split_hash,
            'visualmetro_070_source_complete_frames':len(manifest['frames']),
            'source_GT_only_SYNTHETIC_INJECTION':True,
            'source_polylines_never_provided_to_pixel_detectors':True,
            'native_H50Z_physical_track_GT_count':0,
            'native_photographs_one_physical_run':True,
            'fixed_mask_activation_fraction':.035,
            'smoke_only_NOT_confirmatory':smoke,
            'arms':ARMS,'directions':{}}
    scene_n=2 if smoke else N_SCENES_PER_SEED
    for source,target in (('005','006'),('006','005')):
        photos=sorted(tiles.glob(f'H50Z_030_01_{target}_*.png'))
        expected=18 if target=='006' else 21
        if len(photos)!=expected:raise ValueError('Incorrect private H50Z photo crop bank')
        public['directions'][f'{source}_to_{target}']={'photo_for_detection':target,
                'source_frame_set_used':'S37_TEST_ONLY_36_FRAMES',
                'seeds':{}}
        for seed in SEEDS[source] if not smoke else (20266291+(source=='006'),):
            rng=np.random.default_rng(seed)
            collected={k:[] for k in ARMS}
            for k in range(scene_n):
                photo=photos[int(rng.integers(len(photos)))]
                scene=make_three_piece_scene(photo,templates,
                    seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
                results,prediction=evaluate_one(scene.gray,scene.nodes)
                for name,data in results.items():
                    collected[name].append({key:value for key,value in data.items()
                                            if isinstance(value,(int,float,bool))})
                if private_dir is not None and k==0:
                    private_preview(scene,prediction,
                        private_dir/f'S62_PRIVATE_SYNTHETIC_{source}_to_{target}_seed{seed}.png')
            public['directions'][f'{source}_to_{target}']['seeds'][str(seed)]=(
                {arm:aggregate(rows,seed=seed+23+q) for q,(arm,rows) in enumerate(collected.items())})
            print('S62',source,'->',target,'seed',seed,'image-derived proposals',
                  public['directions'][f'{source}_to_{target}']['seeds'][str(seed)]
                  ['image_only_ridge_polylines']['image_only_proposals_mean'],flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(public,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return public


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--private-dir',type=Path)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    run(a.visualmetro,a.photo_tiles,a.out,a.private_dir,smoke=a.smoke)
if __name__=='__main__':main()
