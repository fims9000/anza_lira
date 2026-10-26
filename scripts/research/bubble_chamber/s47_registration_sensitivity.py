#!/usr/bin/env python3
"""S47 post-hoc sensitivity of S45 repeat cue to normal registration shifts.

The crop seeds, images, and synthetic geometry are re-used from S46. This
is NOT an independent holdout, and thresholds are NOT changed.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
from hashlib import sha256
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s46_registered_repeat_counterfactuals import (
    generate_path,inject,perpendicular_shift,SEED)
from bubble_chamber.s44_crossframe_persistence import (
    ExactBMP8,fit_common_fiducials,warp_corresponding_window,compare_registered_ridges)
from bubble_chamber.s45_registered_unknown import repeat_advisory

NORMAL_OFFSETS=(0,2,4,6,8)


def eval_offset(src:np.ndarray,paired:np.ndarray,path:np.ndarray,px:int)->tuple[bool,float]:
    if px not in NORMAL_OFFSETS:raise ValueError('Offset not preregistered')
    source=inject(src,path)
    other=inject(paired,perpendicular_shift(path,float(px)))
    score=compare_registered_ridges(source,other,[{
        'draft_id':'S47_SYNTHETIC_NOT_NATIVE','points_xy':path.tolist()}])[0]
    score['known_fiducial_uncertainty_NOT_NATIVE_TRUTH']=False
    return repeat_advisory(score)['registered_repeat_risk'],score['both_path_coverage_advisory']


def run(bmps:Path,s39_path:Path,s40_path:Path,out:Path)->dict:
    out.mkdir(parents=True,exist_ok=True)
    source=json.loads(s39_path.read_text())
    zones=json.loads(s40_path.read_text())['fiducials_PRIVATE']
    if len(source.get('tiles',[]))!=39 or len(zones)!=15:
        raise ValueError('Unproven source')
    reg=fit_common_fiducials(json.loads((bmps/'labels.json').read_text()))
    forward={'005':reg['forward_005_to_006_PRIVATE'],
             '006':reg['inverse_006_to_005_PRIVATE']}
    images={k:ExactBMP8(bmps/f'H50Z_030_01_{k}.bmp') for k in ('005','006')}
    rng=np.random.default_rng(SEED)
    rows=[]
    for i,tile in enumerate(source['tiles']):
        photo=tile['photo_id'];other='006' if photo=='005' else '005'
        origin=tile['native_origin_xy_PRIVATE']
        src=images[photo].crop(*origin,512,512)
        if sha256(src.tobytes()).hexdigest()!=tile['source_gray_sha256']:
            raise ValueError('Source SHA verification failed')
        aligned=warp_corresponding_window(images[other],origin,forward[photo])
        path=generate_path(rng,zones,origin,photo)
        results={}
        for shift in NORMAL_OFFSETS:
            hit,score=eval_offset(src,aligned,path,shift)
            results[str(shift)]={'flagged':hit,'coincident_coverage':round(float(score),5)}
        rows.append({'crop_id_PRIVATE':tile['crop_id'],'photo':photo,'normal_offsets':results})
        if (i+1)%8==0:print('S47 post-hoc normal-shift sweep',i+1,'/',39,flush=True)
    byphoto={}
    for p in ('005','006','all'):
        r=[r for r in rows if r['photo']==p or p=='all']
        summary={}
        for s in NORMAL_OFFSETS:
            flags=np.asarray([x['normal_offsets'][str(s)]['flagged'] for x in r],dtype=float)
            rng2=np.random.default_rng(20261047+int(s)+int(p if p!='all' else '0'))
            idx=rng2.integers(0,len(flags),(3000,len(flags)))
            summary[str(s)]={'n':len(flags),'flagged':int(flags.sum()),
                         'fraction':round(float(flags.mean()),5),
                         'crop_resample_CI_exploratory':np.quantile(flags[idx].mean(1),[.025,.975]).round(5).tolist()}
        byphoto[p]=summary
    out_public={'schema':'s47_posthoc_normal_registration_sensitivity_v1',
      'normal_shift_pixels':NORMAL_OFFSETS,
      'known_calibration_fiducial_mean_residual_px':round(reg['mean_common_residual_px'],3),
      'known_calibration_fiducial_max_residual_px':round(reg['max_common_residual_px'],3),
      'native_physics_GT_count':0,'independent_physical_runs':1,
      'images_native_original_2':True,'all_lines_programmatic_not_native_particles':True,
      'reuses_S46_seeds_and_geometry_not_independent_confirmation':True,
      'fixed_S45_thresholds_not_tuned':True,
      'results':byphoto,
      'use_only_for_robustness_not_particle_scratch_classification':True}
    (out/'S47_PUBLIC_REGISTRATION_SENSITIVITY.json').write_text(json.dumps(out_public,indent=2)+'\n')
    (out/'S47_PRIVATE_PATH_RESPONSES.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(out_public,indent=2),flush=True)
    return out_public


def cli():
    p=argparse.ArgumentParser()
    for x in ('bmp-directory','s39-private','s40-private','out'):
        p.add_argument('--'+x,required=True,type=Path)
    a=p.parse_args()
    run(a.bmp_directory,a.s39_private,a.s40_private,a.out)
if __name__=='__main__':cli()
