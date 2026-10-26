#!/usr/bin/env python3
"""S46 frozen synthetic single-view vs registered-repeat negative controls.

All curves are computer-injected. A repeated stroke could physically be a
track in both scans: this test verifies the cue, not track/scratch identity.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import argparse,json,sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s44_crossframe_persistence import (
    ExactBMP8, fit_common_fiducials, warp_corresponding_window,
    compare_registered_ridges)
from bubble_chamber.s40_fiducial_safety import fiducial_guard
from bubble_chamber.s45_registered_unknown import repeat_advisory,SCHEMA

SEED=20261046
SCENARIOS=('single_view_only','identical_registered_repeat','normal_shifted_eight_px_repeat')


def generate_path(rng:np.random.Generator,zones:list,origin:list,photo:str)->np.ndarray:
    for _ in range(250):
        theta=float(rng.uniform(-.45,.45))
        cx=float(rng.uniform(195,310));cy=float(rng.uniform(160,350))
        t=np.linspace(-136,136,100)
        curv=float(rng.uniform(-.002,.002))
        x=cx+t*np.cos(theta)-curv*t*t*np.sin(theta)
        y=cy+t*np.sin(theta)+curv*t*t*np.cos(theta)
        p=np.column_stack([x,y])
        if (p.min()>=45 and p.max()<=460 and not
                fiducial_guard(p,origin,photo,zones)['fiducial_uncertainty_any']):
            return p
    raise ValueError('No fiducial-safe synthetic path found in preregistered search')


def perpendicular_shift(path:np.ndarray, pixels:float=8.)->np.ndarray:
    path=np.asarray(path,dtype=float)
    if path.ndim!=2 or path.shape[1]!=2 or len(path)<4:
        raise ValueError('Bad synthetic normal-shift geometry')
    tangent=np.gradient(path,axis=0)
    length=np.maximum(np.linalg.norm(tangent,axis=1),1e-9)
    normal=np.column_stack((-tangent[:,1],tangent[:,0]))/length[:,None]
    return path+pixels*normal


def inject(gray:np.ndarray,path:np.ndarray,*,shift=(0,0),color=32)->np.ndarray:
    if gray.dtype!=np.uint8 or gray.ndim!=2 or len(path)<10:
        raise ValueError('Wrong injection photo/path')
    out=gray.copy()
    pts=np.rint(path+np.asarray(shift,dtype=float)).astype('int32')
    cv2.polylines(out,[pts],False,int(color),2,lineType=cv2.LINE_AA)
    return out


def challenge_once(src:np.ndarray,paired:np.ndarray,path:np.ndarray,scenario:str)->dict:
    if scenario not in SCENARIOS:raise ValueError('Unregistered scenario')
    s=inject(src,path,color=32)
    other=paired.copy()
    if scenario=='identical_registered_repeat':
        other=inject(paired,path,color=32)
    elif scenario=='normal_shifted_eight_px_repeat':
        other=inject(paired,perpendicular_shift(path,8.),color=32)
    report=compare_registered_ridges(s,other,[{
        'draft_id':'s46_SYNTHETIC_COUNTERFACTUAL_NOT_GT','points_xy':path.tolist()}])[0]
    report['known_fiducial_uncertainty_NOT_NATIVE_TRUTH']=False
    risk=repeat_advisory(report)
    return {'scenario':scenario,'registered_repeat_advisory':risk['registered_repeat_risk'],
            'source_response':report['source_multiscale_ridge_coverage'],
            'other_response':report['counterpart_same_registered_path_coverage'],
            'shifted_control_response':report['displaced_same_shape_counterpart_control_coverage'],
            'known_synthetic_geometry_not_native_GT':True}


def run(bmpdir:Path,s39_file:Path,s40_file:Path,out:Path)->dict:
    out.mkdir(parents=True,exist_ok=True)
    s39=json.loads(s39_file.read_text())
    fid=json.loads(s40_file.read_text())['fiducials_PRIVATE']
    if len(s39.get('tiles',[]))!=39 or len(fid)!=15:
        raise ValueError('Original machine source changed')
    reg=fit_common_fiducials(json.loads((bmpdir/'labels.json').read_text()))
    forward={'005':reg['forward_005_to_006_PRIVATE'],
             '006':reg['inverse_006_to_005_PRIVATE']}
    images={k:ExactBMP8(bmpdir/f'H50Z_030_01_{k}.bmp') for k in ('005','006')}
    rng=np.random.default_rng(SEED)
    private=[]
    by_photo={k:{n:[] for n in SCENARIOS} for k in images}
    for i,tile in enumerate(s39['tiles']):
        photo=tile['photo_id'];other='006' if photo=='005' else '005'
        origin=tile['native_origin_xy_PRIVATE']
        src=images[photo].crop(*origin,512,512)
        if sha256(src.tobytes()).hexdigest()!=tile['source_gray_sha256']:
            raise ValueError('Wrong source image, synthetic control aborted')
        aligned=warp_corresponding_window(images[other],origin,forward[photo])
        path=generate_path(rng,fid,origin,photo)
        results=[challenge_once(src,aligned,path,scenario) for scenario in SCENARIOS]
        for r in results:by_photo[photo][r['scenario']].append(r['registered_repeat_advisory'])
        private.append({'crop_id':tile['crop_id'],'photo':photo,
                        'synthetic_only_path_PRIVATE':np.round(path,2).tolist(),
                        'scenario_results_NOT_NATIVE':results})
        if (i+1)%8==0:print('S46 synthetic control',i+1,'/',39,flush=True)
    summary={}
    for key,vals in by_photo.items():
        summary[key]={}
        for scene,hits in vals.items():
            a=np.asarray(hits,dtype=float)
            rs=np.random.default_rng(SEED+int(key)+len(scene))
            idx=rs.integers(0,len(a),(3000,len(a)))
            summary[key][scene]={'tested_synthetic_scenes':len(hits),
                                'registered_repeat_trigger_fraction':round(float(a.mean()),5),
                                'crop_resample_95pct_CI_not_run_independence':
                                    np.quantile(a[idx].mean(1),[.025,.975]).round(5).tolist()}
    report={'schema':'s46_corrected_normal_offset_registered_control_v2',
            'source_photos':2,'independent_physics_runs':1,'native_physics_GT_count':0,
            'simulated_original_native_background_tiles':39,
            'synthetic_cases_per_tile':len(SCENARIOS),
            'synthetic_cases_total':39*len(SCENARIOS),
            'random_seed_frozen':SEED,
            'line_geometry_identical_between_single_and_repeat_conditions':True,
            'normal_offset_corrects_prior_incorrect_x_axis_shift':True,
            'line_identity':'ARTIFICIALLY_INJECTED_STROKE_NOT_PHYSICAL_TRACK',
            'comparisons_by_photo':summary,
            'S45_fixed_threshold_not_refit_on_this_trial':True,
            'this_test_does_not_identify_native_scratch_or_particle':True,
            'native_machine_candidates_removed':0,
            'manual_labels_needed':0}
    (out/'S46_PUBLIC_SYNTHETIC_COUNTERFACTUALS.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'S46_PRIVATE_SYNTHETIC_PATHS.json').write_text(json.dumps(private,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    for a in ('bmp-directory','s39-private','s40-private','out'):
        p.add_argument('--'+a,required=True,type=Path)
    a=p.parse_args()
    run(a.bmp_directory,a.s39_private,a.s40_private,a.out)
if __name__=='__main__':main()
