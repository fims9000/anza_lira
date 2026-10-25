#!/usr/bin/env python3
"""S29 frozen seed-offset + synthetic occlusion challenge on real H50Z photo background.

The synthetic injected trace is *known* for metric evaluation. Native photo
contents, source START operators, physical particle ID are still unlabelled.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from bubble_chamber.s28_anchor_tracker import candidate_paths,evidence_from_photo
from scripts.research.bubble_chamber.s27_seam_crossing_injection import injected_long_track

FIXED={
 'protocol':'S29_FROZEN_JITTER_AND_GAP_20261025',
 'injection_seed':20261031,
 'source_photo':'006',
 'jitter_y_px':[0,3,6,10],
 'erased_injected_trace_x_windows':[[140,155],[305,325]],
 'gap_condition':[False,True],
 'real_background_crops':8,
 'candidate_length_px':190,
 'candidate_half_width_px':48,
 'n_directions':24,
 'top_k':4,
 'hit_radius_px':3.,
 'evaluated_true_prefix_points':195,
 'success_coverage_threshold':.7,
 'score_calibration':'None',
 'physical_validation':'None',
 'model_selection_using_new_trial_truth':False,
}


def recovery(paths,true_path:np.ndarray):
    target=np.asarray(true_path,dtype=float)[:FIXED['evaluated_true_prefix_points']]
    answers=[]
    for o in paths:
        pred=np.asarray(o['path_points_original_xy'],dtype=float)
        distances=cKDTree(pred).query(target)[0]
        precision=cKDTree(target).query(pred)[0]
        answers.append({'prefix_coverage':float(np.mean(distances<=FIXED['hit_radius_px'])),
                        'candidate_precision':float(np.mean(precision<=FIXED['hit_radius_px']))})
    return {'top1':answers[0] if answers else None,
            'best_of_top4_oracle_NOT_OPERATOR_AVAILABLE':max(answers,key=lambda q:q['prefix_coverage']) if answers else None}


def run(source_trial_manifest:Path,source_rois:Path,out:Path):
    src=json.loads(Path(source_trial_manifest).read_text())
    if src.get('schema')!='s27_seam_crossing_synthetic_superimposed_on_genuine_H50Z_v1':
        raise ValueError('S27 source manifest required')
    source_rows=[]
    for r in src['trials']:
        if r['crop_id'] not in {x['crop_id'] for x in source_rows}:
            source_rows.append(r)
        if len(source_rows)==FIXED['real_background_crops']:break
    if len(source_rows)!=8:raise ValueError('Expected 8 preselected authentic H50Z-006 crops')
    per=[];out.mkdir(parents=True,exist_ok=True)
    for index,row in enumerate(source_rows):
        tile=Path(source_rois)/(row['crop_id']+'.png')
        if hashlib.sha256(tile.read_bytes()).hexdigest()!=row['real_photo_source_sha256']:
            raise ValueError('source tile changed after prior real-photo audit')
        bg=cv2.imread(str(tile),cv2.IMREAD_GRAYSCALE)
        if bg is None or bg.shape!=(512,512):raise ValueError('real background shape invalid')
        im,fg,scratch,truth=injected_long_track(bg,FIXED['injection_seed']*1000+index)
        truth=np.asarray(truth,dtype=float)
        for has_gap in FIXED['gap_condition']:
            observed=im.copy()
            if has_gap:
                erasure=fg.copy()
                xx=np.arange(bg.shape[1])[None,:]
                allowed=np.zeros_like(erasure,dtype=bool)
                for lo,hi in FIXED['erased_injected_trace_x_windows']:
                    allowed|=(xx>=lo)&(xx<=hi)
                erasure&=allowed
                observed[erasure]=bg[erasure]
            evidence=evidence_from_photo(observed)
            for offset in FIXED['jitter_y_px']:
                seed_xy=(float(truth[0,0]),float(truth[0,1]+offset))
                proposed=candidate_paths(evidence,seed_xy,length=190,width=48,step=2,angles=24,top_k=4)
                result=recovery(proposed,truth)
                per.append({'crop_id':row['crop_id'],'source_sha256':row['real_photo_source_sha256'],
                            'seed_perturbation_vertical_px':offset,'software_gap_injected':has_gap,
                            'model_feature':'fixed_classical_photometry_steerable',
                            'known_truth_only_software_injected_pixels':True,
                            'top1':result['top1'],'best_of_top4_oracle_NOT_OPERATOR_AVAILABLE':result['best_of_top4_oracle_NOT_OPERATOR_AVAILABLE']})
        print('S29 crop',index,row['crop_id'],'done',flush=True)
    groups={}
    for gap in FIXED['gap_condition']:
        for off in FIXED['jitter_y_px']:
            rows=[r for r in per if r['software_gap_injected']==gap and r['seed_perturbation_vertical_px']==off]
            for kind in ('top1','best_of_top4_oracle_NOT_OPERATOR_AVAILABLE'):
                scores=[r[kind] for r in rows if r[kind]]
                groups[f'gap{int(gap)}_offset{off}_{kind}']={
                    'n':len(scores),
                    'mean_prefix_coverage':float(np.mean([q['prefix_coverage'] for q in scores])),
                    'successes_70pct':sum(q['prefix_coverage']>=.7 for q in scores),
                    'mean_precision':float(np.mean([q['candidate_precision'] for q in scores]))}
    receipt={'schema':'s29_true_operator_seed_jitter_simulation_on_real_background_v1',
             'frozen_protocol':FIXED,'source_manifest_sha256':hashlib.sha256(Path(source_trial_manifest).read_bytes()).hexdigest(),
             'measured_real_photo_start_localization_errors':'UNKNOWN',
             'real_native_track_pixel_labels':0,'independent_physical_runs':0,
             'number_of_software_injected_trials':len(source_rows),
             'backgrounds_from_original_photo':len({r['crop_id'] for r in source_rows}),
             'conditions_evaluated':len(per),'summary':groups,'per_condition':per,
             'physical_use':'PROVISIONAL_REVIEW_ONLY'}
    output=out/'S29_seed_jitter_gap_stress.json'
    output.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-trials',type=Path,required=True)
    p.add_argument('--crops',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();r=run(args.source_trials,args.crops,args.out)
    for name,stat in r['summary'].items():
        if name.endswith('_top1'):print(name,stat)

if __name__=='__main__':main()
