#!/usr/bin/env python3
"""S28 assess anchor-guided paths on software-injected lines over real H50Z 006 pixels.

Only the newly inserted line is known. The native photo remains unlabelled;
this is an exploratory external-form stress test, NOT physical reconstruction.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,cv2,torch
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from bubble_chamber.s28_anchor_tracker import candidate_paths,evidence_from_photo
from scripts.research.bubble_chamber.s26_visual_inspection import model_for_checkpoint
from bubble_chamber.s27_overlap_inference import overlap_predict_global_features
from scripts.research.bubble_chamber.s27_seam_crossing_injection import injected_long_track
from scripts.research.bubble_chamber.s27_overlap_photo_audit import sha256


def straight_candidates(evidence,seed,length=190,step=2,angles=24,top_k=4):
    rays=[]
    for j in range(angles):
        theta=2*np.pi*j/angles
        length_steps=np.arange(0,length+1,step,dtype=float)
        xs=seed[0]+length_steps*np.cos(theta);ys=seed[1]+length_steps*np.sin(theta)
        if np.min(xs)<0 or np.min(ys)<0 or np.max(xs)>=evidence.shape[1] or np.max(ys)>=evidence.shape[0]:continue
        pts=np.stack([xs,ys],axis=1)
        vals=cv2.remap(evidence,xs[None,:].astype('float32'),ys[None,:].astype('float32'),cv2.INTER_LINEAR)[0]
        rays.append({'direction_angle_deg':j*360/angles,'model_selection_score_UNCALIBRATED':float(vals.mean()),
                     'path_points_original_xy':pts.round(2).tolist()})
    return sorted(rays,key=lambda r:-r['model_selection_score_UNCALIBRATED'])[:top_k]


def recovery(paths,true_path:np.ndarray,prefix:int=195,hit_radius:float=3.0):
    ground=np.asarray(true_path)[:prefix]
    outcomes=[]
    for obj in paths:
        test=np.asarray(obj['path_points_original_xy'],float)
        if test.ndim!=2 or test.shape[1]!=2:raise ValueError('bad predicted centerline')
        precision=float((cKDTree(ground).query(test)[0]<=hit_radius).mean())
        coverage=float((cKDTree(test).query(ground)[0]<=hit_radius).mean())
        outcomes.append({'precision_against_synthetic_trace':precision,
                         'synthetic_prefix_coverage':coverage,
                         'mean_distance_to_injected_centerline_px':float(cKDTree(ground).query(test)[0].mean())})
    return {'top1':outcomes[0] if outcomes else None,
            'best_of_top4_oracle_NOT_DEPLOYABLE':max(outcomes,key=lambda x:x['synthetic_prefix_coverage']) if outcomes else None,
            'top1_direction_deg':paths[0]['direction_angle_deg'] if paths else None}


def run(args):
    torch.set_num_threads(2)
    source=json.loads(args.trials.read_text())
    if source.get('schema')!='s27_seam_crossing_synthetic_superimposed_on_genuine_H50Z_v1':
        raise ValueError('need precommitted long-gap synthetic trial seeds')
    model,prep=model_for_checkpoint(args.guided_az)
    per=[]
    cached_dir=args.out/'trial_receipts'
    cached_dir.mkdir(parents=True,exist_ok=True)
    for ii,row in enumerate(source['trials']):
        photo=args.crops/(row['crop_id']+'.png')
        if sha256(photo)!=row['real_photo_source_sha256']:raise ValueError('original H50Z crop digest changed')
        dest=cached_dir/f'S28_trial_{ii:02d}.json'
        if dest.is_file():
            cached=json.loads(dest.read_text())
            if cached.get('original_source_crop_sha256')!=sha256(photo) or cached.get('seed')!=row['seed'] or cached.get('original_source_crop')!=row['crop_id']:
                raise ValueError('stale cached truth source or model proposal')
            per.append(cached)
            continue
        original=cv2.imread(str(photo),cv2.IMREAD_GRAYSCALE)
        idx=sum(item['seed']==row['seed'] for item in source['trials'][:ii])
        im,fg,scratch,truth=injected_long_track(original,row['seed']*1000+idx)
        truth=np.asarray(truth,float)
        seed=tuple(truth[0])
        az=overlap_predict_global_features(im,model,prep,stride=32)
        classical_evidence=evidence_from_photo(im)
        mixed_evidence=evidence_from_photo(im,az)
        methods={'straight_evidence':straight_candidates(classical_evidence,seed),
                 'curvature_constrained_evidence':candidate_paths(classical_evidence,seed,length=190,width=48,top_k=4),
                 'original_AZ_plus_curvature':candidate_paths(mixed_evidence,seed,length=190,width=48,top_k=4)}
        result={'seed':row['seed'],'original_source_crop':row['crop_id'],
                'original_source_crop_sha256':sha256(photo),'native_pixel_tracks_labeled':0,
                'injected_trace_seed_xy':list(seed),'measured_methods':{m:recovery(c,truth) for m,c in methods.items()}}
        per.append(result)
        dest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        if ii in (0,1,8,9):
            img=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
            reference=np.rint(truth).astype(int).reshape(-1,1,2)
            cv2.polylines(img,[reference],False,(0,230,0),1,cv2.LINE_AA)
            for name,color in [('curvature_constrained_evidence',(80,210,255)),('original_AZ_plus_curvature',(230,80,220))]:
                if methods[name]:
                    path=np.rint(np.array(methods[name][0]['path_points_original_xy'])).astype(int).reshape(-1,1,2)
                    cv2.polylines(img,[path],False,color,1,cv2.LINE_AA)
            cv2.putText(img,'Green=SOFTWARE injected trace; cyan=classical, magenta=AZ-assisted',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
            args.out.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(args.out/f'S28_SYNTHETIC_NOT_REAL_TRUTH_{ii:02d}.jpg'),img)
        print('S28 synthetic',ii,row['crop_id'],[(m,round(v['top1']['synthetic_prefix_coverage'],3) if v['top1'] else None) for m,v in result['measured_methods'].items()],flush=True)
    aggregate={}
    for name in ('straight_evidence','curvature_constrained_evidence','original_AZ_plus_curvature'):
        for kind in ('top1','best_of_top4_oracle_NOT_DEPLOYABLE'):
            group=[q['measured_methods'][name][kind] for q in per if q['measured_methods'][name][kind]]
            aggregate.setdefault(name,{})[kind]={
                'n':len(group),'mean_precision_against_injected_trace':float(np.mean([r['precision_against_synthetic_trace'] for r in group])),
                'mean_injected_prefix_coverage':float(np.mean([r['synthetic_prefix_coverage'] for r in group])),
                'successful_at_70pct_prefix_coverage_count':int(sum(r['synthetic_prefix_coverage']>=.7 for r in group))}
    result={'schema':'s28_pixel_space_anchor_trace_control_on_generated_overlay_v1',
      'source_photo':'006','same_physical_H50Z_run':True,'n_exploratory_trials':len(per),
      'anchor_is_true_ONLY_FOR_SOFTWARE_INJECTED_LINE':True,'real_native_track_ground_truth_labels':0,
      'ANZA_source_model_weights_unchanged':True,'no_physical_particle_identities_inferred':True,
      'top4_oracle_requires_hidden_synthetic_truth_not_available_to_operator':True,
      'summary':aggregate,'per_trial':per}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'S28_anchored_synthetic_trace_assessment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    for k in ('trials','crops','guided_az','out'):p.add_argument('--'+k.replace('_','-'),type=Path,required=True)
    run(p.parse_args())

if __name__=='__main__':main()
