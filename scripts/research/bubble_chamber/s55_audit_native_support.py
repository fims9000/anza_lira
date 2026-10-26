#!/usr/bin/env python3
"""Compare S54/S55 simulated-feature support for unlabelled H50Z proposals.

No physical track classifications; this is a conditional simulator coverage
check. Both H50Z photos remain one physical run.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s39_machine_auto import ridge_evidence
from bubble_chamber.s43_artifact_triage import path_features
from bubble_chamber.s53_bead_texture import texture_features,merge_features
from bubble_chamber.s54_context_ood import source_context_crop
from scripts.research.bubble_chamber.s55_native_shape_simulator import (
    SCHEMA,PROTOCOL,split_unlabeled_source_geometries,create_examples,
)


def inference(x:np.ndarray,model:dict,*,source:str,target:str):
    a=np.asarray(x,dtype=float)
    mu=np.asarray(model['mean'],dtype=float)
    sd=np.asarray(model['sd'],dtype=float)
    weight=np.asarray(model['weights'],dtype=float)
    if (model.get('schema')!=SCHEMA or model.get('trained_on_source_photo')!=source
        or model.get('target_photo')!=target or source==target
        or model.get('native_true_track_labels')!=0
        or model.get('can_autojoin_native_tracks') is not False
        or a.ndim!=2 or a.shape[1]!=len(mu) or
        any(v.shape!=mu.shape for v in (sd,weight))
        or len(model.get('feature_names',[]))!=len(mu)
        or np.any(sd<=0) or not np.isfinite(a).all()
        or not np.isfinite([*mu,*sd,*weight,model['intercept']]).all()):
        raise ValueError('Unsafe S55 simulator model or feature schema')
    standardized=(a-mu)/sd
    return expit(standardized@weight+float(model['intercept'])),\
           np.max(np.abs(standardized),axis=1)


def run(s39:Path,s54_private:Path,images:Path,models_root:Path,
        private_out:Path,public_out:Path):
    source=json.loads(s39.read_text())
    old=json.loads(s54_private.read_text())
    if (len(source.get('tiles',[]))!=39 or len(old.get('tiles',[]))!=39
        or old.get('native_expert_GT_count')!=0 or
        old.get('native_physical_auto_joins')!=0):
        raise ValueError('Untrusted native H50Z provenance')
    folds=split_unlabeled_source_geometries(source)
    config={}
    for source_photo,target,seed in [('005','006',20265503),('006','005',20265513)]:
        p=models_root/f'{source_photo}_to_{target}_{seed}'
        model=json.loads((p/'S43_plus_bead_texture_S53_native_shape_simulator_only.json').read_text())
        cal=create_examples(folds[source_photo]['cal'],images,photo=source_photo,
                      n=PROTOCOL['cal_scene_count'],seed=PROTOCOL['seeds_cal'][source_photo],
                      test=False)
        x=merge_features(cal['X43'],cal['X53'])
        _,z=inference(x,model,source=source_photo,target=target)
        cutoff=float(np.quantile(z,.99))
        config[target]=(model,cutoff)
    rows=[];counts=defaultdict(Counter);old_counts=defaultdict(Counter);values=defaultdict(list)
    for t,t_old in zip(source['tiles'],old['tiles']):
        cid=t['crop_id'];photo=t['photo_id']
        if cid!=t_old['crop_id'] or photo!=t_old['photo']:
            raise ValueError('Wrong native H50Z machine IDs')
        gray=cv2.imread(str(images/(cid+'.png')),cv2.IMREAD_GRAYSCALE)
        if gray is None or sha256(gray.tobytes()).hexdigest()!=t['source_gray_sha256']:
            raise ValueError('Corrupt native photo bytes')
        model,cutoff=config[photo]
        if len(t['proposals'])!=len(t_old['candidates']):
            raise ValueError('Native proposal disappeared')
        crop_rows=[]
        for c,reference in zip(t['proposals'],t_old['candidates']):
            if c['draft_id']!=reference['draft_id']:
                raise ValueError('Misordered native machine proposal')
            crop=source_context_crop(gray,c['points_xy'])
            if crop is None:
                result={'draft_id':c['draft_id'],'status':'UNKNOWN_UNSCORABLE_SOURCE_CONTEXT'}
            else:
                roi,points,_=crop
                x43=path_features(roi,points,maps=ridge_evidence(roi))[None,:]
                x=merge_features(x43,texture_features(roi,points)[None,:])
                s,z=inference(x,model,source=('006' if photo=='005' else '005'),target=photo)
                ood=bool(z[0]>cutoff)
                result={'draft_id':c['draft_id'],
                        'status':'UNKNOWN_OUTSIDE_SOURCE_SIMULATOR_SUPPORT' if ood else 'UNKNOWN_WITHIN_SIMULATOR_SUPPORT_NOT_PHYSICS_TRUTH',
                        'score_S55_SIMULATOR_ORIGIN_NOT_PHYSICS_PROB':round(float(s[0]),6),
                        'max_standardized_feature_deviation':round(float(z[0]),5),
                        'source_calibration_q99_max_z':round(cutoff,5)}
                values[photo].append(float(s[0]))
            result['native_track_identity']='UNKNOWN'
            result['candidate_preserved']=True
            result['physical_auto_join_authorized']=False
            counts[photo][result['status']]+=1
            old_counts[photo][reference['status']]+=1
            crop_rows.append(result)
        rows.append({'crop_id':cid,'photo':photo,'candidate_advisories':crop_rows})
    if sum(sum(c.values()) for c in counts.values())!=567:
        raise AssertionError('Original native candidate preservation failed')
    private={'schema':SCHEMA,'all_native_candidate_lines_preserved':567,
        'native_expert_GT_count':0,'physical_auto_joins':0,'calibration_split_only':True,
        'no_source_native_labels_used':True,'tiles':rows}
    private_out.parent.mkdir(parents=True,exist_ok=True)
    private_out.write_text(json.dumps(private,indent=2,ensure_ascii=False)+'\n')
    public={'schema':SCHEMA,'scientific_status':'posthoc simulator coverage comparison NOT native particle validation',
            'same_physical_H50Z_run':True,'native_particle_ground_truth':0,
            'physical_auto_joins':0,'all_567_original_machine_candidates_preserved':True,
            'target_photo_never_used_for_model_training_or_calibration':True,
            'source_native_geometry_was_unlabelled':True,
            'source_simulator_calibrated_q99_not_validated_native_coverage':True,
            'counts':{}}
    for photo in ('005','006'):
        c=counts[photo];before=old_counts[photo];a=values[photo]
        public['counts'][photo]={
            'total':int(sum(c.values())),
            'unscorable_192px_context':int(c['UNKNOWN_UNSCORABLE_SOURCE_CONTEXT']),
            'S54_previous_outside_simulator_support':int(before['UNKNOWN_OUTSIDE_CALIBRATION_SUPPORT']),
            'S55_new_outside_simulator_support':int(c['UNKNOWN_OUTSIDE_SOURCE_SIMULATOR_SUPPORT']),
            'S55_new_within_simulator_support_NOT_true_physics':int(c['UNKNOWN_WITHIN_SIMULATOR_SUPPORT_NOT_PHYSICS_TRUTH']),
            'S55_simulator_score_median_NOT_NATIVE_PROB':float(np.median(a)),
        }
    public_out.parent.mkdir(parents=True,exist_ok=True)
    public_out.write_text(json.dumps(public,indent=2,ensure_ascii=False)+'\n')
    return public


def main():
    p=argparse.ArgumentParser()
    for key in ('s39_private','s54_private','images','models','private_out','public_out'):
        p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.s39_private,a.s54_private,a.images,a.models,
                         a.private_out,a.public_out),indent=2),flush=True)
if __name__=='__main__':main()
