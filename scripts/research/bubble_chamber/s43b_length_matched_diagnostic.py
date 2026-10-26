#!/usr/bin/env python3
"""S43B: post-hoc length/renderer matched injection negative control."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_simulated_gaps import extract_track_templates,make_scene
from bubble_chamber.s39_machine_auto import ridge_evidence
from bubble_chamber.s43_artifact_triage import path_features
from scripts.research.bubble_chamber.s43_machine_artifact_trial import (
    frozen_frame_folds, select_calibration_threshold, _stroke,
)
SEEDS={'005':202610459,'006':202610460}
COUNTS={'train':16,'calibration':8,'test':16}


def counterfactual_scene(scene, rng):
    """Copy 2D measured-fragment geometry to synthetic scratch origins.

    Their 2D appearance can be made indistinguishable in principle. This tests
    the fragility of source-origin classification, NOT real scratch statistics.
    """
    canvas=scene.gray.astype('float32')
    originals=[*scene.left,*scene.right[:2]]
    scratch=[]
    for idx, pts in enumerate(originals):
        q=pts.copy()
        q[:,1]+=float(rng.choice([-1,1])*rng.uniform(20,27))
        q[:,0]+=float(rng.uniform(-3,3))
        q=np.clip(q,7,184)
        # Same ink/width family as original injection. Candidate centerline is
        # slightly displaced from the rendered stroke, as in S35 noise model.
        ink=float(rng.uniform(27,75));width=int(rng.integers(1,4))
        _stroke(canvas,q,ink=ink,width=width)
        noisy=q.copy();noisy[:,1]+=rng.normal(0,1.1,size=len(q))
        noisy=np.clip(noisy,7,184)
        scratch.append(('counterfactual_geometry_clone',noisy))
    # Equal-length straight scratch with a source-fragment-matched arc extent.
    ref=originals[0]
    u=np.linspace(0,1,len(ref))
    le=float(np.sum(np.linalg.norm(np.diff(ref,axis=0),axis=1)))
    x0=float(rng.uniform(16,70));y0=float(rng.uniform(35,150))
    direction=float(rng.uniform(-.45,.45))
    straight=np.column_stack((x0+le*np.cos(direction)*u,
                              y0+le*np.sin(direction)*u))
    straight=np.clip(straight,7,184)
    _stroke(canvas,straight,ink=float(rng.uniform(27,75)),width=int(rng.integers(1,4)))
    scratch.append(('length_matched_straight',straight))
    scratch.append(('original_s35_matched',scene.right[2]))
    rendered=cv2.GaussianBlur(np.clip(canvas,0,255).astype('uint8'),(0,0),.35)
    return rendered, originals, scratch


def dataset(manifest,folds,tiles,photo,fold,seed):
    sources=sorted(tiles.glob(f'H50Z_030_01_{photo}_*.png'))
    if len(sources)!=(21 if photo=='005' else 18):raise ValueError('Wrong photo tiles')
    templates=extract_track_templates(manifest,chosen_frames=folds[fold])
    rng=np.random.default_rng(seed)
    x=[];y=[];family=[];scene_ids=[];lengths=[]
    for i in range(COUNTS[fold]):
        tile=sources[int(rng.integers(len(sources)))]
        source=make_scene(tile,templates,seed=int(rng.integers(0,2**31)),
                          scenario=('parallel','crossing','scratched','empty_gap')[i%4])
        if not set(source.source_frame_ids)<=folds[fold]:raise AssertionError('Source leakage')
        gray,true,neg=counterfactual_scene(source,rng)
        maps=ridge_evidence(gray)
        for cls,records in ((1,[('measured_curve_simulation',v) for v in true]),(0,neg)):
            for name,pts in records:
                feat=path_features(gray,pts,maps=maps)
                x.append(feat);y.append(cls);family.append(name);scene_ids.append(i)
                lengths.append(float(np.expm1(feat[0])))
    return {'X':np.asarray(x),'y':np.asarray(y),'family':np.array(family),
            'scene':np.asarray(scene_ids),'length':np.asarray(lengths)}


def experiment(manifest,folds,tiles,train_photo,test_photo):
    base=SEEDS[train_photo]
    d={fold:dataset(manifest,folds,tiles,train_photo if fold!='test' else test_photo,
                    fold,base+off) for fold,off in (('train',11),('calibration',22),('test',33))}
    out={}
    for variant,indices in [('all_features',np.arange(d['train']['X'].shape[1])),
                            ('without_length_and_sinuosity',np.arange(2,d['train']['X'].shape[1]))]:
        clf=make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0,class_weight='balanced',max_iter=1000,random_state=base))
        clf.fit(d['train']['X'][:,indices],d['train']['y'])
        cal=clf.predict_proba(d['calibration']['X'][:,indices])[:,1]
        th=select_calibration_threshold(cal,d['calibration']['y']==0)
        test=clf.predict_proba(d['test']['X'][:,indices])[:,1]
        families={name:round(float(np.mean(test[(d['test']['y']==0)&(d['test']['family']==name)]>th)),6)
                  for name in sorted(set(d['test']['family'][d['test']['y']==0]))}
        out[variant]={'posthoc_synthetic_origin_AUROC':round(float(roc_auc_score(d['test']['y'],test)),6),
                      'simulated_true_acceptance':round(float(np.mean(test[d['test']['y']==1]>th)),6),
                      'simulated_false_origin_acceptance':round(float(np.mean(test[d['test']['y']==0]>th)),6),
                      'scratch_by_family_acceptance':families,
                      'test_scenes':COUNTS['test'], 'source_training_photo':train_photo,
                      'independent_physical_runs':1}
    data=d['test']
    out['confound_audit']={key:round(float(np.median(data['length'][data['family']==key])),3)
                           for key in sorted(set(data['family']))}
    return out


def run(visualmetro,tiles,out):
    manifest=audited_large_manifest(visualmetro)
    folds=frozen_frame_folds(manifest)
    a=experiment(manifest,folds,tiles,'005','006')
    print('S43B 005 -> 006 done',flush=True)
    b=experiment(manifest,folds,tiles,'006','005')
    print('S43B 006 -> 005 done',flush=True)
    report={'schema':'s43b_posthoc_length_matched_counterfactual_v1',
      'status':'POSTHOC_SENSITIVITY_NOT_INDEPENDENT_CONFIRMATION',
      'frame_folds':{k:len(v) for k,v in folds.items()},
      'source_physical_runs':1,'native_GT_line_count':0,'native_physics_autojoin_allowed':False,
      'experimental_scenes_per_direction':COUNTS,
      'results':{'005_train_006_test':a,'006_train_005_test':b},
      'warning':'Counterfactual origin is chosen by simulator. Equal-looking 2D traces can have arbitrary origin; this cannot validate native particle identity.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run(a.visualmetro,a.photo_tiles,a.output)
    print(json.dumps(r['results'],indent=2))
if __name__=='__main__':main()
