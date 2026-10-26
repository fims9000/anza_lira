#!/usr/bin/env python3
"""S55: synthetic renderers on SOURCE-ONLY unlabeled machine-candidate geometry.

Synthetic labels are RENDERER ORIGINS, not real particle identity. This study
fixes length/turn simulator support; it cannot validate real-camera physics.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s39_machine_auto import ridge_evidence
from bubble_chamber.s43_artifact_triage import path_features
from bubble_chamber.s53_bead_texture import texture_features,merge_features,SCHEMA as S53_SCHEMA
from bubble_chamber.s53_native_texture_advisory import _linear_predict
from bubble_chamber.s54_context_ood import calibration_ood_threshold
from scripts.research.bubble_chamber.s53_benchmark_bead_texture import (
    PROTOCOL as S53_PROTOCOL, RENDERERS, renderer_mask, evaluate, _paired_bootstrap_delta,
)

SCHEMA='s55_machine_candidate_geometry_simulator_v1'
PROTOCOL={'train_scene_count':40,'cal_scene_count':20,'test_scene_count_per_seed':40,
 'seeds_train':{'005':20265501,'006':20265511},
 'seeds_cal':{'005':20265502,'006':20265512},
 'seeds_test':{'005':[20265503,20265504],'006':[20265513,20265514]},
 'test_source_photo_heldout_for_model_fit':True,
 'machine_source_cropped_and_translated_NOT_resized':True,
 'native_particle_truth_count':0}


def split_unlabeled_source_geometries(source:dict):
    if (source.get('native_expert_gt_created')!=0
        or source.get('auto_physical_join_allowed') is not False
        or len(source.get('tiles',[]))!=39):
        raise ValueError('Untrusted H50Z machine candidate source')
    all_by_photo={'005':[],'006':[]}
    for t in source['tiles']:
        photo=t['photo_id'];cid=t['crop_id']
        if photo not in all_by_photo:raise ValueError('Foreign H50Z source')
        for c in t['proposals']:
            pts=np.asarray(c['points_xy'],dtype=float)
            if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<3 or not np.isfinite(pts).all():
                raise ValueError('Invalid machine-proposed shape')
            key=cid+':'+c['draft_id']
            all_by_photo[photo].append({'id':key,'crop_id':cid,'points':pts})
    if [len(all_by_photo[k]) for k in ('005','006')] != [321,246]:
        raise ValueError('Source machine geometry count changed')
    folds={}
    for photo,entries in all_by_photo.items():
        rows=sorted(entries,key=lambda c:sha256(c['id'].encode()).hexdigest())
        split=int(.8*len(rows))
        folds[photo]={'train':rows[:split],'cal':rows[split:],'test':rows}
        assert not ({a['id'] for a in rows[:split]} & {a['id'] for a in rows[split:]})
    return folds


def translate_measured_machine_shape(raw_points:np.ndarray, rng:np.random.Generator,
                                     *,size:int=192) -> np.ndarray:
    """Cut original arc-length path, translate and interpolate; never rescale."""
    p=np.asarray(raw_points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all():
        raise ValueError('Invalid unlabelled source polyline')
    steps=np.linalg.norm(np.diff(p,axis=0),axis=1)
    valid=np.r_[True, steps>1e-6];p=p[valid]
    if len(p)<3:raise ValueError('Degenerate native candidate path')
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    total=float(arc[-1])
    if total<12:raise ValueError('Candidate too short for S55 generator')
    for _ in range(8):
        extent=float(min(total, rng.uniform(30.,151.)))
        start=float(rng.uniform(0.,max(.001,total-extent)))
        want=np.linspace(start,start+extent,max(30,int(extent/1.3)))
        shifted=np.column_stack((np.interp(want,arc,p[:,0]),np.interp(want,arc,p[:,1])))
        bbox=shifted.max(axis=0)-shifted.min(axis=0)
        if np.all(bbox<=size-35):
            # Centering is strictly a translation, no rotation or rescaling.
            pos=shifted+(np.array([size/2,size/2])-(shifted.min(0)+shifted.max(0))/2)
            if np.any(pos<13.) or np.any(pos>size-13.):
                raise AssertionError('Native shape escaped safe 192px window')
            return pos
    raise ValueError('Highly curled machine candidate cannot fit 192px without scaling')



def render_short_native_shapes(background:np.ndarray,curve:np.ndarray,*,seed:int,test:bool):
    """S53 identical renderer families with a lower small-fragment mass floor.

    S53 fixed 110-145px curves required >=70 opacity pixels; real S39
    fragments are often 18-60px, so that floor is not scientifically valid
    for S55. Nominal opacity matching is still preserved in every pair.
    """
    if background.shape!=(192,192) or background.dtype!=np.uint8:
        raise ValueError('Unchanged source 192px grayscale required')
    rng=np.random.default_rng(seed)
    ink=float(rng.uniform(22,54))
    kinds=RENDERERS if test else RENDERERS[:4]
    masks={k:renderer_mask(k,curve,rng,background.shape) for k in kinds}
    ref=float(masks['simulated_bead_track'].sum())
    if ref<9 or any(float(m.sum())<4 for m in masks.values()):
        raise ValueError('Sparse synthetic renderer cannot be compared')
    return {k:cv2.GaussianBlur(np.clip(background.astype('float32')-ink*m*(ref/float(m.sum())),
                              0,255).astype('uint8'),(0,0),.35) for k,m in masks.items()}


def create_examples(rows:list[dict],photos:Path,*,photo:str,n:int,
                    seed:int,test:bool,preview:bool=False)->dict:
    if photo not in ('005','006') or len(rows)<20 or n<1:
        raise ValueError('Invalid source split of machine geometries')
    files=sorted(photos.glob(f'H50Z_030_01_{photo}_*.png'))
    if len(files)!=(21 if photo=='005' else 18):
        raise ValueError('Unexpected real H50Z source photo tiles')
    rng=np.random.default_rng(seed)
    feats_old=[];feats_new=[];labels=[];family=[];scene_ids=[];previews=[]
    for scene_id in range(n):
        source=None;shape=None
        for attempt in range(30):
            candidate=rows[int(rng.integers(len(rows)))]
            try:shape=translate_measured_machine_shape(candidate['points'],rng)
            except ValueError:continue
            source=candidate
            break
        if shape is None:raise ValueError('Source machine geometries do not fit generator')
        backgrounds=[p for p in files if p.stem!=source['crop_id']]
        file=backgrounds[int(rng.integers(len(backgrounds)))]
        raw=cv2.imread(str(file),cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape!=(512,512):raise ValueError('Bad H50Z source crop')
        x=int(rng.integers(0,321));y=int(rng.integers(0,321))
        background=raw[y:y+192,x:x+192]
        cases=render_short_native_shapes(background,shape,seed=int(rng.integers(0,2**31)),
                                  test=test)
        for kind,img in cases.items():
            maps=ridge_evidence(img)
            feats_old.append(path_features(img,shape,maps=maps))
            feats_new.append(texture_features(img,shape))
            labels.append(int(kind=='simulated_bead_track'))
            family.append(kind);scene_ids.append(scene_id)
        if preview and scene_id<3:
            previews.append({'scene_id':scene_id,'images':cases})
    return {'X43':np.asarray(feats_old),'X53':np.asarray(feats_new),
       'labels':np.asarray(labels),'families':np.asarray(family),
       'scenes':np.asarray(scene_ids),'previews':previews}


def original_frozen_predict(ckpt:dict,test:dict,source_photo:str,target_photo:str)->np.ndarray:
    x=merge_features(test['X43'],test['X53'])
    return _linear_predict(x,ckpt,source_photo,target_photo,True)


def run(s39:Path,tiles:Path,original_model_root:Path,out:Path,private_dir:Path|None=None,
        *,train_n:int=40,cal_n:int=20,test_n:int=40,bootstrap:int=500):
    if min(train_n,cal_n,test_n)<5:raise ValueError('Insufficient synthetic cases')
    folds=split_unlabeled_source_geometries(json.loads(s39.read_text()))
    if private_dir:private_dir.mkdir(parents=True,exist_ok=True)
    report={'schema':SCHEMA,'synthetic_positive_truth':'rendered_bead_stroke_only',
      'native_H50Z_expert_track_GT_count':0,'native_physical_auto_joins':0,
      'unlabeled_machine_geometries_are_not_particles':True,
      'test_photo_disjoint_from_all_model_fit_geometries':True,
      'same_physical_run':True,
      'frozen_protocol':dict(PROTOCOL,train_scene_count=train_n,cal_scene_count=cal_n,
                             test_scene_count_per_seed=test_n),
      'source_unlabeled_geometry_counts':{photo:{split:len(v) for split,v in arms.items()}
                                         for photo,arms in folds.items()},
      'directions':{}}
    for source,target,oldseed in [('005','006',20265303),('006','005',20265313)]:
        train=create_examples(folds[source]['train'],tiles,photo=source,n=train_n,
                              seed=PROTOCOL['seeds_train'][source],test=False)
        cal=create_examples(folds[source]['cal'],tiles,photo=source,n=cal_n,
                            seed=PROTOCOL['seeds_cal'][source],test=False)
        old=json.loads((original_model_root/f'{source}_to_{target}_{oldseed}'/
          'S43_plus_bead_texture_S53_simulator_only.json').read_text())
        result={'train_photo':source,'test_photo':target,'tests':{},
                'machine_source_training_shapes':len(folds[source]['train']),
                'machine_source_calibration_shapes':len(folds[source]['cal'])}
        for seed in PROTOCOL['seeds_test'][source]:
            test=create_examples(folds[target]['test'],tiles,photo=target,n=test_n,
                                seed=seed,test=True,preview=bool(private_dir))
            metrics,models=evaluate(train,cal,test,seed=PROTOCOL['seeds_train'][source],
                                    iterations=bootstrap)
            new=metrics['S43_plus_bead_texture_S53']
            old_sc=original_frozen_predict(old,test,source,target)
            new_sc=models['S43_plus_bead_texture_S53'].predict_proba(
                merge_features(test['X43'],test['X53']))[:,1]
            labels=test['labels'];positive=labels==1
            clone=test['families']=='same_bead_renderer_scratch'
            old_stats={'overall_AUROC_FROZEN_S53_on_NEW_TEST_simulator_only':float(roc_auc_score(labels,old_sc)),
               'clone_only_AUROC_SYNTHETIC':float(roc_auc_score(labels[positive|clone],old_sc[positive|clone]))}
            metrics['frozen_old_S53_no_refit']=old_stats
            metrics['S43_plus_bead_texture_S53']['paired_scene_bootstrap_delta_vs_old_S53']=(
                _paired_bootstrap_delta(test,old_sc,new_sc,seed=seed+155,iterations=bootstrap))
            result['tests'][str(seed)]={'arms':metrics,'test_scenes':test_n}
            print('S55',source,'->',target,seed,'old frozen S53',round(old_stats['overall_AUROC_FROZEN_S53_on_NEW_TEST_simulator_only'],4),
                  'new machine-shape S55',round(new['overall_AUROC_SYNTHETIC_RENDERER_ONLY'],4),
                  'clone',round(new['matched_bead_clone_only_AUROC_SYNTHETIC_RENDERER_ONLY'],4),flush=True)
            if private_dir:
                folder=private_dir/f'{source}_to_{target}_{seed}'
                folder.mkdir(parents=True,exist_ok=True)
                for case in test['previews']:
                    for name,img in case['images'].items():
                        cv2.imwrite(str(folder/f"scene_{case['scene_id']:02d}_{name}.png"),img)
                for name,model in models.items():
                    sc=model.named_steps['standardscaler'];clf=model.named_steps['logisticregression']
                    ck={'schema':SCHEMA,'arm':name,'trained_on_source_photo':source,
                         'target_photo':target,'native_true_track_labels':0,
                         'can_autojoin_native_tracks':False,
                         'feature_names':list(__import__('bubble_chamber.s43_artifact_triage',fromlist=['FEATURE_NAMES']).FEATURE_NAMES)+
                           (list(__import__('bubble_chamber.s53_bead_texture',fromlist=['FEATURE_NAMES']).FEATURE_NAMES) if 'S53' in name else []),
                         'mean':sc.mean_.tolist(),'sd':sc.scale_.tolist(),
                         'weights':clf.coef_[0].tolist(),'intercept':float(clf.intercept_[0])}
                    (folder/(name+'_native_shape_simulator_only.json')).write_text(json.dumps(ck,indent=2)+'\n')
        report['directions'][f'{source}_to_{target}']=result
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--s39-private',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--frozen-s53-models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--private-preview-dir',type=Path)
    p.add_argument('--train-scenes',type=int,default=40)
    p.add_argument('--cal-scenes',type=int,default=20)
    p.add_argument('--test-scenes',type=int,default=40)
    p.add_argument('--bootstrap',type=int,default=500)
    a=p.parse_args()
    run(a.s39_private,a.photo_tiles,a.frozen_s53_models,a.output,a.private_preview_dir,
        train_n=a.train_scenes,cal_n=a.cal_scenes,test_n=a.test_scenes,bootstrap=a.bootstrap)
if __name__=='__main__':main()
