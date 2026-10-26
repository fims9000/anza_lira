#!/usr/bin/env python3
"""S37: fixed-split link study using authentic backgrounds but *synthetic* truth.

Public outputs never include original pixels, source track coordinates, raw
per-scene predictions, or private frame IDs. Only summary metrics are saved.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from dataclasses import asdict
from pathlib import Path
import json
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score
import torch
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_simulated_gaps import make_scene,extract_track_templates
from bubble_chamber.s35_fragment_link import (GEOMETRY_FIELDS,PHOTO_FIELDS,ANZA_FIELDS,
    prepare_evidence,pair_features,optimal_endpoints,hermite_gap,_photo_score,
    feature_matrix,calibrated_negative_threshold)
from bubble_chamber.s33_photometric_domain import image_input
from bubble_chamber.s37_source_model import build_model,verify_model_source
from bubble_chamber.s37_risk_calibration import (
    scene_stats,calibrate_review_risk,cp_upper,public_stats)

N_TRAIN=160
N_CAL=450
N_TEST=80
MARGIN=.04
SCENE_RISK_TARGET=.02
FAMILY_ALPHA=.05
TEST_SEEDS={'005':(20261221,20261222),'006':(20261223,20261224)}
TRAIN_SEED={'005':20261201,'006':20261203}
CAL_SEED={'005':20261202,'006':20261204}
UNET_FIELDS=('UNet_endpoint_contrast_diff','UNet_endpoint_contrast_min',
             'UNet_bridge_center_flank')
ARMS={
    'distance':('endpoint_gap_px',),
    'geometry':GEOMETRY_FIELDS,
    'geometry+raw':GEOMETRY_FIELDS+PHOTO_FIELDS[:2],
    'geometry+local':GEOMETRY_FIELDS+PHOTO_FIELDS,
    'geometry+AZUNet':GEOMETRY_FIELDS+PHOTO_FIELDS+ANZA_FIELDS,
    'geometry+UNet':GEOMETRY_FIELDS+PHOTO_FIELDS+UNET_FIELDS,
    'geometry+both':GEOMETRY_FIELDS+PHOTO_FIELDS+ANZA_FIELDS+UNET_FIELDS,
}


def sha_file(p:Path)->str:
    h=sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def split_source_frames(manifest):
    ids=[str(x['frame_id']) for x in manifest['frames']]
    if len(ids)!=145 or len(set(ids))!=145:raise ValueError('Exact complete S37 source required (145 frames)')
    ordered=sorted(ids,key=lambda x:sha256(('S37_SOURCE_LOCK_v1|'+x).encode()).digest())
    test=set(ordered[:36]);cal=set(ordered[36:66]);train=set(ordered[66:])
    assert len(test)==36 and len(cal)==30 and len(train)==79
    assert not test&cal and not test&train and not cal&train
    digest=sha256(('TRAIN:'+','.join(sorted(train))+'|CAL:'+','.join(sorted(cal))+
                   '|TEST:'+','.join(sorted(test))).encode()).hexdigest()
    return train,cal,test,digest


def load_pretrained(az_root:Path,unet_root:Path,photo:str):
    """Both original architectures; verify exact source and weak-label provenance."""
    verify_model_source()
    folder='S33_models' if photo=='005' else 'S33_models_reverse'
    az_file=az_root/folder/'azunet_quantile_seed20261025.pt'
    unet_file=unet_root/folder/'unet_local_seed20261025.pt'
    out=[];markers=[]
    for file,architecture,mode in [(az_file,'azunet','quantile'),(unet_file,'unet','local')]:
        ck=torch.load(file,map_location='cpu',weights_only=True)
        expected={'schema':'s33_machine_draft_photo_normalized_v1','architecture':architecture,
                  'mode':mode,'photo_train':photo,
                  'label_type':'MACHINE_DRAFT_NOT_HUMAN_EXPERT',
                  'physical_auto_joins_authorized':False}
        for key,value in expected.items():
            if ck.get(key)!=value:raise ValueError('Checkpoint provenance failed: '+key)
        m=build_model(architecture,widths=(4,6,8,12),rules=2)
        m.load_state_dict(ck['model_state']);m.eval()
        out.append(m)
        markers.append({'original_architecture':architecture,'preprocess':mode,
            'source_weak_manifest_sha256':ck.get('source_weak_manifest_sha256'),
            'checkpoint_sha256':sha_file(file),'human_verified_pixel_labels':0})
    if markers[0]['source_weak_manifest_sha256']!=markers[1]['source_weak_manifest_sha256']:
        raise ValueError('AZUNet and U-Net training labels have different source manifests')
    return out,markers


def _maps_from_model(scenes,model,mode,*,batch_size=24):
    """Identical 2x64x64 evidence locations for the two original networks."""
    positions=((48,56),(112,56))
    patches=[]
    for scene in scenes:
        if scene.gray.shape!=(192,192):raise ValueError('Bad generated scene shape')
        patches.extend(image_input(scene.gray[y:y+64,x:x+64],mode)
                       for y,x in positions)
    tensor=torch.from_numpy(np.stack(patches).astype('float32')[:,None])
    pred=[]
    with torch.no_grad():
        for start in range(0,len(tensor),batch_size):
            pred.append(torch.sigmoid(model(tensor[start:start+batch_size])).numpy()[:,0])
    stacked=np.concatenate(pred)
    grids=[]
    for idx,scene in enumerate(scenes):
        grid=np.full(scene.gray.shape,.5,dtype='float32')
        for j,(y,x) in enumerate(positions):grid[y:y+64,x:x+64]=stacked[2*idx+j]
        if not np.isfinite(grid).all():raise ValueError('Nonfinite neural output')
        grids.append(grid)
    return grids


def _unet_pair_features(a,b,unet_grid):
    """Same photometric score functional as the original S35 ANZA arm."""
    conn=optimal_endpoints(a,b)
    ap,bp,ta,tb=[conn[k] for k in ('endpoint_a','endpoint_b','outward_a','outward_b')]
    sel_a=a[-20:] if conn['endpoint_index_a']==-1 else a[:20]
    sel_b=b[-20:] if conn['endpoint_index_b']==-1 else b[:20]
    x=_photo_score(unet_grid,sel_a)
    y=_photo_score(unet_grid,sel_b)
    bridge=_photo_score(unet_grid,hermite_gap(ap,bp,ta,tb))
    return dict(zip(UNET_FIELDS,[abs(x-y),min(x,y),bridge]))


def generate_cases(tiles_dir,photo,templates,models,*,n,seed):
    paths=sorted(tiles_dir.glob(f'H50Z_030_01_{photo}_*.png'))
    if len(paths)!=(21 if photo=='005' else 18):raise ValueError('Missing authentic source crops')
    rng=np.random.default_rng(seed)
    modes=('parallel','crossing','scratched','empty_gap')
    generated=[]
    for i in range(n):
        tile=paths[int(rng.integers(len(paths)))]
        scene=make_scene(tile,templates,seed=int(rng.integers(1,2**31-1)),scenario=modes[i%4])
        generated.append(scene)
    azmaps=_maps_from_model(generated,models[0],'quantile')
    unetmaps=_maps_from_model(generated,models[1],'local')
    cases=[]
    for scene,az,unet in zip(generated,azmaps,unetmaps):
        evidence=prepare_evidence(scene.gray,anza_map=az)
        rows=[]
        for i,a in enumerate(scene.left):
            for j,b in enumerate(scene.right):
                row={'left_index':i,'right_index':j,
                     'label_SYNTHETIC_ONLY':int((i,j) in scene.correct_pairs)}
                row.update({key:value for key,value in pair_features(a,b,evidence).items()
                            if key in GEOMETRY_FIELDS+PHOTO_FIELDS+ANZA_FIELDS})
                row.update(_unet_pair_features(a,b,unet))
                rows.append(row)
        assert len(rows)==6 and sum(r['label_SYNTHETIC_ONLY'] for r in rows)==2
        cases.append({'pairs':rows,'scenario':scene.scenario,
                      # The group is kept in RAM only; no private original IDs in public report.
                      '_private_geometry_source':scene.source_frame_ids})
    return cases


def flatten(cases):
    return [row for scene in cases for row in scene['pairs']],np.asarray(
        [r['label_SYNTHETIC_ONLY'] for scene in cases for r in scene['pairs']],dtype=int)


def fit_arm(arm,train_rows,train_y):
    fields=ARMS[arm]
    train_x=feature_matrix(train_rows,fields)
    if arm=='distance':
        return lambda rows:-feature_matrix(rows,fields)[:,0]
    scaler=StandardScaler().fit(train_x)
    model=LogisticRegression(max_iter=600,class_weight='balanced',random_state=1337,C=.3)
    model.fit(scaler.transform(train_x),train_y)
    return lambda rows:model.predict_proba(scaler.transform(feature_matrix(rows,fields)))[:,1]


def decision_report(cases,scores,threshold,*,margin=MARGIN):
    metrics=public_stats(scene_stats(cases,scores,threshold,margin=margin))
    true_links=2*len(cases)
    metrics['synthetic_true_link_recall']=metrics['correct_links']/true_links
    metrics['review_precision_when_suggesting']=(metrics['correct_links']/metrics['accepted_suggestions']
        if metrics['accepted_suggestions'] else None)
    metrics['one_sided_95pct_false_scene_CI_upper_NOT_PHYSICAL']=cp_upper(
        metrics['any_false_scene_count'],len(cases),.05)
    metrics['physics_auto_joins_authorized']=False
    return metrics


def run(source,tiles,az_root,unet_root,out,*,train_n=N_TRAIN,cal_n=N_CAL,test_n=N_TEST,only_photo="both"):
    torch.set_num_threads(2)
    if train_n<24 or cal_n<12 or test_n<12:raise ValueError('Insufficient benchmark scenes')
    m=audited_large_manifest(source)
    groups=split_source_frames(m)
    train_frames,cal_frames,test_frames,split_sha=groups
    templates={name:extract_track_templates(m,chosen_frames=fset)
        for name,fset in [('train',train_frames),('cal',cal_frames),('test',test_frames)]}
    print('S37 original source',m['summary']['frames_complete_used'],'complete frames, templates',
          {k:len(v) for k,v in templates.items()},flush=True)
    overall={'schema':'s37_real_background_synthetic_links_aggregate_only_v1',
             'source_dataset_sha256':m['provenance_sha256'],
             'protocol_source_145_frame_source_lock':split_sha,
             'source_frame_counts':{k:len(v) for k,v in [('train',train_frames),('cal',cal_frames),('test',test_frames)]},
             'source_templates_counts':{k:len(v) for k,v in templates.items()},
             'authentic_background_photos_count':2,'independent_physical_runs_tested':0,
             'real_certified_track_masks':0,'real_certified_track_relation_labels':0,
             'models_are_original_source_locked':True,
             'training_only_machine_weak_pixel_masks':True,
             'synthetic_injection_seeds':TEST_SEEDS,'test_seed_used_to_tune':False,
             'n_train_per_direction':train_n,'n_cal_per_direction':cal_n,'n_test_per_seed':test_n,
             'n_arms':len(ARMS),'multiple_testing_family_alpha':FAMILY_ALPHA,
             'simulated_scene_false_event_target':SCENE_RISK_TARGET,
             'never_automatic_physical_reconstruction':True,'directions':{}}
    if only_photo not in ('005','006','both'):
        raise ValueError('Unknown research direction')
    for photo in ('005','006'):
        if only_photo!='both' and photo!=only_photo:continue
        target='006' if photo=='005' else '005'
        models,model_info=load_pretrained(az_root,unet_root,photo)
        print(f'S37 train {photo}, calibration {photo}, test {target}',flush=True)
        train=generate_cases(tiles,photo,templates['train'],models,n=train_n,seed=TRAIN_SEED[photo])
        cal=generate_cases(tiles,photo,templates['cal'],models,n=cal_n,seed=CAL_SEED[photo])
        train_rows,train_y=flatten(train);cal_rows,cal_y=flatten(cal)
        fitted={};thresholds={};raw_risk_cutoffs={}
        for name in ARMS:
            fn=fit_arm(name,train_rows,train_y)
            tr=fn(train_rows);ca=fn(cal_rows)
            empirical=calibrated_negative_threshold(ca,cal_y,max_fpr=.02)
            strict=calibrate_review_risk(tr,cal,ca,target_scene_false_risk=SCENE_RISK_TARGET,
                family_alpha=FAMILY_ALPHA/(len(ARMS)*2),n_cutoffs=13,margin=MARGIN)
            fitted[name]=fn
            raw_risk_cutoffs[name]=strict.threshold
            safe_decision=asdict(strict)
            if not np.isfinite(strict.threshold):safe_decision['threshold']=None
            thresholds[name]={'empirical_pairwise_fpr_2pct':float(empirical),
                 's37_scene_risk_threshold':float(strict.threshold) if np.isfinite(strict.threshold) else None,
                 's37_calibration_decision':safe_decision,
                 'calibration_empirical':decision_report(cal,ca,empirical),
                 'calibration_s37':decision_report(cal,ca,strict.threshold)}
        direction={'photo_train':photo,'photo_holdout':target,
            'pretrained_models':model_info,'arms':{name:dict(v,tests={}) for name,v in thresholds.items()}}
        for seed in TEST_SEEDS[photo]:
            test=generate_cases(tiles,target,templates['test'],models,n=test_n,seed=seed)
            rows,y=flatten(test)
            for name,fn in fitted.items():
                sc=fn(rows)
                rec={'test_SYNTHETIC_AUROC':float(roc_auc_score(y,sc)),
                     'test_SYNTHETIC_AUPRC':float(average_precision_score(y,sc)),
                     'empirical_pairwise_policy':decision_report(test,sc,thresholds[name]['empirical_pairwise_fpr_2pct']),
                     's37_scene_risk_policy':decision_report(test,sc,raw_risk_cutoffs[name])}
                direction['arms'][name]['tests'][str(seed)]=rec
                print('S37',photo,'->',target,seed,name,'AUC',round(rec['test_SYNTHETIC_AUROC'],4),
                      'empirical',rec['empirical_pairwise_policy']['correct_links'],
                      rec['empirical_pairwise_policy']['wrong_links'],
                      'risk-gated',rec['s37_scene_risk_policy']['correct_links'],
                      rec['s37_scene_risk_policy']['wrong_links'],flush=True)
        overall['directions'][photo+'->'+target]=direction
        # Atomic checkpoint after each direction: never lose completed results.
        p=Path(out);p.parent.mkdir(exist_ok=True,parents=True)
        tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(overall,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        tmp.replace(p)
        print('S37 direction persisted',photo,'->',target,flush=True)
    p=Path(out);p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(overall,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print('SAFE S37 public summary written',p,flush=True)
    return overall


if __name__=='__main__':
    arg=argparse.ArgumentParser()
    arg.add_argument('--visualmetro',required=True,type=Path)
    arg.add_argument('--photo-tiles',required=True,type=Path)
    arg.add_argument('--az-weights',required=True,type=Path)
    arg.add_argument('--unet-weights',required=True,type=Path)
    arg.add_argument('--out',required=True,type=Path)
    arg.add_argument('--train-scenes',type=int,default=N_TRAIN)
    arg.add_argument('--cal-scenes',type=int,default=N_CAL)
    arg.add_argument('--test-scenes',type=int,default=N_TEST)
    arg.add_argument('--direction',choices=['005','006','both'],default='both')
    a=arg.parse_args()
    run(a.visualmetro,a.photo_tiles,a.az_weights,a.unet_weights,a.out,
        train_n=a.train_scenes,cal_n=a.cal_scenes,test_n=a.test_scenes,only_photo=a.direction)
