#!/usr/bin/env python3
"""S35 photo-disjoint synthetic fragment-link benchmark with abstention.

Known join labels exist ONLY for newly injected visualMetro-derived strokes, not
native H50Z track pixels. Photos 005/006 from same physical run; feature
calibration and model selection never use test photo labels. VisualMetro source
frames are also disjoint between calibration/training and synthetic test.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import argparse,json,sys,time
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_simulated_gaps import frame_split,extract_track_templates,make_scene
from bubble_chamber.s35_fragment_link import (GEOMETRY_FIELDS,PHOTO_FIELDS,ANZA_FIELDS,
    pair_features,prepare_evidence,feature_matrix,
    calibrated_negative_threshold,select_review_links)
from bubble_chamber.s33_photometric_domain import image_input
from bubble_chamber.s37_source_model import build_model,verify_model_source


ARMS={
    'distance':('endpoint_gap_px',),
    'geometry':GEOMETRY_FIELDS,
    'geometry+raw':GEOMETRY_FIELDS+PHOTO_FIELDS[:2],
    'geometry+local':GEOMETRY_FIELDS+PHOTO_FIELDS,
    'geometry+AZUNet':GEOMETRY_FIELDS+PHOTO_FIELDS+ANZA_FIELDS,
}
PROTOCOL={'train_scenes':90,'calibration_scenes':50,'test_scenes_per_seed':90,
          'test_seeds':(20261121,20261122),'train_seed':20261111,'calib_seed':20261112,
          'empirical_max_calibration_FPR':.02,'minimum_margin':.04}


def _sha(path):
    d=sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):d.update(c)
    return d.hexdigest()


def load_model(checkpoints,train_photo,*,seed=20261025):
    """Original source-locked AZUNet, already trained on S33 machine masks."""
    verify_model_source()
    file=Path(checkpoints)/(f'S33_models{"_reverse" if train_photo=="006" else ""}')/f'azunet_quantile_seed{seed}.pt'
    ck=torch.load(file,weights_only=True,map_location='cpu')
    if ck.get('schema')!='s33_machine_draft_photo_normalized_v1' or ck.get('photo_train')!=train_photo or ck.get('mode')!='quantile' or ck.get('architecture')!='azunet':
        raise ValueError('ANZA checkpoint came from different photo or model')
    if ck.get('physical_auto_joins_authorized') is not False or ck.get('label_type')!='MACHINE_DRAFT_NOT_HUMAN_EXPERT':
        raise ValueError('Model must be machine-only and cannot certify joins')
    model=build_model('azunet',widths=(4,6,8,12),rules=2)
    model.load_state_dict(ck['model_state']);model.eval()
    return model,{'checkpoint':file.name,'sha256':_sha(file),
                  'trained_on_source_photo':train_photo,'certified_native_real_labels':0}


def anza_probabilities_batch(scenes,model,*,batch_size=16):
    """Vectorized AZUNet evidence around known gap; pixels outside ROI UNKNOWN.

    Two original-size 64px source windows per scene. Never resize the
    photographic input or treat the neutral outer area as an observation.
    """
    torch.set_num_threads(2)
    positions=[(48,56),(112,56)]  # x:56-120, y:48-176, around gap.
    patches=[]
    for scene in scenes:
        if scene.gray.shape!=(192,192):raise ValueError('Bad 192px test scene')
        for y,x in positions:
            patches.append(image_input(scene.gray[y:y+64,x:x+64],'quantile'))
    tensor=np.stack(patches).astype('float32')[:,None]
    predicted=[]
    model.eval()
    with torch.no_grad():
        for k in range(0,len(tensor),batch_size):
            predicted.append(torch.sigmoid(model(torch.from_numpy(tensor[k:k+batch_size]))).cpu().numpy()[:,0])
    probs=np.concatenate(predicted)
    results=[]
    for i,scene in enumerate(scenes):
        m=np.full((192,192),.5,dtype='float32')
        for j,(y,x) in enumerate(positions):
            m[y:y+64,x:x+64]=probs[2*i+j]
        if not np.isfinite(m).all():raise ValueError('Non-finite model outputs')
        results.append(m)
    return results


def generate_cases(tiles_dir,photo,templates,model,*,n,seed,store_images=None):
    files=sorted(Path(tiles_dir).glob(f'H50Z_030_01_{photo}_*.png'))
    if len(files)!=(21 if photo=='005' else 18):raise ValueError('Missing original 512px crops')
    rng=np.random.default_rng(seed)
    scenarios=['parallel','crossing','scratched','empty_gap']
    scenes=[]
    for i in range(n):
        file=files[int(rng.integers(len(files)))]
        scene=make_scene(file,templates,seed=int(rng.integers(1,2**31-1)),scenario=scenarios[i%4])
        if store_images is not None and i<8:
            folder=Path(store_images);folder.mkdir(parents=True,exist_ok=True)
            rgb=cv2.cvtColor(scene.gray,cv2.COLOR_GRAY2BGR)
            for a in scene.left:cv2.polylines(rgb,[np.rint(a).astype(np.int32)],False,(100,250,100),1)
            for b in scene.right:cv2.polylines(rgb,[np.rint(b).astype(np.int32)],False,(100,100,240),1)
            cv2.imwrite(str(folder/f'synthetic_control_{photo}_{seed}_{i:02d}_PRIVATE.png'),rgb)
        scenes.append(scene)
    print('S35 scenes generated',photo,n,flush=True)
    maps=anza_probabilities_batch(scenes,model)
    print('S35 ANZA window inference',photo,n,'complete',flush=True)
    out=[]
    for i,(s,az) in enumerate(zip(scenes,maps)):
        evidence=prepare_evidence(s.gray,anza_map=az)
        candidate=[]
        for a,pa in enumerate(s.left):
            for b,pb in enumerate(s.right):
                attrs=pair_features(pa,pb,evidence)
                pair={'left_index':a,'right_index':b,'label_SYNTHETIC_ONLY':int((a,b) in s.correct_pairs)}
                pair.update({k:attrs[k] for k in GEOMETRY_FIELDS+PHOTO_FIELDS+ANZA_FIELDS})
                candidate.append(pair)
        assert len(candidate)==6 and sum(x['label_SYNTHETIC_ONLY'] for x in candidate)==2
        out.append({'scene_index':i,'scenario':s.scenario,'photo':photo,
                    'synthetic_detector_centerline_error_px':s.simulated_measurement_error_px,
                    'source_visualMetro_frames':s.source_frame_ids,'pairs':candidate})
    return out


def flatten(cases):
    rows=[];groups=[];labels=[]
    for i,scene in enumerate(cases):
        for pair in scene['pairs']:
            rows.append(pair);labels.append(pair['label_SYNTHETIC_ONLY']);groups.append(i)
    return rows,np.asarray(labels,int),np.asarray(groups,int)


def score_arm(arm,train_rows,y_train,cal_rows,y_cal,test_rows):
    if arm=='distance':
        train_scores=-feature_matrix(train_rows,ARMS[arm])[:,0]
        cal_scores=-feature_matrix(cal_rows,ARMS[arm])[:,0]
        test_scores=-feature_matrix(test_rows,ARMS[arm])[:,0]
        weights=None
        model_coefficients=None
    else:
        fields=ARMS[arm]
        scaler=StandardScaler().fit(feature_matrix(train_rows,fields))
        model=LogisticRegression(max_iter=600,class_weight='balanced',random_state=1337,C=.3)
        model.fit(scaler.transform(feature_matrix(train_rows,fields)),y_train)
        train_scores=model.predict_proba(scaler.transform(feature_matrix(train_rows,fields)))[:,1]
        cal_scores=model.predict_proba(scaler.transform(feature_matrix(cal_rows,fields)))[:,1]
        test_scores=model.predict_proba(scaler.transform(feature_matrix(test_rows,fields)))[:,1]
        weights={k:float(c) for k,c in zip(fields,model.coef_[0])}
        model_coefficients={'fields':list(fields),'scaler_mean':scaler.mean_.tolist(),
            'scaler_scale':scaler.scale_.tolist(),'coef':model.coef_[0].tolist(),
            'intercept':float(model.intercept_[0]),'synthetic_training_only':True}
    thr=calibrated_negative_threshold(cal_scores,y_cal,max_fpr=PROTOCOL['empirical_max_calibration_FPR'])
    return {'threshold':float(thr),'train_scores':train_scores,'cal_scores':cal_scores,
            'test_scores':test_scores,'weights_feature_z':weights,
            'empirical_calibration_FPR':float(np.mean(cal_scores[y_cal==0]>=thr)),
            'empirical_calibration_TPR':float(np.mean(cal_scores[y_cal==1]>=thr)),
            'synthetic_only_fitted_model':model_coefficients}


def score_test(cases,score,threshold,*,margin=.04):
    y=np.asarray([r['label_SYNTHETIC_ONLY'] for scene in cases for r in scene['pairs']],int)
    scores=np.asarray(score,float)
    if len(y)!=len(scores):raise ValueError('Scores do not match held-out cases')
    auroc=float(roc_auc_score(y,scores));ap=float(average_precision_score(y,scores))
    approved=[];n_true=0;n_false=0;abstains=0;per_scenario={}
    for i,scene in enumerate(cases):
        local=scores[i*6:(i+1)*6]
        result=select_review_links(scene['pairs'],local,threshold,margin=margin)
        assert result['physics_auto_joins_authorized'] is False
        joins=result['suggested_pairs'];abstains+=result['abstained_left_fragments']
        n_true+=sum(tuple(pair) in {(0,0),(1,1)} for pair in joins)
        n_false+=sum(tuple(pair) not in {(0,0),(1,1)} for pair in joins)
        q=per_scenario.setdefault(scene['scenario'],{'scenes':0,'correct':0,'false':0})
        q['scenes']+=1;q['correct']+=sum(tuple(pair) in {(0,0),(1,1)} for pair in joins)
        q['false']+=sum(tuple(pair) not in {(0,0),(1,1)} for pair in joins)
    count=len(cases)
    return {'synthetic_only_AUROC':auroc,'synthetic_only_AUPRC':ap,
            'correct_provisional_review_suggestions':n_true,'wrong_provisional_review_suggestions':n_false,
            'wrong_suggestions_per_100_synthetic_scenes':100*n_false/max(count,1),
            'true_link_recall_with_abstention':n_true/(2*count),
            'pairwise_wrong_at_threshold':int(np.sum((scores>=threshold)&(y==0))),
            'pairwise_true_at_threshold':int(np.sum((scores>=threshold)&(y==1))),
            'left_fragments_abstained':abstains,
            'all_review_only':True,'native_photo_track_truth_count':0,
            'scenarios':per_scenario}


def run(templates_root,photo_tiles,model_dir,outdir,*,photo_train='005',
        train_scenes=90,calibration_scenes=50,test_scenes=90,seed=20261121):
    torch.set_num_threads(2)
    if photo_train not in ('005','006') or train_scenes<24 or calibration_scenes<12 or test_scenes<12:
        raise ValueError('Unsafe/insufficient benchmark budget')
    photo_test='006' if photo_train=='005' else '005'
    manifest=audited_large_manifest(Path(templates_root))
    train_frames,test_frames=frame_split(manifest)
    train_templates=extract_track_templates(manifest,chosen_frames=train_frames)
    test_templates=extract_track_templates(manifest,chosen_frames=test_frames)
    model,ck=load_model(model_dir,photo_train)
    started=time.monotonic()
    print('S35 generating train',photo_train,train_scenes,flush=True)
    train=generate_cases(photo_tiles,photo_train,train_templates,model,n=train_scenes,
                         seed=20261111,store_images=Path(outdir)/'PRIVATE_synthetic_train_samples')
    print('S35 generating calibration',photo_train,calibration_scenes,flush=True)
    calibrate=generate_cases(photo_tiles,photo_train,train_templates,model,n=calibration_scenes,
                             seed=20261112)
    print('S35 generating holdout',photo_test,test_scenes,flush=True)
    test=generate_cases(photo_tiles,photo_test,test_templates,model,n=test_scenes,
                        seed=seed,store_images=Path(outdir)/'PRIVATE_synthetic_test_samples')
    a,ya,_=flatten(train);b,yb,_=flatten(calibrate);c,yc,_=flatten(test)
    if set((s['source_visualMetro_frames'][0] for s in test))&train_frames:
        raise AssertionError('Test visualMetro template frame leaked into fitting')
    outcomes={};scored={}
    print('S35 fitting only on source photo',photo_train,flush=True)
    for arm in ARMS:
        result=score_arm(arm,a,ya,b,yb,c)
        scored[arm]=result['test_scores']
        summary=score_test(test,result['test_scores'],result['threshold'],margin=PROTOCOL['minimum_margin'] if arm!='distance' else 0.0)
        outcomes[arm]={'empirical_train_photo_calibration_FPR':result['empirical_calibration_FPR'],
            'empirical_train_photo_calibration_TPR':result['empirical_calibration_TPR'],
            'calibration_threshold':result['threshold'],
            'feature_fields':ARMS[arm],
            'feature_coefficients_STANDARDIZED_NOT_PHYSICAL':result['weights_feature_z'],
            'synthetic_only_fitted_model':result['synthetic_only_fitted_model'],
            'heldout':summary}
    # Fixed consensus: only suggestions accepted by BOTH geometry and photo+AZ.
    count={'correct':0,'wrong':0,'abstained':0,'scenes':len(test)}
    for i,scene in enumerate(test):
        a_score=scored['geometry'][i*6:(i+1)*6]
        b_score=scored['geometry+AZUNet'][i*6:(i+1)*6]
        r1=select_review_links(scene['pairs'],a_score,outcomes['geometry']['calibration_threshold'])
        r2=select_review_links(scene['pairs'],b_score,outcomes['geometry+AZUNet']['calibration_threshold'])
        intersection=set(r1['suggested_pairs'])&set(r2['suggested_pairs'])
        count['correct']+=len(intersection&{(0,0),(1,1)})
        count['wrong']+=len(intersection-{(0,0),(1,1)})
        count['abstained']+=2-len(intersection)
    result={'schema':'s35_photo_and_frame_disjoint_synthetic_fragment_links_v1',
       'source_data':'Authentic H50Z photo pixels used as background; only drawn fragments are ground-truth synthetic',
       'photo_train':photo_train,'photo_heldout':photo_test,
       'one_physical_run_only':True,'native_physical_track_GT':0,
       'visualMetro_frame_split_SHA256':sha256(('\n'.join(sorted(train_frames))+'|'+('\n'.join(sorted(test_frames)))).encode()).hexdigest(),
       'visualMetro_train_frames':len(train_frames),'visualMetro_heldout_frames':len(test_frames),
       'visualMetro_train_templates':len(train_templates),'visualMetro_heldout_templates':len(test_templates),
       'background_crops_train':21 if photo_train=='005' else 18,
       'background_crops_test':18 if photo_train=='005' else 21,
       'model_checkpoint_provenance':ck,'seed_test':seed,'train_n_scenes':len(train),
       'calibration_n_scenes':len(calibrate),'test_n_scenes':len(test),
       'test_synthetic_true_pairs':len(test)*2,'test_synthetic_wrong_pairs':len(test)*4,
       'train_synthetic_true_pairs':len(train)*2,'train_synthetic_wrong_pairs':len(train)*4,
       'algorithms':outcomes,
       'geometry_and_ANZA_consensus':{'status':'PROVISIONAL_REVIEW_ONLY','correct':count['correct'],
           'wrong':count['wrong'],'wrong_per_100_synthetic_scenes':100*count['wrong']/len(test),
           'true_link_recall':count['correct']/(2*len(test)),'abstained_left_fragments':count['abstained'],
           'physics_auto_joins_authorized':False},
       'original_ANZA_model_edited':False,'physicist_expert_confirmed_links_added':0,
       'thresholds_fit_on_train_photo_only':True,
       'empirical_calibration_false_link_bound_IS_NOT_STATISTICAL_GUARANTEE':True,
       'time_seconds':round(time.monotonic()-started,2)}
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    (out/f'S35_{photo_train}_to_{photo_test}_seed{seed}_SYNTHETIC_RESEARCH.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    for name,entry in outcomes.items():
        m=entry['heldout'];print('S35',photo_train+'→'+photo_test,'test',seed,name,
                'AUROC',round(m['synthetic_only_AUROC'],4),
                'wrong/100scenes',round(m['wrong_suggestions_per_100_synthetic_scenes'],2),
                'recall',round(m['true_link_recall_with_abstention'],4),flush=True)
    print('S35 consensus',result['geometry_and_ANZA_consensus'],'seconds',result['time_seconds'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--train-photo',choices=['005','006'],default='005')
    p.add_argument('--train-scenes',type=int,default=90)
    p.add_argument('--cal-scenes',type=int,default=50)
    p.add_argument('--test-scenes',type=int,default=90)
    p.add_argument('--seed',type=int,default=20261121)
    a=p.parse_args()
    run(a.visualmetro,a.photo_tiles,a.checkpoint_dir,a.outdir,photo_train=a.train_photo,
        train_scenes=a.train_scenes,calibration_scenes=a.cal_scenes,test_scenes=a.test_scenes,seed=a.seed)
