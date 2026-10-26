#!/usr/bin/env python3
"""Rank S31's 28 authentic-photo pair QUESTIONS, without deciding physical joins.

S35 scorers learned only from synthetic overlays on other source photo; their
numbers are NOT calibrated physics probabilities or human same/different labels.
"""
from __future__ import annotations
import argparse,json,sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s35_fragment_link import optimal_endpoints,pair_features,prepare_evidence
from scripts.research.bubble_chamber.s35_benchmark_fragment_links import load_model,anza_probabilities_batch,ARMS


def _sha(path:Path):return sha256(path.read_bytes()).hexdigest()


def _endpoint_local_poly(full,endpoint_index,xy0):
    p=np.asarray(full,dtype='float64')
    # Take the endpoint-local 30 samples; a distant fragment section has no
    # relation to current putative bridge and is outside the ROI.
    end=p[:30] if endpoint_index==0 else p[-30:]
    end=end-np.asarray(xy0)
    keep=(end[:,0]>=0)&(end[:,0]<=191)&(end[:,1]>=0)&(end[:,1]<=191)
    if np.sum(keep)<8:raise ValueError('Fragment endpoint too close to image edge to inspect')
    valid=end[keep]
    if np.linalg.norm(valid[-1]-valid[0])<4:raise ValueError('Short observed endpoint path')
    return valid


def real_review_pairs(draft_file:Path,photo_dir:Path,checkpoint_dir:Path,bench_forward:Path,bench_reverse:Path):
    pack=json.loads(Path(draft_file).read_text(encoding='utf8'))
    if pack.get('schema')!='s31_machine_first_h50z_draft_v1' or pack.get('physical_joins_authorized') is not False:
        raise ValueError('Wrong machine-only candidate source')
    if len(pack.get('photos',[]))!=39 or pack.get('counts',{}).get('possible_relation_questions')!=28:
        raise ValueError('Expected S31 original 39 real crop contexts and 28 possible relation questions')
    fit={}
    for photo,path in [('006',bench_forward),('005',bench_reverse)]:
        b=json.loads(Path(path).read_text())
        if b.get('photo_heldout')!=photo or b.get('thresholds_fit_on_train_photo_only') is not True:
            raise ValueError('Synthetic scorer fitted on source photo being evaluated')
        fit[photo]=b
    models={}
    print('S35 source lock verified; loading AZ models',flush=True)
    for source in ('005','006'):
        model,prov=load_model(checkpoint_dir,source)
        models[source]=model
    print('S35 both AZ models loaded',flush=True)
    reviewed=[];skipped=[];scenes_by_photo={'005':[],'006':[]}
    for r in pack['photos']:
        crop=r['crop_id'];idmap={d['draft_id']:d for d in r['drafts']}
        if not r['relation_drafts']:continue
        image=Path(photo_dir)/(crop+'.png')
        if not image.is_file() or _sha(image)!=r['derived_png_sha256_from_BMP']:
            raise ValueError('Original H50Z source SHA does not match S31 machine draft')
        gray=cv2.imread(str(image),cv2.IMREAD_GRAYSCALE)
        if gray is None or sha256(np.ascontiguousarray(gray).tobytes()).hexdigest()!=r['source_gray_pixels_sha256']:
            raise ValueError('Original photo pixel values changed')
        for rel in r['relation_drafts']:
            if rel.get('status')!='MACHINE_POSSIBLE_LINK_UNKNOWN' or rel.get('expert_confirmed') is not False or rel.get('physical_join_allowed') is not False:
                raise ValueError('Foreign physics verification embedded in S31 relation')
            da=idmap[rel['candidate_a']];db=idmap[rel['candidate_b']]
            geom=optimal_endpoints(da['points_xy'],db['points_xy'])
            center=(geom['endpoint_a']+geom['endpoint_b'])/2
            x0=int(np.clip(round(center[0])-96,0,320));y0=int(np.clip(round(center[1])-96,0,320))
            cropgray=gray[y0:y0+192,x0:x0+192]
            if cropgray.shape!=(192,192):raise ValueError('Bad local photo window')
            try:
                aa=_endpoint_local_poly(da['points_xy'],geom['endpoint_index_a'],(x0,y0))
                bb=_endpoint_local_poly(db['points_xy'],geom['endpoint_index_b'],(x0,y0))
            except ValueError as err:
                skipped.append({'relation_id':rel['relation_id'],'reason':str(err),'crop_id':crop})
                continue
            scenes_by_photo[r['photo_id']].append((SimpleNamespace(gray=cropgray),
                    {'relation_id':rel['relation_id'],'crop_id':crop,'photo':r['photo_id'],
                     'src_png_sha256':r['derived_png_sha256_from_BMP'],
                     'candidate_a':rel['candidate_a'],'candidate_b':rel['candidate_b'],
                     'path_a':aa,'path_b':bb}))
    print('S35 real relation source windows',[(p,len(v)) for p,v in scenes_by_photo.items()],'skipped',len(skipped),flush=True)
    for photo,data in scenes_by_photo.items():
        model=models['005' if photo=='006' else '006']
        if not data:continue
        print('S35 real AZ inference start',photo,len(data),flush=True)
        scores=anza_probabilities_batch([scene for scene,_ in data],model)
        print('S35 real AZ inference done',photo,flush=True)
        params=fit[photo]['algorithms']
        for idx,((scene,r),az) in enumerate(zip(data,scores)):
            if idx%5==0:print('S35 scoring real',photo,idx,'/',len(data),flush=True)
            f=pair_features(r['path_a'],r['path_b'],prepare_evidence(scene.gray,anza_map=az))
            outputs={}
            for name in ('geometry','geometry+raw','geometry+local','geometry+AZUNet'):
                spec=params[name]['synthetic_only_fitted_model']
                fields=spec['fields']
                feature=np.array([f[k] for k in fields],dtype='float64')
                z=(feature-np.asarray(spec['scaler_mean']))/np.asarray(spec['scaler_scale'])
                logit=float(np.dot(z,np.asarray(spec['coef']))+spec['intercept'])
                score=float(1/(1+np.exp(-np.clip(logit,-60,60))))
                outputs[name+'_UNVERIFIED_SYNTHETIC_SCORER']=round(score,6)
            disagreement=abs(outputs['geometry+local_UNVERIFIED_SYNTHETIC_SCORER']-outputs['geometry+AZUNet_UNVERIFIED_SYNTHETIC_SCORER'])
            reviewed.append({'relation_id':r['relation_id'],'crop_id':r['crop_id'],
                'photo_id':photo,'candidate_a':r['candidate_a'],'candidate_b':r['candidate_b'],
                'source_image_sha256':r['src_png_sha256'],
                'provisional_scores_NOT_PHYSICAL_PROBABILITIES':outputs,
                'local_vs_ANZA_score_disagreement':round(disagreement,6),
                'measured_endpoint_gap_px':round(f['endpoint_gap_px'],3),
                'outgoing_tangent_a':round(f['outgoing_tangent_A_to_gap'],4),
                'outgoing_tangent_b':round(f['outgoing_tangent_B_to_gap'],4),
                'status':'UNKNOWN_REVIEW_REQUIRED',
                'human_expert_label':None,'physics_auto_join_allowed':False})
    ids={q['relation_id'] for q in reviewed}
    if len(reviewed)+len(skipped)!=28 or len(ids)!=len(reviewed):
        raise AssertionError('Lost or duplicated S31 relation questions')
    reviewed.sort(key=lambda row:-row['local_vs_ANZA_score_disagreement'])
    return {'schema':'s35_private_real_h50z_pair_review_evidence_v1',
        'source_original_S31_draft_sha256':_sha(Path(draft_file)),
        'genuine_source_H50Z_photo_count':2,'cropped_source_images':39,
        'original_S31_relation_questions':28,'evaluable_relation_questions':len(reviewed),
        'insufficient_local_geometry':skipped,
        'human_confirmed_same_different_pairs':0,'physics_auto_join_allowed':False,
        'S35_synthetic_scorer_not_calibrated_on_real_tracks':True,
        'model_source_only_machine_pseudolabels':True,
        'ranked_by':'disagreement between independently calibrated synthetic models, not likelihood of correctness',
        'rows':reviewed}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--drafts',required=True,type=Path)
    p.add_argument('--photos',required=True,type=Path)
    p.add_argument('--models',required=True,type=Path)
    p.add_argument('--benchmark-forward',required=True,type=Path)
    p.add_argument('--benchmark-reverse',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    result=real_review_pairs(a.drafts,a.photos,a.models,a.benchmark_forward,a.benchmark_reverse)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('S35 real pair questions',result['evaluable_relation_questions'],
          'skipped',len(result['insufficient_local_geometry']),
          'certified:',result['human_confirmed_same_different_pairs'])
