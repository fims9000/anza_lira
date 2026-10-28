#!/usr/bin/env python3
"""Detector-specific synthetic TRAIN/CAL/TEST for developer interoperability.

Unlike the previous fixed-heuristic comparison, this fits an independent
geometry scorer per pixel detector and CAL-selects thresholds *before* TEST.

Everything here is generated from synthetic pixels/templates; no real H50Z,
visualMetro 070 measurements, physicist labels or S85 learned weights appear.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from bubble_chamber.interop_api import ScoreBinding,assemble_lines,validate_lines
from bubble_chamber.interop_detectors import HessianRidgeDetector,OpenCVLSDDetector,OpenCVHoughDetector
from bubble_chamber.s35_fragment_link import GEOMETRY_FIELDS,feature_matrix
from bubble_chamber.s56_three_piece_scenes import SCENARIOS,make_three_piece_scene
from bubble_chamber.s56_global_path_assembly import pair_rows
from bubble_chamber.s59_candidate_retrieval import filter_rows
from bubble_chamber.s64_pixel_to_path import _candidate_associations,INJECTED_TRUE_PAIRS,ASSOCIATION
from bubble_chamber.s69_orientation_invariant_audit import reference_facing_ports
from bubble_chamber.s85_image_disturbances import condition_pixels
from scripts.research.bubble_chamber.s80_calibrate_then_test_multiscale import score_after_inference

ID='ANZA-LIRA-BC-S85-INTEROP-CAL-20261028'
SCHEMA='S85_INTEROP_DETECTOR_SPECIFIC_TRAIN_CAL_FRESH_TEST_V1'
PROTOCOL=ROOT/'docs/bubble_chamber/S85_INTEROP_CALIBRATED_TOY_PROTOCOL_2026_10_28.md'
SEEDS={'train':20268610,'cal':20268611,'test':20268612}
SCENE_N={'train':24,'cal':16,'test':24}
RNG_MODEL=20268613
CONDITIONS=('clean','blur_sigma_11','scratch_grain')
STRATEGIES=('none','node_mutual','port_greedy','global_milp')
CUTS=(.50,.65,.80,.90,.97)
FIELDS=GEOMETRY_FIELDS
DETECTORS=(HessianRidgeDetector,OpenCVLSDDetector,OpenCVHoughDetector)
KEYS=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
      'complete_three_piece_paths_synthetic','shape_true_fragments_matched',
      'solver_failure','candidate_polylines','candidate_pairs_after_strict_gate')


def generated_templates(seed):
    rng=np.random.default_rng(seed)
    templates=[]
    for i in range(32):
        t=np.linspace(0,1,80)
        amp=float(rng.uniform(4,17))
        phase=float(rng.uniform(-np.pi,np.pi))
        yy=210+amp*np.sin(2*np.pi*t+phase)+float(rng.uniform(-6,6))
        pts=np.column_stack((30+740*t,yy))
        templates.append({'frame':f'GENERATED_{seed}_NOT_VISUALMETRO',
                          'id':f'not-film070-{seed}-{i}','points':pts})
    return templates


def generated_photos(directory,seed):
    rng=np.random.default_rng(seed)
    yy,xx=np.indices((512,512))
    out=[]
    for i in range(3):
        noise=rng.normal(0,4+i,size=(512,512))
        phase=float(rng.uniform(0,6.3))
        photo=np.clip(202+10*np.sin(xx/(34+i*11)+phase)+7*np.cos(yy/(27+i*6))+noise,0,255).astype(np.uint8)
        p=directory/f'GENERATED_{seed}_{i}_NOT_H50Z.png'
        if not cv2.imwrite(str(p),photo):raise OSError('Could not write generated QA texture')
        out.append(p)
    return out


def make_split(phase, n, tempdir):
    if phase not in SEEDS or not 1<=n<=SCENE_N[phase]:
        raise ValueError('Only predeclared phase and scene budgets')
    seed=SEEDS[phase]
    templates=generated_templates(seed)
    photos=generated_photos(tempdir,seed)
    rng=np.random.default_rng(seed)
    for i in range(n):
        photo=photos[int(rng.integers(len(photos)))]
        scene=make_three_piece_scene(photo,templates,seed=int(rng.integers(1,2**31-1)),
                                     scenario=SCENARIOS[i%len(SCENARIOS)])
        noise_seed=int(rng.integers(1,2**31-1))
        yield [(condition_pixels(scene.gray,c,noise_seed=noise_seed),scene.nodes)
               for c in CONDITIONS]


def extract_pair_features(gray,detector):
    """Pixel-only candidate/row extraction. NEVER receives synthetic truth."""
    if gray.shape!=(192,192) or gray.dtype!=np.uint8:raise ValueError('Invalid pixel contract')
    # Validate to use the exact public pluggable detector interchange boundary.
    pix=gray.copy();pix.setflags(write=False)
    lines=validate_lines(list(detector.detect(pix)),max_lines=detector.max_lines)
    rows=filter_rows(pair_rows(list(lines)),'strict_s39') if len(lines)>=3 else []
    x=feature_matrix(rows,FIELDS) if rows else np.empty((0,len(FIELDS)),float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite 2D geometry')
    return lines,rows,x


def labels_after_blind_extraction(lines,rows,reference):
    """Synthetic generator identities ONLY here: never inside pixel inference."""
    assoc,_=_candidate_associations(list(lines),reference)
    y=[]
    for r in rows:
        i,j=int(r['i']),int(r['j'])
        a,b=assoc.get(i),assoc.get(j)
        label=0
        if a is not None and b is not None and tuple(sorted((a,b))) in INJECTED_TRUE_PAIRS:
            pa,pb=reference_facing_ports(reference[a],reference[b])
            if (np.linalg.norm(lines[i][int(r['end_i'])]-reference[a][pa])<=ASSOCIATION['max_each_endpoint_px'] and
                np.linalg.norm(lines[j][int(r['end_j'])]-reference[b][pb])<=ASSOCIATION['max_each_endpoint_px']):
                label=1
        y.append(label)
    return y


def train_model(scene_data):
    """Fit ONLY detector-specific TRAIN pairs and scene-level grouped labels."""
    xs=[];ys=[];groups=[]
    for scene_i,groups_per_cond in enumerate(scene_data):
        for lines,rows,x,y in groups_per_cond:
            assert len(x)==len(y)
            xs.extend(x);ys.extend(y);groups.extend([scene_i]*len(y))
    positives=int(sum(ys));negatives=len(ys)-positives
    unique_pos=len(set(g for g,y in zip(groups,ys) if y==1))
    enough=positives>=6 and negatives>=12 and unique_pos>=4
    receipt={'trained':bool(enough),'pairs':len(ys),'positive':positives,'negative':negatives,
             'independent_positive_source_scenes':unique_pos,'feature_fields':list(FIELDS),
             'model_type':'StandardScaler + regularized LogisticRegression' if enough else 'ABSTAIN'}
    if not enough:
        return None,receipt
    x=np.asarray(xs,dtype=float)
    scaler=StandardScaler();standard=scaler.fit_transform(x)
    clf=LogisticRegression(C=1.,class_weight='balanced',solver='liblinear',
                           random_state=RNG_MODEL,max_iter=200)
    clf.fit(standard,np.asarray(ys))
    h=sha256(np.concatenate((scaler.mean_,scaler.scale_,clf.coef_.ravel(),clf.intercept_)).astype('<f8').tobytes()).hexdigest()
    receipt['frozen_private_parameters_sha256']=h
    return (scaler,clf),receipt


def score_model(model):
    def score(rows):
        if not rows:return np.empty((0,),dtype=float)
        if model is None:return np.zeros((len(rows),),float)
        x=feature_matrix(rows,FIELDS)
        return model[1].predict_proba(model[0].transform(x))[:,1]
    return score


def predict_one(lines,scorer,cutoff,strategy):
    """No synthetic identity in this function; reuse the actual public solver API."""
    binding=ScoreBinding('S85_TOY_TRAINED_PER_DETECTOR','toy-generated',scorer,
                         cutoff,.04,'GENERATED_CAL_ONLY',strategy)
    r=assemble_lines(lines,binding,detector_id='toy-generated',solver_timeout=.12)
    from bubble_chamber.s56_global_path_assembly import Edge
    return {'polylines':list(r.lines),
            'chosen':[Edge(e.source_line,e.target_line,e.source_end,e.target_end,e.score) for e in r.joins],
            'pair_count':r.pairs_considered,'solver_success':r.solver_success,
            'physics_auto_joins_authorized':False,'added_pixel_instances':0,
            'duplicate_suppressed':0,'scale_consensus_rejected':0,
            'original_polylines_retained':len(r.lines)}


def stats(scene_records):
    return {**{k:int(sum(int(d[k]) for d in scene_records)) for k in KEYS},
            'scenes':len(scene_records)}


def acceptable(combined):
    """Frozen no-more-than-1-false-per-4-correct CAL risk cap, per condition."""
    all_correct=sum(d['selected_synthetic_correct_links'] for d in combined.values())
    all_wrong=sum(d['synthetic_noncorrect_selected'] for d in combined.values())
    if all_correct<2 or all_wrong>all_correct//4:
        return False
    return all(d['solver_failure']==0 and d['synthetic_noncorrect_selected']<=
               d['selected_synthetic_correct_links']//4 for d in combined.values())


def rank(combined,strategy,cutoff):
    complete=sum(v['complete_three_piece_paths_synthetic'] for v in combined.values())
    good=sum(v['selected_synthetic_correct_links'] for v in combined.values())
    bad=sum(v['synthetic_noncorrect_selected'] for v in combined.values())
    order={'port_greedy':0,'node_mutual':1,'global_milp':2,'none':3}
    return (-complete,-good,bad,order[strategy],-cutoff)


def cal_policy(observations):
    """Choose thresholds per strategy from CAL only; never read TEST."""
    policies={}
    qualifying=[]
    for strategy in STRATEGIES[1:]:
        options=[]
        for cutoff in CUTS:
            by_cond={cond:stats(observations[(strategy,cutoff,cond)]) for cond in CONDITIONS}
            if acceptable(by_cond):options.append((rank(by_cond,strategy,cutoff),cutoff,by_cond))
        if options:
            _,best,summary=min(options,key=lambda x:x[0])
            policies[strategy]={'threshold':float(best),'cal':summary,'eligible':True}
            qualifying.append((rank(summary,strategy,best),strategy))
        else:
            policies[strategy]={'threshold':1.0,'cal':None,'eligible':False}
    winner=min(qualifying)[1] if qualifying else 'none'
    return policies,winner


def paired(a,b,key='complete_three_piece_paths_synthetic'):
    delta=np.asarray([y[key]-x[key] for x,y in zip(a,b)],float)
    if not len(delta):raise ValueError('Empty paired CAL/TEST scenes')
    rng=np.random.default_rng(20268620)
    draws=rng.integers(0,len(delta),size=(1200,len(delta)))
    return {'delta_total':int(delta.sum()),'mean_delta_per_scene':float(delta.mean()),
            'paired_scene_bootstrap_95pct':[float(v) for v in np.quantile(delta[draws].mean(1),(.025,.975))]}


def run(*,train_n=SCENE_N['train'],cal_n=SCENE_N['cal'],test_n=SCENE_N['test']):
    if any(type(x) is not int or x<1 or x>SCENE_N[part] for part,x in
           zip(('train','cal','test'),(train_n,cal_n,test_n))):
        raise ValueError('Frozen scene budget exceeded')
    model_by_name={};model_receipts={}
    detectors={cls().detector_id:cls(max_lines=12) for cls in DETECTORS}
    # Deliberately run all detectors on EXACTLY the same scene for each phase.
    with tempfile.TemporaryDirectory(prefix='s85_cal_generated_only_') as temp:
        for phase,n in [('train',train_n),('cal',cal_n),('test',test_n)]:
            scenes=list(make_split(phase,n,Path(temp)))
            cache={name:[] for name in detectors}
            for scene in scenes:
                for name,det in detectors.items():
                    per_cond=[]
                    for gray,reference in scene:
                        lines,rows,x=extract_pair_features(gray,det)
                        labels=labels_after_blind_extraction(lines,rows,reference) if phase=='train' else None
                        per_cond.append((lines,rows,x,labels,reference))
                    cache[name].append(per_cond)
            if phase=='train':
                for name,cases in cache.items():
                    fitting=[[ (lines,rows,x,labels) for lines,rows,x,labels,ref in scene ] for scene in cases]
                    model,receipt=train_model(fitting)
                    model_by_name[name]=model
                    model_receipts[name]=receipt
            elif phase=='cal':
                frozen_policies={};choices={}
                for name,cases in cache.items():
                    scorer=score_model(model_by_name[name])
                    # Evaluate all CAL options without leaking synthetic labels into scoring.
                    observations={(a,t,c):[] for a in STRATEGIES[1:] for t in CUTS for c in CONDITIONS}
                    for case in cases:
                        for i,(lines,rows,x,labels,reference) in enumerate(case):
                            cond=CONDITIONS[i]
                            for strategy in STRATEGIES[1:]:
                                for cutoff in CUTS:
                                    predicted=predict_one(lines,scorer,cutoff,strategy)
                                    s=score_after_inference({'result':predicted},reference)['result']
                                    observations[(strategy,cutoff,cond)].append(s)
                    policies,winner=cal_policy(observations)
                    frozen_policies[name]=policies
                    choices[name]=winner
            else:
                test_results={};paired_results={}
                for name,cases in cache.items():
                    scorer=score_model(model_by_name[name])
                    observations={cond:{strategy:[] for strategy in STRATEGIES} for cond in CONDITIONS}
                    for case in cases:
                        for i,(lines,rows,x,labels,reference) in enumerate(case):
                            cond=CONDITIONS[i]
                            for strategy in STRATEGIES:
                                cutoff=(frozen_policies[name][strategy]['threshold']
                                        if strategy!='none' else 1.0)
                                pred=predict_one(lines,scorer,cutoff,strategy)
                                observations[cond][strategy].append(score_after_inference(
                                       {'result':pred},reference)['result'])
                    test_results[name]={c:{s:stats(observations[c][s]) for s in STRATEGIES} for c in CONDITIONS}
                    paired_results[name]={c:paired(observations[c]['port_greedy'],
                                                   observations[c]['global_milp']) for c in CONDITIONS}
    source=(ROOT/'scripts/research/bubble_chamber/interop_calibrated_benchmark.py')
    lock={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (
            source,PROTOCOL,ROOT/'bubble_chamber/interop_api.py',ROOT/'bubble_chamber/interop_detectors.py')}
    return {'schema':SCHEMA,'experiment_id':ID,'mode':'GENERATED_ONLY_NOT_H50Z',
            'source_locks':lock,'training_source_seeds':SEEDS,'train_scenes':train_n,'cal_scenes':cal_n,
            'test_scenes':test_n,'frozen_design_budgets':SCENE_N,'smoke_NOT_CONFIRMATORY':
                 (train_n,cal_n,test_n)!=(SCENE_N['train'],SCENE_N['cal'],SCENE_N['test']),
            'three_conditions':CONDITIONS,'detectors':list(detectors),'cal_frozen_before_test':True,
            'models':model_receipts,'CAL_strategies':frozen_policies,'CAL_selected':choices,
            'TEST_aggregates':test_results,'paired_global_minus_greedy_TEST':paired_results,
            'physical_GT_count':0,'physical_auto_joins_authorized':False,
            'learned_models_are_generated_proxy_only':True,
            'H50Z_native_physics_improvement_demonstrated':False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--train-scenes',type=int,default=SCENE_N['train'])
    p.add_argument('--cal-scenes',type=int,default=SCENE_N['cal'])
    p.add_argument('--test-scenes',type=int,default=SCENE_N['test'])
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Frozen generated CAL/TEST result exists')
    result=run(train_n=args.train_scenes,cal_n=args.cal_scenes,test_n=args.test_scenes)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for name in result['detectors']:
        print(name,'trained',result['models'][name]['trained'],'CAL selected',result['CAL_selected'][name],
              'TEST clean',result['TEST_aggregates'][name]['clean'][result['CAL_selected'][name]])
if __name__=='__main__':main()
