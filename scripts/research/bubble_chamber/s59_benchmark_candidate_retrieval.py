#!/usr/bin/env python3
"""S59: calibrate *before* fresh test; native output is private machine hypotheses.

This benchmarks candidate retrieval, not image-based segmentation, on synthetic
shapes.  Real H50Z photo crops have NO validated physical identity labels.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_simulated_gaps import extract_track_templates
from bubble_chamber.s35_fragment_link import GEOMETRY_FIELDS,pair_features,optimal_endpoints,calibrated_negative_threshold
from bubble_chamber.s56_global_path_assembly import pair_rows,scored_edges,select_global_forest
from bubble_chamber.s59_candidate_retrieval import POLICIES,filter_rows
from scripts.research.bubble_chamber.s56_benchmark_global_paths import generate,flatten,fit
from scripts.research.bubble_chamber.s37_benchmark_review_risk import split_source_frames

SCHEMA='s59_locked_candidate_retrieval_machine_only_v1'
SEEDS={'train':{'005':20265601,'006':20265611},
       'cal':{'005':20265602,'006':20265612},
       'test':{'005':(20265903,20265904),'006':(20265913,20265914)}}
N_TRAIN=64;N_CAL=64;N_TEST=64
BASE='strict_s39'; EXPAND='relaxed_s59'; UPPER='all_pairs_oracle_upper_bound'
PROTOCOL=ROOT/'docs/bubble_chamber/S59_FROZEN_NATIVE_CANDIDATE_COVERAGE_PROTOCOL_2026_10_26.md'


def sha256(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_inputs(visualmetro:Path):
    manifest=audited_large_manifest(visualmetro)
    train,cal,test,split_sha=split_source_frames(manifest)
    templates={phase:extract_track_templates(manifest,chosen_frames=frames)
               for phase,frames in [('train',train),('cal',cal),('test',test)]}
    return manifest,templates,split_sha


def fit_model(tiles:Path,source:str,templates:dict):
    train=generate(tiles,source,templates['train'],n=N_TRAIN,seed=SEEDS['train'][source])
    cal=generate(tiles,source,templates['cal'],n=N_CAL,seed=SEEDS['cal'][source])
    train_rows,train_y=flatten(train)
    cal_rows,cal_y=flatten(cal)
    scorer=fit(train_rows,train_y)
    cutoff=calibrated_negative_threshold(scorer(cal_rows),cal_y,max_fpr=.05)
    return scorer,cutoff,cal


def evaluate_policy(cases:list[dict],scorer,cutoff:float,policy:str,*,full:bool=True):
    if policy not in POLICIES:raise ValueError('Unknown policy')
    counts=Counter();scene_records=[]
    for case in cases:
        rows=filter_rows(case['rows'],policy)
        true={(row['i'],row['j']) for row in rows if row['label_SYNTHETIC_ONLY']}
        counts['possible_true_joins']+=4
        counts['available_true_joins']+=len(true)
        counts['false_candidates']+=len(rows)-len(true)
        counts['candidate_pairs']+=len(rows)
        counts['full_path_available']+=int({(0,1),(1,2)}<=true)+int({(3,4),(4,5)}<=true)
        if not full:
            continue
        scores=scorer(rows) if rows else np.empty((0,),dtype=float)
        edges=scored_edges(rows,scores)
        selected,qa=select_global_forest(edges,threshold=cutoff,margin=.04,timeout=3.)
        if not qa['solver_success']:
            counts['solver_failed_scenes']+=1
        got={(e.i,e.j) for e in selected}
        false=got-{(0,1),(1,2),(3,4),(4,5)}
        correct=len(got)-len(false)
        full_path=int({(0,1),(1,2)}<=got and not any({e.i,e.j}&{0,1,2} for e in selected if (e.i,e.j) in false))
        full_path+=int({(3,4),(4,5)}<=got and not any({e.i,e.j}&{3,4,5} for e in selected if (e.i,e.j) in false))
        scene={'correct':correct,'wrong':len(false),'any_wrong':int(bool(false)),
               'full_paths':full_path,'selected_links':len(got)}
        scene_records.append(scene)
        counts['correct_selected']+=correct
        counts['wrong_selected']+=len(false)
        counts['any_wrong_scenes']+=int(bool(false))
        counts['full_paths_selected']+=full_path
        counts['selected_links']+=len(got)
    counts['scenes']=len(cases)
    counts['native_physics_GT_links']=0
    return {'counts':dict(counts),'scene_records_PRIVATE_SYNTHETIC':scene_records}


def cal_decision(assess:dict)->dict:
    strict=assess[BASE]['counts'];new=assess[EXPAND]['counts']
    reasons={'cal_true_recall_gain_at_least_10_percentage_points':
             (new['available_true_joins']-strict['available_true_joins'])>=.10*strict['possible_true_joins'],
             'cal_false_candidate_burden_at_most_4x':
             new['false_candidates']<=4*strict['false_candidates'],
             'cal_scene_wrong_join_events_not_worse':
             new['any_wrong_scenes']<=strict['any_wrong_scenes'],
             'cal_no_new_solver_failures':
             new.get('solver_failed_scenes',0)<=strict.get('solver_failed_scenes',0)}
    return {'policy':EXPAND if all(reasons.values()) else BASE,
            'criteria_satisfied':reasons,'selected_using':'CAL_ONLY_S59_FROZEN_GATE'}


def calibrate(visualmetro:Path,tiles:Path,out:Path):
    manifest,templates,split_hash=source_inputs(visualmetro)
    public={'schema':SCHEMA,'phase':'frozen_calibration','locked_protocol_sha256':sha256(PROTOCOL),
            'source_visualmetro_frame_split_sha256':split_hash,
            'source_complete_frames':len(manifest['frames']),
            'train_cal_scene_counts':{'train':N_TRAIN,'cal':N_CAL},
            'all_metrics_synthetic_only':True,'native_physics_verified':0,
            'decisions':{},'sources':{}}
    for source in ('005','006'):
        print(f'S59 CAL {source}: fit / synthetic calibration',flush=True)
        scorer,cutoff,cal=fit_model(tiles,source,templates)
        assess={p:evaluate_policy(cal,scorer,cutoff,p,full=(p!=UPPER)) for p in POLICIES}
        decision=cal_decision(assess)
        public['decisions'][source]=decision
        public['sources'][source]={'calibration_pair_cutoff':cutoff,
            'policies':{p:v['counts'] for p,v in assess.items()}}
        print('S59 CAL',source,decision,'recall',
              {p:v['counts']['available_true_joins'] for p,v in assess.items()},flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(public,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return public


def paired_interval(a:list[dict],b:list[dict],seed:int,boot:int=1500)->dict:
    if len(a)!=len(b):raise ValueError('Need paired scenes')
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(a),size=(boot,len(a)))
    return {key:{'mean_delta_per_simulated_scene':float(np.mean([y[key]-x[key] for x,y in zip(a,b)])),
                 'synthetic_scene_bootstrap_95pct':np.quantile(
                    np.asarray([y[key]-x[key] for x,y in zip(a,b)])[idx].mean(1),
                    [.025,.975]).tolist()}
            for key in ('correct','wrong','any_wrong','full_paths')}


def rows_from_native_nodes(nodes:list[dict])->list[dict]:
    if len(nodes)>32:raise ValueError('Too many native candidate nodes')
    result=[]
    for i,j in combinations(range(len(nodes)),2):
        a=np.asarray(nodes[i]['points_xy'],dtype=float)
        b=np.asarray(nodes[j]['points_xy'],dtype=float)
        features=pair_features(a,b,evidence={})
        orient=optimal_endpoints(a,b)
        result.append({'i':i,'j':j,'end_i':int(orient['endpoint_index_a']),
                       'end_j':int(orient['endpoint_index_b']),
                       **{name:float(features[name]) for name in GEOMETRY_FIELDS}})
    return result


def native_audit(private_graph:Path,models:dict,cutoffs:dict,choices:dict,private_out:Path):
    d=json.loads(private_graph.read_text())
    if d.get('status')!='MACHINE_ONLY_NEVER_GROUND_TRUTH' or len(d.get('graphs',[]))!=39:
        raise ValueError('Unknown private machine-only graph version')
    counters=Counter();private=[]
    for original in d['graphs']:
        photo=original['photo_id'];source='006' if photo=='005' else '005'
        if source not in choices:raise ValueError('Unexpected photo ID')
        rows=rows_from_native_nodes(original['nodes'])
        strict=filter_rows(rows,BASE)
        relaxed=filter_rows(rows,EXPAND)
        current=filter_rows(rows,choices[source])
        original_set={(int(x['i']),int(x['j'])) for x in original['candidate_edges']}
        strict_set={(x['i'],x['j']) for x in strict}
        if original_set!=strict_set:
            raise AssertionError('S59 strict retrieval did not reproduce native S39 candidate pairs')
        base_selected,base_qa=select_global_forest(
            scored_edges(strict,models[source](strict)) if strict else [],
            threshold=cutoffs[source],margin=.04)
        new_selected,new_qa=select_global_forest(
            scored_edges(current,models[source](current)) if current else [],
            threshold=cutoffs[source],margin=.04)
        baseline={(e.i,e.j) for e in base_selected}
        candidate={(e.i,e.j) for e in new_selected}
        counters['native_photo_tiles']+=1
        counters['native_machine_nodes']+=len(original['nodes'])
        counters['strict_pairs']+=len(strict)
        counters['relaxed_pairs']+=len(relaxed)
        counters['candidate_pool_deployed']+=len(current)
        counters['native_s56_strict_link_suggestions']+=len(baseline)
        counters['native_s59_selected_link_suggestions']+=len(candidate)
        counters['new_native_s59_links_without_physics_GT']+=len(candidate-baseline)
        counters['dropped_previous_machine_link_hypotheses']+=len(baseline-candidate)
        counters['native_solver_fail_closed']+=int(not base_qa['solver_success'] or not new_qa['solver_success'])
        private.append({'crop_id_PRIVATE':original['crop_id'], 'photo_id':photo,
              'calibrated_on_OPPOSITE_photo':source,
              'policy_cal_only_selected':choices[source],
              'native_truth':'UNKNOWN','native_auto_physics_joins_authorized':False,
              'original_s39_eligible_pairs':sorted(map(list,original_set)),
              'new_retrieval_candidate_pairs_UNVERIFIED':sorted(map(list,((x['i'],x['j']) for x in current))),
              'old_s56_suggestions_UNVERIFIED':sorted(map(list,baseline)),
              's59_deployed_suggestions_UNVERIFIED':sorted(map(list,candidate)),
              'source_2d_polylines_PRIVATE':[n['points_xy'] for n in original['nodes']]})
    private_out.parent.mkdir(parents=True,exist_ok=True)
    private_out.write_text(json.dumps({'schema':'s59_native_expanded_candidates_private_v1',
            'machine_only':True,'physics_verified_join_count':0,
            'graphs':private},ensure_ascii=False,indent=2)+'\n')
    counters['verified_physical_particle_joins']=0
    return dict(counters)


def test(visualmetro:Path,tiles:Path,calibration:Path,out:Path,private_graph:Path|None,
         private_dir:Path|None):
    if not calibration.exists():raise FileNotFoundError('S59 CAL-only decision file REQUIRED before TEST')
    c=json.loads(calibration.read_text())
    if c.get('phase')!='frozen_calibration' or c['locked_protocol_sha256']!=sha256(PROTOCOL):
        raise ValueError('Calibration freeze/protocol mismatch')
    manifest,templates,split_hash=source_inputs(visualmetro)
    if split_hash!=c['source_visualmetro_frame_split_sha256']:
        raise ValueError('S37 frozen source-frame split changed')
    public={'schema':SCHEMA,'phase':'fresh_locked_test','calibration_SHA256':sha256(calibration),
            'S37_frozen_frame_split_sha256':split_hash,
            'all_truth_synthetic_only':True,'real_H50Z_source_physical_runs':1,
            'native_verified_joins':0,'native_3D_physical_reconstruction':False,
            'sources':{},'calibration_only_deployment_policy':c['decisions']}
    models={};cutoffs={}
    for source,target in (('005','006'),('006','005')):
        print(f'S59 TEST locked fresh seeds {source}->{target}',flush=True)
        scorer,cutoff,_=fit_model(tiles,source,templates)
        if abs(cutoff-c['sources'][source]['calibration_pair_cutoff'])>1e-12:
            raise ValueError('Frozen cal cutoff cannot be reproduced')
        models[source]=scorer;cutoffs[source]=cutoff
        tests={}
        for seed in SEEDS['test'][source]:
            cases=generate(tiles,target,templates['test'],n=N_TEST,seed=seed)
            results={p:evaluate_policy(cases,scorer,cutoff,p,full=True) for p in POLICIES}
            strict=results[BASE]['scene_records_PRIVATE_SYNTHETIC']
            tests[str(seed)]={'n_synthetic_scenes':len(cases),'candidate_and_joins':
                {name:r['counts'] for name,r in results.items()},
                'paired_relaxed_minus_strict':paired_interval(strict,
                    results[EXPAND]['scene_records_PRIVATE_SYNTHETIC'],seed=seed+259),
                'paired_full_oracle_minus_strict':paired_interval(strict,
                    results[UPPER]['scene_records_PRIVATE_SYNTHETIC'],seed=seed+261)}
            print('S59 TEST',source,'seed',seed,
                  {p:(r['counts']['available_true_joins'],r['counts']['correct_selected'],
                      r['counts']['wrong_selected']) for p,r in results.items()},flush=True)
        public['sources'][source]={'heldout_test_photo':target,'test_seeds':tests}
    if private_graph is not None:
        if private_dir is None:raise ValueError('Private native output directory required')
        public['native_machine_graph_diagnostic_ONLY']=native_audit(
            private_graph,models,cutoffs,
            {s:c['decisions'][s]['policy'] for s in ('005','006')},
            private_dir/'S59_NATIVE_RETRIEVAL_PRIVATE.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(public,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return public


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('calibrate','test'))
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--calibration',type=Path)
    p.add_argument('--native-private-graph',type=Path)
    p.add_argument('--private-dir',type=Path)
    args=p.parse_args()
    if args.phase=='calibrate':calibrate(args.visualmetro,args.photo_tiles,args.out)
    else:
        if not args.calibration:p.error('--calibration required for TEST')
        test(args.visualmetro,args.photo_tiles,args.calibration,args.out,args.native_private_graph,args.private_dir)
if __name__=='__main__':main()
