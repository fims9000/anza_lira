#!/usr/bin/env python3
"""S21 expert-label-efficient Graph-LIRA benchmark with held-out longer gaps.

Reproduction on 1mH2Data measurements from film 070. Every sample's geometry
is real but hidden measured clicks / occlusions are artificial. Whole frames
are disjoint; independent physical films are not available.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.large_relation_study import frame_partition
from bubble_chamber.real_sparse_lira import SparseProtocol,make_examples
from bubble_chamber.s21_selective_graph import (
    S21Protocol,grouped_graph_features,acquire_human_pairs,fit_selective_models,
    blind_validation_reviews,_support_cut,accept_with_support,calibrate_zero_false_joins,
    group_metrics,paired_frame_ci)


def row_split(manifest,seed:int):
    splits=frame_partition([f['frame_id'] for f in manifest['frames']],seed)
    result={};availability={}
    for gap in (1,3,5):
        cfg=SparseProtocol(seed=seed,missing_clicks=gap,cuts=(.5,),k_candidates=5,
                           missing_right_fraction=.18)
        rows,stats=make_examples(manifest,cfg)
        for r in rows:r['scene_id']=f'gap{gap}|'+r['scene_id']
        result[gap]={part:[r for r in rows if r['frame_id'] in set(fids)] for part,fids in splits.items()}
        availability[gap]={}
        for part,fids in splits.items():
            scenes=[s for s in stats['scenes'] if s['frame_id'] in set(fids)]
            availability[gap][part]={'candidate_true':sum(x['positive_in_candidates'] for x in scenes),
                                    'true_possible':sum(x['positive_possible'] for x in scenes),
                                    'whole_frames':len(fids),
                                    'unique_candidate_scenes':len(scenes)}
    return result,splits,availability


def dataset_view(rows):
    base,ctx=grouped_graph_features(rows)
    return {'rows':rows,'base':base,'context':ctx}


def study(manifest,protocol:S21Protocol,split_seeds=(20261024,20261025,20261026),
          cal_review_budget:int=160):
    protocol.validate()
    if manifest['audit_schema']!='visualmetro_large_audit_v1':raise ValueError('audited source only')
    if manifest['summary']['distinct_film_ids']!=['070']:
        raise ValueError('S21 applies only to audited film 070')
    all_rows=[];split_reports=[]
    started=time.monotonic()
    for seed in split_seeds:
        pools,splits,avail=row_split(manifest,seed)
        train_rows=pools[1]['train']+pools[3]['train']
        train=dataset_view(train_rows)
        val=dataset_view(pools[3]['val'])
        tests={k:dataset_view(pools[k]['test']) for k in (1,3,5)}
        if set(splits['train'])&set(splits['val']) or set(splits['train'])&set(splits['test']) or set(splits['val'])&set(splits['test']):
            raise AssertionError('source frame leakage')
        reviewed_ids,_=acquire_human_pairs(train['rows'],train['context'],train['base'],
                max(protocol.label_budgets),seed=seed,warm_start=protocol.warm_start,
                batch_size=protocol.review_batch,
                max_per_frame=protocol.max_from_one_frame_per_batch)
        val_review=blind_validation_reviews(val['rows'],cal_review_budget,seed=seed)
        # Train-only support clipping; same for all models and budgets.
        support_distance=_support_cut(train['base'],protocol.support_train_quantile)
        by_budget={}
        for budget in protocol.label_budgets:
            chosen=reviewed_ids[:budget]
            models=fit_selective_models(train['rows'],train['base'],train['context'],chosen,seed)
            scores_by_model={}
            calibrations={}
            evaluated={}
            for modelname,(model,group) in models.items():
                features=val[group]
                vprob=model.predict_proba(features)[:,1]
                cal=calibrate_zero_false_joins(val['rows'],vprob,
                    margin=protocol.margin,dist_cut=support_distance,
                    reviewed_indices=val_review,max_false=protocol.cal_max_false_joins)
                calibrations[modelname]=cal
                evaluated[modelname]={}
                scores_by_model[modelname]={}
                for gap,t in tests.items():
                    probs=model.predict_proba(t[group])[:,1]
                    accepted=accept_with_support(t['rows'],probs,cal['threshold'],protocol.margin,support_distance)
                    met=group_metrics(t['rows'],accepted)
                    met['true_continuations_possible']=avail[gap]['test']['true_possible']
                    met['candidate_generation_recall']=round(avail[gap]['test']['candidate_true']/max(avail[gap]['test']['true_possible'],1),6)
                    met['end_to_end_recall_at_observed_clicks']=round(met['tp']/max(avail[gap]['test']['true_possible'],1),6)
                    evaluated[modelname][str(gap)]=met
                    scores_by_model[modelname][gap]=accepted
            for gap in (1,3,5):
                a=scores_by_model['geometry_logreg'][gap]
                b=scores_by_model['context_hgb'][gap]
                compare=paired_frame_ci(tests[gap]['rows'],a,b,seed=seed+gap,repeats=protocol.bootstrap_repeats)
                all_rows.append({'split_seed':seed,'training_labels':budget,
                                 'additional_validation_labels':cal_review_budget,
                                 'tested_missing_clicks':gap,
                                 'baseline':evaluated['geometry_logreg'][str(gap)],
                                 'graph_context':evaluated['context_hgb'][str(gap)],
                                 'context_logreg':evaluated['context_logreg'][str(gap)],
                                 'paired_graph_minus_baseline':compare,
                                 'support_distance_from_train':support_distance,
                                 'calibration':calibrations,
                                 'train_class_balance':{'positive':int(sum(train_rows[i]['truth_same_track'] for i in chosen)),
                                                        'negative':int(sum(not train_rows[i]['truth_same_track'] for i in chosen))}})
            by_budget[str(budget)]={'training_labels':len(chosen),'calibration_reviewed':len(val_review)}
        split_reports.append({'seed':seed,'frame_partition':splits,
                              'candidate_availability':avail,
                              'train_label_acquisition_last_budget':by_budget,
                              'training_candidate_count':len(train_rows),
                              'validation_candidate_count':len(val['rows']),
                              'source_film_ids':['070']})
        print('S21 split',seed,'done in',round(time.monotonic()-started,1),'s',flush=True)
    report={'schema':'S21_SELECTIVE_GRAPH_LIRA_FEWSHOT_GAP_SHIFT_V1',
            'source_sha256':manifest['provenance_sha256'],
            'source_summary':manifest['summary'],
            'protocol':asdict(protocol),
            'validation_pairs_in_addition_to_training':cal_review_budget,
            'frame_split_seeds':list(split_seeds),
            'split_reports':split_reports,
            'results':all_rows,
            'study_limits':[
                'All measured clicks from film 070; no independent medium or chamber/run transfer',
                'Artificially hidden EXPERT clicks, not genuine photographic discontinuities',
                'Graph context only uses measured xy geometry; neither authentic AZConv pixel features nor true image-based Graph-LIRA train in this benchmark',
                'A human review label equals ground-truth track identity from the archive, revealed ONLY for preselected train/validation candidates',
                'Calibration depends on a small held-out reviewed set; zero observed mistakes there is NOT an expected physical error guarantee',
                'No unattended release or real particle identity, pion/neutron classification, B-field momentum or 3D claim',
            ],
            'release_to_physical_auto_join':False,
            'total_elapsed_seconds':round(time.monotonic()-started,2)}
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[20261024,20261025,20261026])
    p.add_argument('--budgets',type=int,nargs='+',default=[160,320,640])
    p.add_argument('--cal-reviews',type=int,default=160)
    args=p.parse_args()
    manifest=audited_large_manifest(args.measurements)
    cfg=S21Protocol(label_budgets=tuple(args.budgets))
    res=study(manifest,cfg,split_seeds=tuple(args.seeds),cal_review_budget=args.cal_reviews)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('RESULTS:',args.out,'seconds',res['total_elapsed_seconds'],flush=True)

if __name__=='__main__':main()
