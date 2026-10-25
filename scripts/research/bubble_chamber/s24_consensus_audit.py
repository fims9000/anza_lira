#!/usr/bin/env python3
"""S24 registered consensus/abstention study on new SAME-FILM frame splits.

- 640 training pseudo-human pair reviews from real expert coordinate tracks.
- 80 additional calibration reviews on separate validation frames.
- 80 independent threshold-frozen audit reviews on disjoint validation frames.
- Hold out entire TEST frames for final metrics (synthetic missing clicks).
- Never use test truths for acquisition/training/thresholds. Test oracle reviews
  are evaluation-only counterfactual human responses, not autonomous links.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s21_selective_graph import (
    acquire_human_pairs,blind_validation_reviews,_support_cut,
    calibrate_zero_false_joins,accept_with_support,group_metrics,paired_frame_ci,
)
from bubble_chamber.s24_consensus_review import (
    AuditConfig,split_validation_frames,unlabeled_audit_indices,
    conservative_consensus,route_provisional_disagreements,
    replay_review_answers,audit_labeled_subset,
)
from scripts.research.bubble_chamber.s21_run_safe_graph import row_split,dataset_view
from scripts.research.bubble_chamber.s22_gap_curriculum import fit_classifier


def _slice(full:dict,indices:np.ndarray)->dict:
    ix=np.asarray(indices,dtype=int)
    return {'rows':[full['rows'][i] for i in ix],
            'base':full['base'][ix], 'context':full['context'][ix]}


def _score(model,features):
    return np.asarray(model.predict_proba(features)[:,1],float)


def _hash_queue(rows:list[dict],options:list[int],seed:int)->list[int]:
    return sorted(options,key=lambda i:sha256(
       f"{seed}|{rows[i]['scene_id']}|{rows[i]['left_id']}|{rows[i]['right_id']}".encode()).digest())


def study(manifest:dict,cfg:AuditConfig=AuditConfig()):
    cfg.validate()
    if manifest.get('audit_schema')!='visualmetro_large_audit_v1':
        raise ValueError('requires complete audited visualMetro source')
    if manifest['summary']['distinct_film_ids']!=['070']:
        raise ValueError('only registered single film 070 is supported')
    results=[];started=time.monotonic()
    for seed in cfg.split_seeds:
        pools,frames,availability=row_split(manifest,seed)
        train=dataset_view([r for gap in cfg.training_gaps for r in pools[gap]['train']])
        validation=dataset_view(pools[3]['val']+pools[5]['val'])
        test=dataset_view(pools[cfg.test_gap]['test'])
        train_frames=set(frames['train']);test_frames=set(frames['test']);val_frames=set(frames['val'])
        if train_frames&test_frames or train_frames&val_frames or test_frames&val_frames:
            raise ValueError('frame split contaminated')
        # Different original validation frames, disjoint by construction.
        cal_ids,audit_ids=split_validation_frames(validation['rows'],seed)
        cal=_slice(validation,cal_ids);audit=_slice(validation,audit_ids)
        calibrators=blind_validation_reviews(cal['rows'],cfg.calibration_reviews,seed)
        audit_review=unlabeled_audit_indices(audit['rows'],cfg.independent_audit_reviews,seed)
        labels,_=acquire_human_pairs(train['rows'],train['context'],train['base'],cfg.training_reviews,
                                    seed=seed,warm_start=60,batch_size=40,max_per_frame=4)
        dist_cut=_support_cut(train['base'],cfg.support_quantile)
        models={}
        calstats={}
        auditstats={}
        val_label_set={}
        tests={}
        for key,field in [('geometry_hgb','base'),('graph_context_hgb','context')]:
            m=fit_classifier(train['rows'],train[field],labels,seed)
            models[key]=(m,field)
            vscore=_score(m,cal[field])
            threshold=calibrate_zero_false_joins(cal['rows'],vscore,margin=cfg.matching_margin,
                dist_cut=dist_cut,max_false=0,reviewed_indices=calibrators)
            calstats[key]=threshold
            a_score=_score(m,audit[field])
            a_binary=accept_with_support(audit['rows'],a_score,threshold['threshold'],cfg.matching_margin,dist_cut)
            auditstats[key]=audit_labeled_subset(audit['rows'],a_binary,audit_review)
            test_score=_score(m,test[field])
            t_binary=accept_with_support(test['rows'],test_score,threshold['threshold'],cfg.matching_margin,dist_cut)
            tests[key]={'scores':test_score,'accepted':t_binary,'test_metrics':group_metrics(test['rows'],t_binary)}
        geometry=tests['geometry_hgb'];graph=tests['graph_context_hgb']
        cons=conservative_consensus(test['rows'],geometry['accepted'],graph['accepted'])
        consensus_metrics=group_metrics(test['rows'],cons)
        # The CONSENSUS is evaluated on the exact same candidate rows and may
        # NOT exceed either base in accepted links, a crucial safety invariant.
        if (cons&~geometry['accepted']).any() or (cons&~graph['accepted']).any():
            raise AssertionError('consensus was not an intersection')
        scored_order=route_provisional_disagreements(test['rows'],geometry['accepted'],graph['accepted'],
                                                     geometry['scores'],graph['scores'],seed)
        random_order=_hash_queue(test['rows'],scored_order,seed)
        reviewed_rank=replay_review_answers(test['rows'],cons,scored_order,cfg.review_budgets)
        reviewed_hash=replay_review_answers(test['rows'],cons,random_order,cfg.review_budgets)
        possible=availability[cfg.test_gap]['test']['true_possible']
        for arm in [*tests.values()]:arm['test_metrics']['end_to_end_recall']=arm['test_metrics']['tp']/max(possible,1)
        consensus_metrics['end_to_end_recall']=consensus_metrics['tp']/max(possible,1)
        results.append({'split_seed':seed,'original_frame_counts':{x:len(y) for x,y in frames.items()},
                        'data_counts':{'training_candidate_pool':len(train['rows']),
                            'validation_calibration_candidate_pool':len(cal['rows']),
                            'validation_audit_candidate_pool':len(audit['rows']),
                            'test_gap5_candidate_pool':len(test['rows']),
                            'test_possible_true_continuations':possible,
                            'test_positive_candidates':int(sum(r['truth_same_track'] for r in test['rows']))},
                        'training_reviewed_pairs':len(labels),
                        'training_reviewed_positives':int(sum(train['rows'][i]['truth_same_track'] for i in labels)),
                        'additional_calibration_pairs':len(calibrators),
                        'additional_audit_pairs':len(audit_review),
                        'calibration_frames':sorted({r['frame_id'] for r in cal['rows']}),
                        'audit_frames':sorted({r['frame_id'] for r in audit['rows']}),
                        'model_calibration':calstats,'independent_audit_pairs':auditstats,
                        'geometry_hgb':geometry['test_metrics'],
                        'graph_context_hgb':graph['test_metrics'],
                        'consensus':consensus_metrics,
                        'paired_graph_minus_consensus':paired_frame_ci(test['rows'],cons,graph['accepted'],seed+19,repeats=750),
                        'paired_geometry_minus_consensus':paired_frame_ci(test['rows'],cons,geometry['accepted'],seed+23,repeats=750),
                        'test_disagreement_candidates_rankable':len(scored_order),
                        'review_simulation':{'uncertainty_diversity_queue':reviewed_rank,'hash_control':reviewed_hash},
                        'all_predictions_are_offline':True,
                        'physical_release_authorized':False})
        print('S24 seed',seed,'complete',round(time.monotonic()-started,1),
              's consensus',consensus_metrics['tp'],consensus_metrics['fp'],flush=True)
    return {'schema':'S24_CONSENSUS_SELECTIVE_AUDIT_V1',
        'registered_config':asdict(cfg),'source_sha256':manifest['provenance_sha256'],
        'source_film_ids':['070'],'results':results,'timing_seconds':round(time.monotonic()-started,2),
        'real_pixel_level_ground_truth':False,'true_independent_run_test':False,
        'test_labels_used_for_fit_or_calibration':False,
        'test_labels_used_for_offline_review_simulation':True,
        'physical_auto_merge_allowed':False,
        'limitations':[
            'Original geometry is manually measured; missing clicks and review responses are synthetic simulation, not labels on real H50Z photographs',
            'Three new frame-disjoint partitions of the SAME film 070, not independent chamber runs or liquid fillings',
            'Calibration (80) and audit (80) are extra reviewed pairs beyond 640 training decisions per split',
            'Audit reviews are a hash-ranked candidate sample on same-film frames; exact binomial iid bound does NOT certify real-world false-join risk',
            'Consensus may sacrifice a large number of correct joins, and any positive auto join must remain provisional',
            'Only expert-confirmed real BMP instance pairs and separate physical-run tests can permit safety/accuracy claims',
            'Review simulation uses held-out truth only AFTER selecting candidate review order and only for test reporting',
        ]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    cfg=AuditConfig();cfg.validate()
    m=audited_large_manifest(args.measurements)
    r=study(m,cfg)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('RESULT',args.out)

if __name__=='__main__':main()
