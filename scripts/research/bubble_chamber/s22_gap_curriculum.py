#!/usr/bin/env python3
"""S22 preregistered gap-curriculum ablation on new unseen frame splits.

This is a proxy experiment on manually measured coordinates with simulated
missing clicks; no RGB/BMP segmentation ground truth or independent film.
Train only from queried simulated expert pair reviews; unknown candidates stay
unknown. Match validation-review budgets and classifier hyperparameters.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s21_selective_graph import (
 S21Protocol,acquire_human_pairs,blind_validation_reviews,
 _support_cut,calibrate_zero_false_joins,accept_with_support,
 group_metrics,paired_frame_ci,fit_selective_models)
from scripts.research.bubble_chamber.s21_run_safe_graph import row_split,dataset_view

REGISTERED={
 'split_seeds':[20261027,20261028,20261029],
 'training_review_budgets':[320,640],
 'additional_calibration_reviews':160,
 'val_gaps':[3,5],
 'training_gaps_short':[1,3],
 'training_gaps_mixed':[1,3,5],
 'test_gaps':[3,5],
 'margin':0.05,
 'train_support_distance_quantile':0.995,
 'max_observed_validation_false_joins_on_reviewed_candidates':0,
 'classifier':'HistGradientBoostingClassifier, 80 trees, 9 leaves, min_leaf7, lr0.07, l2=5',
 'source':'visualMetro film 070 measured points, artificial missing clicks',
 'no_test_selection':True,
 'statistical_note':'Three overlapping same-film frame splits are not independent physical runs',
}


def fit_classifier(rows,x,reviewed,seed):
    y=np.asarray([int(rows[i]['truth_same_track']) for i in reviewed],int)
    if set(y)!={0,1}:raise ValueError('Need BOTH classes from queried training labels')
    m=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=9,
        min_samples_leaf=7,learning_rate=.07,l2_regularization=5,random_state=seed)
    m.fit(x[reviewed],y)
    return m


def study(manifest,registered=REGISTERED):
    if registered!=REGISTERED:raise ValueError('S22 preregistered parameters are frozen')
    reports=[];started=time.monotonic()
    for seed in REGISTERED['split_seeds']:
        pools,splits,availability=row_split(manifest,seed)
        val=dataset_view(pools[3]['val']+pools[5]['val'])
        val_review=blind_validation_reviews(val['rows'],REGISTERED['additional_calibration_reviews'],seed)
        tests={gap:dataset_view(pools[gap]['test']) for gap in REGISTERED['test_gaps']}
        arms={}
        for curriculum,gaps in [('short',(1,3)),('mixed',(1,3,5))]:
            train=dataset_view([r for g in gaps for r in pools[g]['train']])
            acquired,trail=acquire_human_pairs(train['rows'],train['context'],train['base'],
                max(REGISTERED['training_review_budgets']),seed=seed,
                warm_start=60,batch_size=40,max_per_frame=4)
            cut=_support_cut(train['base'],REGISTERED['train_support_distance_quantile'])
            for budget in REGISTERED['training_review_budgets']:
                review=acquired[:budget]
                for geometry,field in [('generic','base'),('graph_context','context')]:
                    model=fit_classifier(train['rows'],train[field],review,seed)
                    val_scores=model.predict_proba(val[field])[:,1]
                    cal=calibrate_zero_false_joins(val['rows'],val_scores,
                        margin=REGISTERED['margin'],dist_cut=cut,
                        reviewed_indices=val_review,max_false=0)
                    test_results={};test_binary={}
                    for gap in REGISTERED['test_gaps']:
                        data=tests[gap]
                        score=model.predict_proba(data[field])[:,1]
                        selected=accept_with_support(data['rows'],score,
                            cal['threshold'],REGISTERED['margin'],cut)
                        metric=group_metrics(data['rows'],selected)
                        metric['true_continuations_possible']=availability[gap]['test']['true_possible']
                        metric['end_to_end_recall']=metric['tp']/max(metric['true_continuations_possible'],1)
                        test_results[str(gap)]=metric
                        test_binary[gap]=selected
                    arms[(curriculum,budget,geometry)]={'calibration':cal,'test':test_results,'selected_binary':test_binary,
                        'training_labels':budget,'training_positive_labels':int(sum(train['rows'][i]['truth_same_track'] for i in review)),
                        'train_support_cut':cut}
        for budget in REGISTERED['training_review_budgets']:
            for gap in REGISTERED['test_gaps']:
                a=arms[('short',budget,'graph_context')]['selected_binary'][gap]
                b=arms[('mixed',budget,'graph_context')]['selected_binary'][gap]
                ci=paired_frame_ci(tests[gap]['rows'],a,b,seed=seed+gap,repeats=750)
                reports.append({'split_seed':seed,'training_label_budget':budget,'test_hidden_clicks':gap,
                    'short_graph':arms[('short',budget,'graph_context')]['test'][str(gap)],
                    'mixed_graph':arms[('mixed',budget,'graph_context')]['test'][str(gap)],
                    'short_generic_hgb':arms[('short',budget,'generic')]['test'][str(gap)],
                    'mixed_generic_hgb':arms[('mixed',budget,'generic')]['test'][str(gap)],
                    'graph_curriculum_paired_frame_ci':ci,
                    'calibration':{f'{curriculum}_{geometry}':arms[(curriculum,budget,geometry)]['calibration']
                                  for curriculum in ('short','mixed') for geometry in ('generic','graph_context')},
                    'train_positive_labels':{curriculum:arms[(curriculum,budget,'graph_context')]['training_positive_labels']
                                              for curriculum in ('short','mixed')}})
        print('S22 split',seed,'complete',round(time.monotonic()-started,1),'s',flush=True)
    return {'schema':'S22_FIXED_CROSS_GAP_CURRICULUM_2026_10_25',
        'registered_protocol':REGISTERED,'provenance_sha256':manifest['provenance_sha256'],
        'results':reports,'elapsed_seconds':round(time.monotonic()-started,2),
        'test_selection_occurred':False,'physical_release_allowed':False,
        'limitations':['synthetically hidden human measured points only, no real BMP track mask',
            '3 shuffled frame splits of SAME film 070, not independent runs',
            'all 160 calibration reviews are extra expert-label budget',
            'zero observed false joins among small calibration reviews cannot guarantee true error probability',
            'No inference of particle mass, magnetic field or identity']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    manifest=audited_large_manifest(args.measurements)
    result=study(manifest)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('RESULT',args.out)

if __name__=='__main__':main()
