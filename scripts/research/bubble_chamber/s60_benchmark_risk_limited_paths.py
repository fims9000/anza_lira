#!/usr/bin/env python3
"""S60: calibration-locked global-stability margins vs S56 baseline.

Never consider TEST results when selecting any source-photo margin. All tracked
identity labels are generated artificially, not verified bubble-chamber events.
"""
from __future__ import annotations
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from bubble_chamber.s35_fragment_link import GEOMETRY_FIELDS
from bubble_chamber.s56_global_path_assembly import scored_edges,select_global_forest
from bubble_chamber.s59_candidate_retrieval import filter_rows
from bubble_chamber.s60_margin_calibration import MARGINS,BASELINE_MARGIN,select_margin
from scripts.research.bubble_chamber.s59_benchmark_candidate_retrieval import (
    source_inputs,fit_model,sha256,rows_from_native_nodes,paired_interval)
from scripts.research.bubble_chamber.s56_benchmark_global_paths import generate

SCHEMA='s60_risk_limited_global_margin_machine_only_v1'
PROTOCOL=ROOT/'docs/bubble_chamber/S60_FROZEN_GRAPH_MARGIN_CALIBRATION_2026_10_26.md'
SEEDS={'005':(20266003,20266004),'006':(20266013,20266014)}


def score_scenes(cases,scorer,threshold:float):
    out=[]
    for c in cases:
        rows=filter_rows(c['rows'],'strict_s39')
        edges=scored_edges(rows,scorer(rows)) if rows else []
        out.append(edges)
    return out


def evaluate_edges(scenes:list[list],threshold:float,margin:float)->dict:
    counts=Counter();events=[]
    for edges in scenes:
        selected,qa=select_global_forest(edges,threshold=threshold,margin=margin)
        got={(x.i,x.j) for x in selected}
        truth={(0,1),(1,2),(3,4),(4,5)}
        wrong=got-truth
        correct=got&truth
        full=(int({(0,1),(1,2)}<=got and not any({x.i,x.j}&{0,1,2} for x in selected if (x.i,x.j) in wrong))+
              int({(3,4),(4,5)}<=got and not any({x.i,x.j}&{3,4,5} for x in selected if (x.i,x.j) in wrong)))
        event={'correct':len(correct),'wrong':len(wrong), 'any_wrong':int(bool(wrong)),
               'full_paths':full,'selected_links':len(got),
               'solver_failed':int(not qa['solver_success'])}
        events.append(event)
        counts['scenes']+=1
        counts['correct_selected']+=len(correct)
        counts['wrong_selected']+=len(wrong)
        counts['any_wrong_scenes']+=int(bool(wrong))
        counts['full_paths_selected']+=full
        counts['solver_failed_scenes']+=int(not qa['solver_success'])
        counts['links_selected']+=len(got)
    return {'counts':dict(counts),'events_PRIVATE_SYNTHETIC':events}


def calibrate(visualmetro:Path,tiles:Path,out:Path):
    manifest,templates,split_hash=source_inputs(visualmetro)
    result={'schema':SCHEMA,'phase':'calibration_frozen_before_fresh_test',
            'locked_protocol_sha256':sha256(PROTOCOL),
            'S37_frame_split_sha256':split_hash,
            'source_film_070_full_frames':len(manifest['frames']),
            'physically_validated_H50Z_track_labels':0,
            'cal_only_decision':{},'directions':{}}
    for source in ('005','006'):
        print('S60 CAL source',source,'training and frozen calibration',flush=True)
        scorer,threshold,cal=fit_model(tiles,source,templates)
        scenes=score_scenes(cal,scorer,threshold)
        evals={str(m):evaluate_edges(scenes,threshold,m)['counts'] for m in MARGINS}
        decision=select_margin(evals)
        result['directions'][source]={'pairwise_cutoff':threshold,'margin_grid_CAL_only':evals}
        result['cal_only_decision'][source]=decision
        print('S60 CAL decision',source,decision,'counts',
              {str(m): {k:evals[str(m)][k] for k in ('correct_selected','any_wrong_scenes','full_paths_selected')}
               for m in MARGINS},flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


def native_audit(private_graph:Path,scorers:dict,thresholds:dict,margins:dict,out:Path)->dict:
    d=json.loads(private_graph.read_text())
    if d.get('status')!='MACHINE_ONLY_NEVER_GROUND_TRUTH' or len(d.get('graphs',[]))!=39:
        raise ValueError('Expected S39 original machine-only private input')
    counts=Counter();output=[]
    for g in d['graphs']:
        target=g['photo_id']
        if target not in ('005','006'):raise ValueError('Unknown photo')
        source='006' if target=='005' else '005'
        rows=filter_rows(rows_from_native_nodes(g['nodes']),'strict_s39')
        original={(int(e['i']),int(e['j'])) for e in g['candidate_edges']}
        if {(r['i'],r['j']) for r in rows}!=original:
            raise ValueError('Original S39 native pair source mismatch')
        edges=scored_edges(rows,scorers[source](rows)) if rows else []
        a,qa1=select_global_forest(edges,threshold=thresholds[source],margin=BASELINE_MARGIN)
        b,qa2=select_global_forest(edges,threshold=thresholds[source],margin=margins[source])
        old={(e.i,e.j) for e in a};new={(e.i,e.j) for e in b}
        counts['tiles']+=1
        counts['source_machine_nodes']+=len(g['nodes'])
        counts['all_original_eligible_pairs']+=len(rows)
        counts['S56_baseline_machine_suggestions']+=len(old)
        counts['S60_cal_selected_machine_suggestions']+=len(new)
        counts['new_UNVERIFIED_machine_hypotheses']+=len(new-old)
        counts['previous_machine_hypotheses_not_retained']+=len(old-new)
        counts['solver_failed_scenes']+=int(not qa1['solver_success'] or not qa2['solver_success'])
        output.append({'crop_id_PRIVATE':g['crop_id'],'photo_id':target,
            'trained_and_calibrated_OPPOSITE_photo':source,
            'margin_frozen_before_TEST':margins[source],
            'S56_previous_suggestions_UNVERIFIED':sorted(map(list,old)),
            'S60_suggestions_UNVERIFIED':sorted(map(list,new)),
            'source_nodes_PRIVATE':[n['points_xy'] for n in g['nodes']],
            'native_particle_relation':'UNKNOWN','physically_verified_joins':0})
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'schema':'s60_native_cal_selected_advisory_private',
        'native_physics_GT_created':False,'native_physics_auto_join_authorized':False,
        'graphs':output},indent=2,ensure_ascii=False)+'\n')
    counts['native_physical_truth_labels']=0
    return dict(counts)


def test(visualmetro:Path,tiles:Path,cal:Path,out:Path,private_graph:Path|None,
         private_dir:Path|None):
    if not cal.exists():raise FileNotFoundError('S60 CAL decision must predate TEST')
    frozen=json.loads(cal.read_text())
    if (frozen.get('phase')!='calibration_frozen_before_fresh_test'
        or frozen.get('locked_protocol_sha256')!=sha256(PROTOCOL)):
        raise ValueError('Locked calibration protocol modified after freeze')
    manifest,templates,split_hash=source_inputs(visualmetro)
    if frozen['S37_frame_split_sha256']!=split_hash:
        raise ValueError('Frozen visualMetro frame-disjoint split changed')
    result={'schema':SCHEMA,'phase':'fresh_locked_test',
            'calibration_SHA256':sha256(cal),
            'S37_frame_split_sha256':split_hash,
            'physical_H50Z_native_GT_count':0,
            'real_H50Z_source_physical_runs':1,
            'original_models_unchanged':True,
            'directions':{},'frozen_CAL_decisions':frozen['cal_only_decision']}
    models={};thresholds={};margins={}
    for source,target in (('005','006'),('006','005')):
        print('S60 TEST',source,'->',target,flush=True)
        scorer,threshold,_=fit_model(tiles,source,templates)
        expected=frozen['directions'][source]['pairwise_cutoff']
        if abs(threshold-expected)>1e-12:
            raise ValueError('S60 CAL pair threshold cannot be reproduced')
        models[source]=scorer;thresholds[source]=threshold
        margin=frozen['cal_only_decision'][source]['selected_margin_CAL_only']
        margins[source]=margin
        results={}
        for seed in SEEDS[source]:
            fresh=generate(tiles,target,templates['test'],n=64,seed=seed)
            scored=score_scenes(fresh,scorer,threshold)
            base=evaluate_edges(scored,threshold,BASELINE_MARGIN)
            selected=evaluate_edges(scored,threshold,margin)
            results[str(seed)]={'n_simulated_scenes':64,
                'baseline_s56_margin':BASELINE_MARGIN,
                'margin_selected_CAL_only':margin,
                'S56_fixed_baseline':base['counts'],
                'S60_CAL_selected':selected['counts'],
                'paired_selected_minus_baseline_scene_bootstrap_95pct':paired_interval(
                    base['events_PRIVATE_SYNTHETIC'],selected['events_PRIVATE_SYNTHETIC'],seed+160)}
            print('S60 TEST',source,seed,'selected margin',margin,
                  'baseline correct/wrong',base['counts']['correct_selected'],base['counts']['wrong_selected'],
                  'selected correct/wrong',selected['counts']['correct_selected'],selected['counts']['wrong_selected'],flush=True)
        result['directions'][f'{source}_to_{target}']={'CAL_only_margin':margin,'fresh_tests':results}
    if private_graph is not None:
        if private_dir is None:raise ValueError('Native coordinates require private-only output')
        result['native_H50Z_unverified_diagnostic']=native_audit(
            private_graph,models,thresholds,margins,
            private_dir/'S60_NATIVE_MARGIN_ADVISORIES_PRIVATE.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return result


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
        if args.calibration is None:p.error('--calibration is required for test')
        test(args.visualmetro,args.photo_tiles,args.calibration,args.out,args.native_private_graph,args.private_dir)
if __name__=='__main__':main()
