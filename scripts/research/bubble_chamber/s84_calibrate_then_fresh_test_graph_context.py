#!/usr/bin/env python3
"""S84: frozen CAL-only graph-context gating + fresh opposite-H50Z test.

Only the post-hoc evaluator may inspect synthetic generator coordinates.
Never exports genuine H50Z pixels, source film points or per-shot coordinates.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np

from bubble_chamber.s56_three_piece_scenes import SCENARIOS,make_three_piece_scene
from bubble_chamber.s56_global_path_assembly import pair_rows,scored_edges,select_global_forest
from bubble_chamber.s59_candidate_retrieval import filter_rows
from bubble_chamber.s76_shape_endpoint_diagnostics import shape_only_match
from bubble_chamber.s80_multiscale_instances import augment_pixel_only
from bubble_chamber.s83_unsaturated_local_support import local_dark_stroke_support
from bubble_chamber.s84_graph_context_candidates import (
    ARMS,THRESHOLD_KEYS,thresholds_from_CAL_pixel_scores,gate_additional_lines,choose_CAL_only)
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75
from scripts.research.bubble_chamber import s80_calibrate_then_test_multiscale as s80

EXPERIMENT_ID='ANZA-LIRA-BC-S84-20261026'
SCHEMA='S84_GRAPH_CONTEXT_EXTRA_PIXEL_INSTANCES_CAL_THEN_FRESH_TEST_v1'
PROTOCOL=ROOT/'docs/bubble_chamber/S84_FROZEN_GRAPH_CONTEXT_EXTRA_INSTANCES_2026_10_26.md'
OPERATOR=ROOT/'bubble_chamber/s84_graph_context_candidates.py'
S83_OPERATOR=ROOT/'bubble_chamber/s83_unsaturated_local_support.py'
S80_OPERATOR=ROOT/'bubble_chamber/s80_multiscale_instances.py'
CAL_SEEDS={'005':20268401,'006':20268411}
TEST_SEEDS={'005':(20268403,20268404),'006':(20268413,20268414)}
CAL_SCENES=24
TEST_SCENES=24
CAL_CONDITIONS=('clean','blur_sigma_11')
TEST_CONDITIONS=('clean','blur_sigma_11','blur_sigma_16')
METRICS=(*s80.MEASURES,'extra_considered','extra_accepted','extra_rejected',
         'rejected_photo','rejected_weak_graph','rejected_conflict','rejected_ambiguous',
         'extra_shape_true_match','extra_shape_decoy_match')


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def _solve(lines,scorer,cutoff):
    if len(lines)<3:
        return [],0,True
    rows=filter_rows(pair_rows(lines),'strict_s39')
    if not rows:return [],0,True
    scores=np.asarray(scorer(rows),float)
    edges=scored_edges(rows,scores)
    chosen,qa=select_global_forest(edges,threshold=float(cutoff),margin=.04)
    return (chosen if qa['solver_success'] else []),len(rows),bool(qa['solver_success'])


def infer_from_pixels(gray:np.ndarray,scorer,cutoff:float,graph_arm:str,thresholds:dict):
    """Blind image interface: no GT, source geometry or physical identity argument."""
    if not isinstance(gray,np.ndarray) or gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('Only 192x192 uint8 image pixels')
    if set(thresholds)!=set(THRESHOLD_KEYS):raise ValueError('Saved source-CAL numeric thresholds required')
    # S80 uses EXACTLY the original frozen S72 source policy for baseline.
    old=s80.infer_from_pixels(gray,scorer,cutoff,graph_arm)
    base=old['baseline'];cons=old['s80_consensus_add6']
    source=base['polylines'];consensus=cons['polylines']
    out={'baseline':{**base,**{key:0 for key in (
        'extra_considered','extra_accepted','extra_rejected','rejected_photo',
        'rejected_weak_graph','rejected_conflict','rejected_ambiguous')}}}
    for arm in ARMS[1:]:
        gated=gate_additional_lines(gray,source,consensus,base['chosen'],scorer,cutoff,arm,thresholds)
        if gated['extra_accepted']==0:
            pred=base.copy()
        elif gated['extra_accepted']==len(consensus)-len(source):
            pred=cons.copy()
        else:
            chosen,nrows,ok=_solve(gated['polylines'],scorer,cutoff)
            pred={'polylines':gated['polylines'],'chosen':chosen,'pair_count':nrows,
                  'solver_success':ok,'physics_auto_joins_authorized':False,
                  'added_pixel_instances':gated['extra_accepted'],
                  'original_polylines_retained':len(source),
                  'duplicate_suppressed':cons['duplicate_suppressed'],
                  'scale_consensus_rejected':cons['scale_consensus_rejected']}
        out[arm]={**pred,**{k:gated[k] for k in (
            'extra_considered','extra_accepted','extra_rejected','rejected_photo',
            'rejected_weak_graph','rejected_conflict','rejected_ambiguous')}}
        assert out[arm]['added_pixel_instances']==gated['extra_accepted']
        assert all(a is b for a,b in zip(source,out[arm]['polylines']))
        assert not out[arm]['physics_auto_joins_authorized']
    return out


def _source_CAL_pixel_thresholds(photos,templates,seed,n,scorer,cutoff,graph_arm):
    """Only SOURCE-CAL image pixels contribute; no synthetic truth labels used."""
    rng=np.random.default_rng(seed)
    vals=[]
    for k in range(n):
        scene=make_three_piece_scene(photos[int(rng.integers(len(photos)))],templates,
            seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
        for condition in CAL_CONDITIONS:
            gray=s75.condition_pixels(scene.gray,condition)
            baseline=s68.infer_pixel_only(gray,scorer,cutoff,graph_arm)
            augmented=augment_pixel_only(gray,baseline['polylines'],'s80_consensus_add6')
            vals.extend(local_dark_stroke_support(gray,p)
                        for p in augmented['polylines'][len(baseline['polylines']):])
    return thresholds_from_CAL_pixel_scores(vals),len(vals)


def _posthoc_score(predictions,source_nodes):
    """Synthetic GT is examined here AFTER all arms finish prediction."""
    aggregated=s80.score_after_inference(predictions,source_nodes)
    results={}
    for arm,pred in predictions.items():
        matched,_,_=shape_only_match(pred['polylines'],source_nodes)
        nbase=int(pred['original_polylines_retained'])
        z={**aggregated[arm],**{k:int(pred[k]) for k in (
            'extra_considered','extra_accepted','extra_rejected','rejected_photo',
            'rejected_weak_graph','rejected_conflict','rejected_ambiguous')},
            'extra_shape_true_match':int(sum(i>=nbase and j<6 for i,j in matched.items())),
            'extra_shape_decoy_match':int(sum(i>=nbase and j==6 for i,j in matched.items()))}
        if set(METRICS)-set(z):raise AssertionError('Missing S84 after-inference metric')
        if z['added_pixel_instances']!=z['extra_accepted']:
            raise AssertionError('Changed source-instance accounting')
        results[arm]=z
    return results


def sample(photos,templates,seed,n,conditions,scorer,cutoff,graph_arm,thresholds):
    rng=np.random.default_rng(seed)
    outcomes={condition:{arm:[] for arm in ARMS} for condition in conditions}
    for k in range(n):
        scene=make_three_piece_scene(photos[int(rng.integers(len(photos)))],templates,
            seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
        for cond in conditions:
            pixels=s75.condition_pixels(scene.gray,cond)
            blind=infer_from_pixels(pixels,scorer,cutoff,graph_arm,thresholds)
            assessed=_posthoc_score(blind,scene.nodes)
            for arm in ARMS:outcomes[cond][arm].append(assessed[arm])
        if (k+1)%8==0:print('S84 scene source-seed',seed,k+1,'/',n,flush=True)
    return outcomes


def aggregate(rows):
    if not rows or any(set(METRICS)-set(row) for row in rows):
        raise ValueError('Missing S84 scene audit')
    return {**{k:int(sum(row[k] for row in rows)) for k in METRICS},'scenes':len(rows)}


def paired(a,b,seed):
    if not a or len(a)!=len(b):raise ValueError('Pair synthetic scenes, never pixels')
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(a),size=(1200,len(a)))
    keys=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
          'complete_three_piece_paths_synthetic','shape_true_fragments_matched',
          'shape_true_one_or_both_endpoints_far','scene_any_noncorrect',
          'extra_shape_true_match','extra_shape_decoy_match')
    out={}
    for k in keys:
        diff=np.asarray([q[k]-p[k] for p,q in zip(a,b)],float)
        out[k]={'delta_total':int(diff.sum()),'mean_delta_per_scene':float(diff.mean()),
                'paired_scene_bootstrap_95pct':[float(v) for v in np.quantile(diff[idx].mean(1),(.025,.975))]}
    return out


def run(visualmetro:Path,tiles:Path,phase:str,calibration:Path|None=None,*,smoke=False):
    if phase not in ('cal','test'):raise ValueError('Phase must be CAL or TEST')
    if digest(s80.S72_CAL)!=s80.LOCKED_S72_SHA256:
        raise ValueError('S72 source CAL changed')
    previous80=json.loads(s80.ROOT.joinpath('docs/bubble_chamber/S80_PUBLIC_FROZEN_CAL_2026_10_26.json').read_text())
    if previous80['operator_sha256']!=digest(S80_OPERATOR):
        raise ValueError('S80 multiscale source changed')
    locked72=s75.frozen_s72(s80.S72_CAL)
    templates,split_sha=s72._source_data(visualmetro)
    if split_sha!=s80.SPLIT_SHA:raise ValueError('Original film-070 source-frame split changed')
    photos={s:sorted(tiles.glob(f'H50Z_030_01_{s}_*.png')) for s in ('005','006')}
    if [len(photos[s]) for s in ('005','006')]!=[21,18]:
        raise ValueError('Need exact 39 original H50Z real source photo crops')
    report={'schema':SCHEMA,'experiment_id':EXPERIMENT_ID,'phase':phase,
            'smoke_NOT_CONFIRMATORY':smoke,'protocol_sha256':digest(PROTOCOL),
            'operator_sha256':digest(OPERATOR),'S80_operator_sha256':digest(S80_OPERATOR),
            'S83_photometric_source_sha256':digest(S83_OPERATOR),
            'S72_frozen_CAL_sha256':s80.LOCKED_S72_SHA256,
            'S72_source_code_locks':s72.LOCKS,'film070_whole_frame_split_sha256':split_sha,
            'H50Z_photo_backgrounds_one_physical_run':True,
            'source_film070_NOT_H50Z_native_GT':True,'physical_GT_count':0,
            'physical_auto_joins_authorized':False,'synthetic_only_metrics':True,
            'arms':ARMS,'directions':{}}
    frozen=None
    if phase=='test':
        if calibration is None or not calibration.is_file():
            raise FileNotFoundError('Test requires saved immutable source-CAL artifact')
        frozen=json.loads(calibration.read_text(encoding='utf8'))
        for key in ('schema','protocol_sha256','operator_sha256','S80_operator_sha256',
                    'S83_photometric_source_sha256','S72_frozen_CAL_sha256',
                    'S72_source_code_locks','film070_whole_frame_split_sha256'):
            if frozen.get(key)!=report[key]:raise ValueError('Frozen S84 CAL changed: '+key)
        if frozen.get('phase')!='cal' or frozen.get('smoke_NOT_CONFIRMATORY'):
            raise ValueError('Only confirmatory full CAL may select S84')
        report['frozen_CAL_sha256']=digest(calibration)
    for direction_idx,source in enumerate(('005','006')):
        target='006' if source=='005' else '005';direction=f'{source}_to_{target}'
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        old_arm=locked72['directions'][direction]['CAL_selected']
        if float(cutoff)!=float(locked72['directions'][direction]['S56_frozen_CAL_cutoff']):
            raise ValueError('Original trained source score cutoff drift')
        item={'source_photo':source,'heldout_photo':target,
              'S72_frozen_graph_arm':old_arm,'S56_frozen_CAL_cutoff':float(cutoff)}
        if phase=='cal':
            thresholds,n_photo_scores=_source_CAL_pixel_thresholds(photos[source],templates['cal'],
                          CAL_SEEDS[source],2 if smoke else CAL_SCENES,scorer,cutoff,old_arm)
            samples=sample(photos[source],templates['cal'],CAL_SEEDS[source],
                   2 if smoke else CAL_SCENES,CAL_CONDITIONS,scorer,cutoff,old_arm,thresholds)
            grid={arm:{condition:aggregate(samples[condition][arm]) for condition in CAL_CONDITIONS}
                  for arm in ARMS}
            item['CAL_only_thresholds']=thresholds
            item['CAL_only_pixel_scores_count']=n_photo_scores
            item['CAL_arms']=grid
            item['CAL_choice']=choose_CAL_only(grid)
            print('S84 CAL',direction,'q25/q40',thresholds,'decision',item['CAL_choice'],flush=True)
        else:
            x=frozen['directions'][direction]
            if x['S72_frozen_graph_arm']!=old_arm or float(x['S56_frozen_CAL_cutoff'])!=cutoff:
                raise ValueError('S84 source-CAL scorer/graph arm drift')
            th=x['CAL_only_thresholds'];chosen=x['CAL_choice']['CAL_only_selected']
            if set(th)!=set(THRESHOLD_KEYS) or chosen not in ARMS:
                raise ValueError('Invalid previously frozen S84 choice')
            item['frozen_CAL_thresholds']=th;item['frozen_CAL_selection']=chosen
            item['TEST_seeds']={}
            for si,seed in enumerate(TEST_SEEDS[source] if not smoke else TEST_SEEDS[source][:1]):
                samples=sample(photos[target],templates['test'],seed,
                       2 if smoke else TEST_SCENES,TEST_CONDITIONS,scorer,cutoff,old_arm,th)
                item['TEST_seeds'][str(seed)]={condition:{
                    'arms':{arm:aggregate(samples[condition][arm]) for arm in ARMS},
                    'paired_vs_baseline':{arm:paired(samples[condition]['baseline'],samples[condition][arm],
                        20268490+direction_idx*100+si*10+ci) for arm in ARMS[1:]}}
                    for ci,condition in enumerate(TEST_CONDITIONS)}
                print('S84 TEST',direction,'seed',seed,'choice',chosen,flush=True)
        report['directions'][direction]=item
    if phase=='test':
        report['all_TEST_aggregates']={cond:{arm:{k:int(sum(seedrow[cond]['arms'][arm][k]
             for d in report['directions'].values() for seedrow in d['TEST_seeds'].values()))
             for k in (*METRICS,'scenes')} for arm in ARMS} for cond in TEST_CONDITIONS}
        report['frozen_CAL_selected_TEST']={cond:{k:int(sum(
             seedrow[cond]['arms'][d['frozen_CAL_selection']][k]
             for d in report['directions'].values() for seedrow in d['TEST_seeds'].values()))
             for k in (*METRICS,'scenes')} for cond in TEST_CONDITIONS}
        report['physical_deployment_authorized']=False
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=('cal','test'))
    ap.add_argument('--visualmetro',type=Path,required=True)
    ap.add_argument('--photo-tiles',type=Path,required=True)
    ap.add_argument('--calibration',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Immutable CAL/TEST file already exists')
    result=run(args.visualmetro,args.photo_tiles,args.phase,args.calibration,smoke=args.smoke)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    print('S84 wrote',args.out,'sha256',digest(args.out),flush=True)
if __name__=='__main__':main()
