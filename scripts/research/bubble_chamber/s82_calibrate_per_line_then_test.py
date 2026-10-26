#!/usr/bin/env python3
"""S82: fixed local line photometric support thresholds, CAL then fresh TEST.

Machine inference sees only pixels and source-CAL pretrained geometry policy.
Simulator coordinates enter post-hoc evaluation only AFTER all arm decisions.
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
from bubble_chamber.s80_multiscale_instances import augment_pixel_only
from bubble_chamber.s82_local_instance_support import (
    ARMS,QUANTILES,local_dark_stroke_support,thresholds_from_source_CAL,
    filter_extra_candidates,choose_CAL_only)
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75
from scripts.research.bubble_chamber import s80_calibrate_then_test_multiscale as s80

EXPERIMENT_ID='ANZA-LIRA-BC-S82-20261026'
SCHEMA='S82_IMAGE_LOCAL_PIXEL_INSTANCE_SUPPORT_SOURCE_CAL_FRESH_TEST_V1'
PROTOCOL=ROOT/'docs/bubble_chamber/S82_FROZEN_PER_LINE_PIXEL_SUPPORT_2026_10_26.md'
OPERATOR=ROOT/'bubble_chamber/s82_local_instance_support.py'
S80_CAL=ROOT/'docs/bubble_chamber/S80_PUBLIC_FROZEN_CAL_2026_10_26.json'
CAL_SEEDS={'005':20268201,'006':20268211}
TEST_SEEDS={'005':(20268203,20268204),'006':(20268213,20268214)}
CAL_SCENES=20
TEST_SCENES=24
CAL_CONDITIONS=('clean','blur_sigma_11')
TEST_CONDITIONS=('clean','blur_sigma_11','blur_sigma_16')
MEASURES=(*s80.MEASURES,'extra_considered','extra_accepted','extra_rejected')


def digest(p):return sha256(Path(p).read_bytes()).hexdigest()


def infer_from_pixels(gray,scorer,cutoff,graph_arm,thresholds):
    """The only dynamic data are image pixels and frozen numerical CAL policy."""
    if not isinstance(gray,np.ndarray) or gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('Only 192x192 uint8 pixels; cannot pass source geometry')
    if set(thresholds)!=set(QUANTILES):
        raise ValueError('Must provide both frozen source CAL threshold values')
    # This is the SAME blind S72 base and S80 consensus as previous studies.
    original=s80.infer_from_pixels(gray,scorer,cutoff,graph_arm)
    baseline=original['baseline']
    source=baseline['polylines']
    consensus=original['s80_consensus_add6']['polylines']
    out={'baseline':{**baseline,'extra_considered':0,'extra_accepted':0,'extra_rejected':0}}
    for arm in ARMS[1:]:
        filt=filter_extra_candidates(gray,source,consensus,arm,thresholds)
        if not filt['extra_accepted']:
            out[arm]={**baseline,'extra_considered':filt['extra_considered'],
                      'extra_accepted':0,'extra_rejected':filt['extra_rejected']}
            continue
        # When every S80 consensus extra survives, precisely reuse S80 solver.
        if filt['extra_accepted']==len(consensus)-len(source):
            x=original['s80_consensus_add6']
            out[arm]={**x,'extra_considered':filt['extra_considered'],
                      'extra_accepted':filt['extra_accepted'], 'extra_rejected':0}
            continue
        lines=filt['polylines']
        if len(lines)<3:
            chosen=[];qa={'solver_success':True};rows=[]
        else:
            rows=filter_rows(pair_rows(lines),'strict_s39')
            if rows:
                score=np.asarray(scorer(rows),float)
                chosen,qa=select_global_forest(scored_edges(rows,score),threshold=float(cutoff),margin=.04)
                if not qa['solver_success']:chosen=[]
            else:
                chosen=[];qa={'solver_success':True}
        out[arm]={'polylines':lines,'chosen':chosen,'pair_count':len(rows),
                  'solver_success':bool(qa['solver_success']),
                  'physics_auto_joins_authorized':False,
                  'added_pixel_instances':filt['extra_accepted'],
                  'duplicate_suppressed':original['s80_consensus_add6']['duplicate_suppressed'],
                  'scale_consensus_rejected':original['s80_consensus_add6']['scale_consensus_rejected'],
                  'original_polylines_retained':len(source),
                  'extra_considered':filt['extra_considered'],
                  'extra_accepted':filt['extra_accepted'],
                  'extra_rejected':filt['extra_rejected']}
    assert all(all(x is y for x,y in zip(source,out[arm]['polylines'])) for arm in ARMS)
    assert all(not out[arm]['physics_auto_joins_authorized'] for arm in ARMS)
    return out


def source_CAL_pixel_scores(photos,templates,seed,n,scorer,cutoff,graph_arm):
    """No evaluator/GT is imported. Same CAL scenes as later aggregate audit."""
    if not photos:raise ValueError('No SOURCE CAL photo backgrounds')
    rng=np.random.default_rng(seed)
    scores=[]
    for k in range(n):
        scene=make_three_piece_scene(photos[int(rng.integers(len(photos)))],templates,
            seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
        for cond in CAL_CONDITIONS:
            pix=s75.condition_pixels(scene.gray,cond)
            base=s68.infer_pixel_only(pix,scorer,cutoff,graph_arm)['polylines']
            result=augment_pixel_only(pix,base,'s80_consensus_add6')['polylines']
            scores.extend(local_dark_stroke_support(pix,p) for p in result[len(base):])
    return thresholds_from_source_CAL(scores),len(scores)


def audit_after_blind_inference(predictions,source_nodes):
    """Only this step may inspect artificially injected source geometry."""
    measured=s80.score_after_inference(predictions,source_nodes)
    out={}
    for arm,metrics in measured.items():
        d={**metrics,
           'extra_considered':int(predictions[arm]['extra_considered']),
           'extra_accepted':int(predictions[arm]['extra_accepted']),
           'extra_rejected':int(predictions[arm]['extra_rejected'])}
        if set(MEASURES)-set(d):raise AssertionError('Missing S82 synthetic-only metric')
        if arm!='baseline' and d['extra_accepted']!=d['added_pixel_instances']:
            raise AssertionError('Inconsistent added instance count')
        out[arm]=d
    return out


def sample(photos,templates,seed,n,conditions,scorer,cutoff,graph_arm,thresholds):
    rng=np.random.default_rng(seed)
    result={cond:{arm:[] for arm in ARMS} for cond in conditions}
    for i in range(n):
        scene=make_three_piece_scene(photos[int(rng.integers(len(photos)))],templates,
            seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[i%len(SCENARIOS)])
        for condition in conditions:
            pixels=s75.condition_pixels(scene.gray,condition)
            # Blind inference MUST finish before evaluating against source nodes.
            inference=infer_from_pixels(pixels,scorer,cutoff,graph_arm,thresholds)
            metric=audit_after_blind_inference(inference,scene.nodes)
            for arm in ARMS:result[condition][arm].append(metric[arm])
        if (i+1)%8==0:print('S82',seed,'scenes',i+1,'/',n,flush=True)
    return result


def aggregate(rows):
    if not rows or any(set(MEASURES)-set(r) for r in rows):
        raise ValueError('Incomplete posthoc S82 audit')
    return {**{name:int(sum(row[name] for row in rows)) for name in MEASURES},'scenes':len(rows)}


def paired(a,b,seed):
    if len(a)!=len(b) or not a:raise ValueError('Scene-level pairing violated')
    rng=np.random.default_rng(seed)
    picks=rng.integers(0,len(a),size=(1200,len(a)))
    names=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
           'complete_three_piece_paths_synthetic','shape_true_fragments_matched',
           'shape_true_one_or_both_endpoints_far','scene_any_noncorrect')
    result={}
    for key in names:
        delta=np.asarray([new[key]-old[key] for old,new in zip(a,b)],float)
        result[key]={'delta_total':int(delta.sum()),'mean_delta_per_scene':float(delta.mean()),
                     'paired_scene_bootstrap_95pct':[
                     float(x) for x in np.quantile(delta[picks].mean(1),[.025,.975])]}
    return result


def run(visualmetro:Path,tiles:Path,phase:str,calibration:Path|None=None,*,smoke=False):
    if phase not in ('cal','test'):raise ValueError('S82 phase must be cal or test')
    if digest(s80.S72_CAL)!=s80.LOCKED_S72_SHA256:
        raise ValueError('Frozen S72 CAL has changed')
    old80=json.loads(S80_CAL.read_text(encoding='utf8'))
    if old80['operator_sha256']!=digest(s80.OPERATOR):
        raise ValueError('S80 frozen source code changed')
    original_s72=s75.frozen_s72(s80.S72_CAL)
    templates,split_sha=s72._source_data(visualmetro)
    if split_sha!=s80.SPLIT_SHA:raise ValueError('Frozen whole-frame split mismatch')
    photos={p:sorted(tiles.glob(f'H50Z_030_01_{p}_*.png')) for p in ('005','006')}
    if (len(photos['005']),len(photos['006']))!=(21,18):
        raise ValueError('Expected original private H50Z 21+18 source crop bank')
    report={'schema':SCHEMA,'phase':phase,'experiment_id':EXPERIMENT_ID,
            'smoke_NOT_CONFIRMATORY':bool(smoke),'protocol_sha256':digest(PROTOCOL),
            'operator_sha256':digest(OPERATOR),'S80_source_operator_sha256':digest(s80.OPERATOR),
            'frozen_S72_CAL_sha256':s80.LOCKED_S72_SHA256,'S72_source_code_locks':s72.LOCKS,
            'whole_frame_split_sha256':split_sha,'one_H50Z_physical_run':True,
            'no_native_H50Z_GT':True,'native_GT_count':0,'all_results_synthetic_injections_only':True,
            'physics_auto_join_authorized':False,'arms':ARMS,'directions':{}}
    prev=None
    if phase=='test':
        if calibration is None or not calibration.is_file():
            raise FileNotFoundError('Independent held-out TEST requires frozen saved CAL')
        prev=json.loads(calibration.read_text())
        for key in ('schema','protocol_sha256','operator_sha256','S80_source_operator_sha256',
                    'frozen_S72_CAL_sha256','S72_source_code_locks','whole_frame_split_sha256'):
            if prev.get(key)!=report[key]:raise ValueError('S82 CAL provenance mismatch: '+key)
        if prev.get('phase')!='cal' or prev.get('smoke_NOT_CONFIRMATORY'):
            raise ValueError('Cannot use smoke CAL or TEST to tune S82')
        report['frozen_CAL_sha256']=digest(calibration)
    for di,source in enumerate(('005','006')):
        target='006' if source=='005' else '005'
        direction=f'{source}_to_{target}'
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        choice=original_s72['directions'][direction]
        graph_arm=choice['CAL_selected']
        if float(cutoff)!=float(choice['S56_frozen_CAL_cutoff']):
            raise ValueError('Frozen S56 geometry cutoffs shifted')
        row={'source_photo':source,'heldout_photo':target,
             'frozen_S72_graph_arm':graph_arm,'frozen_S56_cutoff':float(cutoff)}
        if phase=='cal':
            thresholds,num_scores=source_CAL_pixel_scores(photos[source],templates['cal'],
                      CAL_SEEDS[source],2 if smoke else CAL_SCENES,scorer,cutoff,graph_arm)
            obs=sample(photos[source],templates['cal'],CAL_SEEDS[source],
                       2 if smoke else CAL_SCENES,CAL_CONDITIONS,scorer,cutoff,graph_arm,thresholds)
            summary={arm:{cond:aggregate(obs[cond][arm]) for cond in CAL_CONDITIONS}
                     for arm in ARMS}
            row['CAL_only_local_thresholds']=thresholds
            row['CAL_only_pixel_extra_line_samples']=num_scores
            row['CAL_arms']=summary
            row['CAL_decision']=choose_CAL_only(summary)
            print('S82 CAL',direction,'thresholds',thresholds,'choose',row['CAL_decision'],flush=True)
        else:
            saved=prev['directions'][direction]
            if saved['frozen_S72_graph_arm']!=graph_arm or float(saved['frozen_S56_cutoff'])!=cutoff:
                raise ValueError('S82 frozen source scorer/arm drift')
            thresholds=saved['CAL_only_local_thresholds']
            if set(thresholds)!=set(QUANTILES):raise ValueError('S82 CAL numeric thresholds absent')
            selected=saved['CAL_decision']['CAL_selected']
            if selected not in ARMS:raise ValueError('Unknown frozen CAL selection')
            row['frozen_CAL_selection']=selected
            row['frozen_local_thresholds']=thresholds
            row['TEST_seeds']={}
            for seed_ix,seed in enumerate((TEST_SEEDS[source] if not smoke else TEST_SEEDS[source][:1])):
                obs=sample(photos[target],templates['test'],seed,
                           2 if smoke else TEST_SCENES,TEST_CONDITIONS,
                           scorer,cutoff,graph_arm,thresholds)
                row['TEST_seeds'][str(seed)]={cond:{
                    'arms':{arm:aggregate(obs[cond][arm]) for arm in ARMS},
                    'paired_vs_baseline':{arm:paired(obs[cond]['baseline'],obs[cond][arm],
                         20268290+di*40+seed_ix*10+ci) for arm in ARMS[1:]}}
                    for ci,cond in enumerate(TEST_CONDITIONS)}
                print('S82 TEST',direction,'seed',seed,'CAL selection',selected,flush=True)
        report['directions'][direction]=row
    if phase=='test':
        report['all_TEST_aggregates']={cond:{arm:{k:int(sum(
            seedrow[cond]['arms'][arm][k]
            for direct in report['directions'].values()
            for seedrow in direct['TEST_seeds'].values()))
            for k in (*MEASURES,'scenes')} for arm in ARMS} for cond in TEST_CONDITIONS}
        report['frozen_CAL_selected_TEST']={cond:{k:int(sum(
            seedrow[cond]['arms'][direct['frozen_CAL_selection']][k]
            for direct in report['directions'].values()
            for seedrow in direct['TEST_seeds'].values()))
            for k in (*MEASURES,'scenes')} for cond in TEST_CONDITIONS}
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
    x=ap.parse_args()
    if x.out.exists():raise FileExistsError('Immutable research output already exists')
    result=run(x.visualmetro,x.photo_tiles,x.phase,x.calibration,smoke=x.smoke)
    x.out.parent.mkdir(parents=True,exist_ok=True)
    x.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('S82 wrote',x.out,'SHA256',digest(x.out),flush=True)
if __name__=='__main__':main()
