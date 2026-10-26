#!/usr/bin/env python3
"""S80 frozen pixel-derived multiscale instances: CAL-before-fresh-TEST.

Inference never sees generator geometry or labels; synthetic evaluation and
shape/endpoint measures run only AFTER all methods have already predicted.
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
from bubble_chamber.s80_multiscale_instances import ARMS,augment_pixel_only,choose_cal_only
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75
from scripts.research.bubble_chamber import s77_calibrate_then_test_short_gaps as s77

EXPERIMENT_ID='ANZA-LIRA-BC-S80-20261026'
SCHEMA='S80_PIXEL_ONLY_MULTISCALE_INSTANCES_CAL_THEN_FRESH_TEST_V1'
PROTOCOL=ROOT/'docs/bubble_chamber/S80_FROZEN_MULTISCALE_PIXEL_INSTANCES_2026_10_26.md'
OPERATOR=ROOT/'bubble_chamber/s80_multiscale_instances.py'
S72_CAL=ROOT/'docs/bubble_chamber/S72_PUBLIC_FROZEN_CAL_2026_10_26.json'
LOCKED_S72_SHA256='1ed7e58667aaab1a7d5ba91252db64b38bf8d3da517396b9e18e9e6bb484a952'
SPLIT_SHA='448f21ca3a709010ed972a596ed046cadff0643112905fd1d029c91f8b14faef'
CAL_SEEDS={'005':20268001,'006':20268011}
TEST_SEEDS={'005':(20268003,20268004),'006':(20268013,20268014)}
CAL_SCENES=16
TEST_SCENES=24
CAL_CONDITIONS=('clean','blur_sigma_11')
TEST_CONDITIONS=('clean','blur_sigma_11','blur_sigma_16')
BOOTSTRAP=1200
MEASURES=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
          'complete_three_piece_paths_synthetic','scene_any_noncorrect',
          'selected_provisional_links','solver_failure','shape_true_fragments_matched',
          'shape_true_both_endpoints_close','shape_true_one_or_both_endpoints_far',
          'shape_only_true_pair_availability','shape_only_complete_chain_availability',
          'candidate_polylines','candidate_pairs_after_strict_gate',
          'added_pixel_instances','duplicate_suppressed','scale_consensus_rejected',
          'original_polylines_retained')


def digest(p):return sha256(Path(p).read_bytes()).hexdigest()


def infer_from_pixels(gray,scorer,cutoff,graph_arm):
    """No simulation nodes, GT identity or track endpoints as arguments."""
    if gray.shape!=(192,192) or gray.dtype!=np.uint8:raise ValueError('Need 192x192 uint8 pixels')
    if graph_arm not in s68.ARMS:raise ValueError('S72 frozen graph arm required')
    if not np.isfinite(cutoff) or cutoff<0 or cutoff>1:raise ValueError('Invalid cutoff')
    original=s68.infer_pixel_only(gray,scorer,cutoff,graph_arm)
    source=original['polylines']
    all_arms={'baseline':{**original,'physics_auto_joins_authorized':False,'added_pixel_instances':0,'duplicate_suppressed':0,
                 'scale_consensus_rejected':0,'original_polylines_retained':len(source)}}
    for arm in ARMS[1:]:
        new=augment_pixel_only(gray,source,arm)
        lines=new['polylines']
        if len(lines)<3:
            chosen=[];qa={'solver_success':True};rows=[]
        else:
            rows=filter_rows(pair_rows(lines),'strict_s39')
            if rows:
                scores=np.asarray(scorer(rows),float)
                chosen,qa=select_global_forest(scored_edges(rows,scores),
                                      threshold=float(cutoff),margin=.04)
                if not qa['solver_success']:chosen=[]
            else:chosen=[];qa={'solver_success':True}
        all_arms[arm]={'polylines':lines,'chosen':chosen,'pair_count':len(rows),
                 'solver_success':bool(qa['solver_success']),
                 'physics_auto_joins_authorized':False,
                 'added_pixel_instances':new['extra_count'],
                 'duplicate_suppressed':new['duplicate_suppressed'],
                 'scale_consensus_rejected':new['scale_consensus_rejected'],
                 'original_polylines_retained':new['original_count']}
    # All new methods retain exact original line objects; source list immutable.
    for arm in ARMS[1:]:
        if not all(x is y for x,y in zip(source,all_arms[arm]['polylines'])):
            raise AssertionError('A new S80 arm modified an original source line')
    return all_arms


def score_after_inference(inferred,ref):
    """GT allowed ONLY after predictions are completed for all arms."""
    x={}
    for arm,pred in inferred.items():
        r=s77.score_after_predictions({**pred,'attempted_short_gap_merged_pairs':0},ref)
        x[arm]={**{key:int(r[key]) for key in MEASURES if key in r},
                'added_pixel_instances':int(pred['added_pixel_instances']),
                'duplicate_suppressed':int(pred['duplicate_suppressed']),
                'scale_consensus_rejected':int(pred['scale_consensus_rejected']),
                'original_polylines_retained':int(pred['original_polylines_retained']),
                'candidate_pairs_after_strict_gate':int(pred['pair_count'])}
        if x[arm]['candidate_polylines']!=len(pred['polylines']):
            raise AssertionError('Pixel source candidate count inconsistent')
        if len(set(MEASURES)-set(x[arm])):raise AssertionError('Missing synthetic audit metric')
    return x


def sample(photos,templates,seed,n,conditions,scorer,cutoff,graph_arm):
    if not photos:raise ValueError('Missing real photo bank')
    rng=np.random.default_rng(seed)
    obs={cond:{arm:[] for arm in ARMS} for cond in conditions}
    for i in range(n):
        scene=make_three_piece_scene(photos[int(rng.integers(len(photos)))],templates,
            seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[i%len(SCENARIOS)])
        for condition in conditions:
            pix=s75.condition_pixels(scene.gray,condition)
            blind=infer_from_pixels(pix,scorer,cutoff,graph_arm)
            # Source artificial truth is NEVER available before all image-only inferences.
            measured=score_after_inference(blind,scene.nodes)
            for arm in ARMS:obs[condition][arm].append(measured[arm])
        if (i+1)%8==0:print('S80',seed,'scenes',i+1,'/',n,flush=True)
    return obs


def aggregate(rows):
    if not rows or any(set(MEASURES)-set(row) for row in rows):
        raise ValueError('Incomplete scene audit')
    return {**{key:int(sum(row[key] for row in rows)) for key in MEASURES},
            'scenes':len(rows)}


def paired(a,b,seed):
    if not a or len(a)!=len(b):raise ValueError('Unpaired scene results')
    rng=np.random.default_rng(seed)
    sampled=rng.integers(0,len(a),size=(BOOTSTRAP,len(a)))
    keys=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
          'complete_three_piece_paths_synthetic','shape_true_fragments_matched',
          'shape_true_one_or_both_endpoints_far','scene_any_noncorrect')
    return {key:{'delta_total':int((z:=np.array([y[key]-x[key] for x,y in zip(a,b)],int)).sum()),
                 'mean_delta_per_scene':float(z.mean()),
                 'paired_scene_bootstrap_95pct':[float(q) for q in
                       np.quantile(z[sampled].mean(1),(.025,.975))]}
            for key in keys}


def run(visualmetro:Path,tiles:Path,phase:str,*,calibration:Path|None=None,smoke=False):
    if phase not in ('cal','test'):raise ValueError('S80 CAL/TEST only')
    if digest(S72_CAL)!=LOCKED_S72_SHA256:
        raise ValueError('S72 reference CAL not frozen')
    frozen_s72=s75.frozen_s72(S72_CAL)
    templates,split_sha=s72._source_data(visualmetro)
    if split_sha!=SPLIT_SHA:raise ValueError('Original film-070 whole-frame split changed')
    photos={label:sorted(tiles.glob(f'H50Z_030_01_{label}_*.png')) for label in ('005','006')}
    if len(photos['005'])!=21 or len(photos['006'])!=18:
        raise ValueError('The original 39 genuine-photo private tiles are missing')
    record={'schema':SCHEMA,'experiment_id':EXPERIMENT_ID,'phase':phase,
            'smoke_NOT_CONFIRMATORY':bool(smoke),'protocol_sha256':digest(PROTOCOL),
            'operator_sha256':digest(OPERATOR),'S72_CAL_sha256':LOCKED_S72_SHA256,
            'frozen_s72_source_code_locks':s72.LOCKS,'source_split_sha256':split_sha,
            'photo_banks_one_physical_run':True,'visualmetro_070_NOT_native_H50Z_GT':True,
            'synthetic_only':True,'native_verified_track_GT':0,
            'physical_auto_joins_authorized':False,'arms':ARMS,
            'scenes_per_seed':2 if smoke else (CAL_SCENES if phase=='cal' else TEST_SCENES),
            'directions':{}}
    prev=None
    if phase=='test':
        if calibration is None or not calibration.is_file():
            raise FileNotFoundError('Must have IMMUTABLE non-smoke CAL first')
        prev=json.loads(calibration.read_text(encoding='utf8'))
        for f in ('schema','protocol_sha256','operator_sha256','S72_CAL_sha256',
                  'frozen_s72_source_code_locks','source_split_sha256'):
            if record[f]!=prev.get(f):raise ValueError('S80 frozen CAL mismatch: '+f)
        if prev.get('phase')!='cal' or prev.get('smoke_NOT_CONFIRMATORY'):
            raise ValueError('Cannot run TEST from non-confirmatory CAL')
        record['frozen_S80_CAL_sha256']=digest(calibration)
    for ix,source in enumerate(('005','006')):
        target='006' if source=='005' else '005';label=f'{source}_to_{target}'
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        entry_s72=frozen_s72['directions'][label]
        graph_arm=entry_s72['CAL_selected']
        if float(cutoff)!=float(entry_s72['S56_frozen_CAL_cutoff']):
            raise ValueError('S56 source-trained CAL scorer cutoff changed')
        record_row={'source_photo':source,'heldout_photo':target,
                    'frozen_S72_extractor':graph_arm,'frozen_S56_CAL_cutoff':float(cutoff)}
        if phase=='cal':
            observations=sample(photos[source],templates['cal'],CAL_SEEDS[source],
                                2 if smoke else CAL_SCENES,CAL_CONDITIONS,scorer,cutoff,graph_arm)
            all_stats={arm:{cond:aggregate(observations[cond][arm])
                         for cond in CAL_CONDITIONS} for arm in ARMS}
            record_row['CAL_arms']=all_stats
            record_row['CAL_decision']=choose_cal_only(all_stats)
            print('S80 CAL',label,record_row['CAL_decision'],flush=True)
        else:
            frozen=prev['directions'][label]
            if frozen['frozen_S72_extractor']!=graph_arm or frozen['frozen_S56_CAL_cutoff']!=float(cutoff):
                raise ValueError('S80 source model/arm differed from frozen CAL')
            selection=frozen['CAL_decision']['CAL_selected']
            if selection not in ARMS:raise ValueError('Unrecognized frozen CAL arm')
            record_row['CAL_selected']=selection
            record_row['TEST_seeds']={}
            seeds=TEST_SEEDS[source] if not smoke else TEST_SEEDS[source][:1]
            for si,seed in enumerate(seeds):
                obs=sample(photos[target],templates['test'],seed,
                           2 if smoke else TEST_SCENES,TEST_CONDITIONS,scorer,cutoff,graph_arm)
                record_row['TEST_seeds'][str(seed)]={cond:{
                    'arms':{arm:aggregate(obs[cond][arm]) for arm in ARMS},
                    'paired_to_baseline':{arm:paired(obs[cond]['baseline'],obs[cond][arm],
                                      20268090+ix*200+si*10+ci) for arm in ARMS[1:]}}
                    for ci,cond in enumerate(TEST_CONDITIONS)}
                print('S80 TEST',label,'seed',seed,'CAL chose',selection,flush=True)
        record['directions'][label]=record_row
    if phase=='test':
        record['all_TEST_aggregates']={cond:{arm:{key:int(sum(
            seed_record[cond]['arms'][arm][key]
            for direction in record['directions'].values()
            for seed_record in direction['TEST_seeds'].values()))
            for key in (*MEASURES,'scenes')}
            for arm in ARMS} for cond in TEST_CONDITIONS}
        record['CAL_selected_TEST_aggregate']={cond:{key:int(sum(
            seed_record[cond]['arms'][direction['CAL_selected']][key]
            for direction in record['directions'].values()
            for seed_record in direction['TEST_seeds'].values()))
            for key in (*MEASURES,'scenes')} for cond in TEST_CONDITIONS}
        record['physical_auto_joins_authorized']=False
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('cal','test'))
    parser.add_argument('--visualmetro',type=Path,required=True)
    parser.add_argument('--photo-tiles',type=Path,required=True)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args()
    if a.out.exists():raise FileExistsError('Never overwrite frozen CAL/TEST runs')
    obj=run(a.visualmetro,a.photo_tiles,a.phase,calibration=a.calibration,smoke=a.smoke)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('S80 saved',a.out,'sha256',digest(a.out),flush=True)
if __name__=='__main__':main()
