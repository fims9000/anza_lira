#!/usr/bin/env python3
"""S77 frozen short-gap instance recovery, CAL before fresh TEST.

Only image inference is allowed to inspect genuine H50Z pixels and extracted
pixel-proposals. Synthetic source geometry enters the *separate* evaluators
AFTER all arm predictions have completed. No native particle IDs are inferred.
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
from bubble_chamber.s69_orientation_invariant_audit import audit_orientation_invariant_synthetic
from bubble_chamber.s76_shape_endpoint_diagnostics import audit_shape_vs_endpoint
from bubble_chamber.s77_short_gap_instances import ARMS,repair_pixel_only
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75

EXPERIMENT_ID='ANZA-LIRA-BC-S77-20261026'
SCHEMA='S77_FROZEN_SHORT_GAP_PIXEL_INSTANCES_SYNTHETIC_ONLY_V1'
PROTOCOL=ROOT/'docs/bubble_chamber/S77_FROZEN_SHORT_GAP_INSTANCE_RECOVERY_2026_10_26.md'
OPERATOR=ROOT/'bubble_chamber/s77_short_gap_instances.py'
SOURCE_CAL=ROOT/'docs/bubble_chamber/S72_PUBLIC_FROZEN_CAL_2026_10_26.json'
LOCKED_S72_CAL_SHA256='1ed7e58667aaab1a7d5ba91252db64b38bf8d3da517396b9e18e9e6bb484a952'
CAL_SEEDS={'005':20267701,'006':20267711}
TEST_SEEDS={'005':(20267703,20267704),'006':(20267713,20267714)}
N_CAL=16
N_TEST=24
CAL_CONDITIONS=('clean','blur_sigma_11')
TEST_CONDITIONS=('clean','blur_sigma_11','blur_sigma_16')
MEASURES=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
          'complete_three_piece_paths_synthetic','scene_any_noncorrect',
          'selected_provisional_links','solver_failure',
          'shape_true_fragments_matched','shape_true_both_endpoints_close',
          'shape_true_one_or_both_endpoints_far','shape_only_true_pair_availability',
          'shape_only_complete_chain_availability','shape_endpoint_separated_complete_chains',
          'candidate_polylines','attempted_short_gap_merged_pairs')


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def infer_arms_from_pixels(gray,scorer,cutoff,graph_arm):
    """No source GT, labels, or artificial identity enters inference."""
    if not isinstance(gray,np.ndarray) or gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('Only 192x192 uint8 pixels')
    if graph_arm not in s68.ARMS or not np.isfinite(cutoff):
        raise ValueError('Invalid frozen geometry scorer policy')
    orig=s68.infer_pixel_only(gray,scorer,cutoff,graph_arm)
    result={'baseline':{**orig,'attempted_short_gap_merged_pairs':0}}
    for arm in ARMS[1:]:
        proposal=repair_pixel_only(gray,orig['polylines'],arm)
        pieces=proposal['polylines']
        if proposal['merged_pairs']==0:
            result[arm]={**orig,'attempted_short_gap_merged_pairs':0}
            continue
        if len(pieces)<3:
            result[arm]={'polylines':pieces,'pair_count':0,'chosen':[],
                         'solver_success':True,'physics_auto_joins_authorized':False,
                         'attempted_short_gap_merged_pairs':proposal['merged_pairs']}
            continue
        rows=filter_rows(pair_rows(pieces),'strict_s39')
        if rows:
            scores=np.asarray(scorer(rows),float)
            chosen,qa=select_global_forest(scored_edges(rows,scores),threshold=float(cutoff),margin=.04)
            if not qa['solver_success']:chosen=[]
        else:
            chosen=[];qa={'solver_success':True}
        result[arm]={'polylines':pieces,'pair_count':len(rows),'chosen':chosen,
                     'solver_success':bool(qa['solver_success']),
                     'physics_auto_joins_authorized':False,
                     'attempted_short_gap_merged_pairs':proposal['merged_pairs']}
    return result


def score_after_predictions(pred,source_nodes):
    """Ground truth limited to THIS evaluator, never imported into inference."""
    official=audit_orientation_invariant_synthetic(pred,source_nodes)
    diagnostic=audit_shape_vs_endpoint(pred,source_nodes)
    if len(pred['chosen'])!=official['selected_provisional_links']:
        raise AssertionError('Official S69 proposal accounting violated')
    if sum(diagnostic[x] for x in (
            'matched_neighbor_correct_port','matched_neighbor_wrong_port',
            'matched_neighbor_endpoint_unlocalized','matched_wrong_identity_pair',
            'unmatched_or_ambiguous_endpoint_blind'))!=len(pred['chosen']):
        raise AssertionError('Diagnostic did not account for all machine proposals')
    return {
      'selected_synthetic_correct_links':int(official['selected_synthetic_correct_links']),
      'synthetic_noncorrect_selected':int(official['synthetic_noncorrect_selected']),
      'complete_three_piece_paths_synthetic':int(official['complete_three_piece_paths_synthetic']),
      'scene_any_noncorrect':int(official['scene_any_noncorrect']),
      'selected_provisional_links':len(pred['chosen']),
      'solver_failure':int(not pred['solver_success']),
      'shape_true_fragments_matched':int(diagnostic['shape_true_fragments_matched']),
      'shape_true_both_endpoints_close':int(diagnostic['shape_true_both_endpoints_close']),
      'shape_true_one_or_both_endpoints_far':int(diagnostic['shape_true_one_or_both_endpoints_far']),
      'shape_only_true_pair_availability':int(diagnostic['shape_only_true_pair_availability']),
      'shape_only_complete_chain_availability':int(diagnostic['shape_only_complete_chain_availability']),
      'shape_endpoint_separated_complete_chains':int(diagnostic['shape_endpoint_separated_complete_chains']),
      'candidate_polylines':len(pred['polylines']),
      'attempted_short_gap_merged_pairs':int(pred['attempted_short_gap_merged_pairs']),
    }


def sample_scenes(photos,templates,seed,n,conditions,scorer,cutoff,graph_arm):
    if not photos:raise ValueError('No private background bank')
    rng=np.random.default_rng(seed)
    observations={condition:{arm:[] for arm in ARMS} for condition in conditions}
    for k in range(n):
        scene=make_three_piece_scene(photos[int(rng.integers(len(photos)))],templates,
            seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
        for cond in conditions:
            pix=s75.condition_pixels(scene.gray,cond)
            # Every variant runs BEFORE consulting the generator's hidden nodes.
            blind=infer_arms_from_pixels(pix,scorer,cutoff,graph_arm)
            for arm,pred in blind.items():
                observations[cond][arm].append(score_after_predictions(pred,scene.nodes))
        if (k+1)%8==0:
            print('S77',seed,'progress',k+1,'of',n,flush=True)
    return observations


def aggregate(rows):
    if not rows:raise ValueError('Empty scene set')
    if any(set(MEASURES)-set(row) for row in rows):raise ValueError('Incomplete metric')
    return {**{key:int(sum(row[key] for row in rows)) for key in MEASURES},'scenes':len(rows)}


def select_from_CAL(stats):
    if set(stats)!=set(ARMS) or any(set(v)!=set(CAL_CONDITIONS) for v in stats.values()):
        raise ValueError('All frozen CAL conditions/arms required')
    base_clean=stats['baseline']['clean'];base_blur=stats['baseline']['blur_sigma_11']
    eligible=[]
    for arm in ARMS[1:]:
        clean=stats[arm]['clean'];blur=stats[arm]['blur_sigma_11']
        if (blur['selected_synthetic_correct_links']>=base_blur['selected_synthetic_correct_links']+2
            and blur['synthetic_noncorrect_selected']<=base_blur['synthetic_noncorrect_selected']
            and clean['synthetic_noncorrect_selected']<=base_clean['synthetic_noncorrect_selected']
            and clean['selected_synthetic_correct_links']>=base_clean['selected_synthetic_correct_links']-1
            and blur['complete_three_piece_paths_synthetic']>=base_blur['complete_three_piece_paths_synthetic']
            and clean['complete_three_piece_paths_synthetic']>=base_clean['complete_three_piece_paths_synthetic']
            and blur['solver_failure']==clean['solver_failure']==0):
            eligible.append(arm)
    chosen=min(eligible,key=lambda arm:(
        -stats[arm]['blur_sigma_11']['selected_synthetic_correct_links'],
        stats[arm]['blur_sigma_11']['synthetic_noncorrect_selected'],
        ARMS.index(arm))) if eligible else 'baseline'
    return {'CAL_selected':chosen,'eligible':eligible,
            'default_unchanged_if_no_qualifier':chosen=='baseline',
            'native_physical_join_authorized':False}


def paired(a,b,seed):
    if len(a)!=len(b) or not a:raise ValueError('Scene-paired observations required')
    keys=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
          'complete_three_piece_paths_synthetic','shape_true_fragments_matched',
          'shape_true_one_or_both_endpoints_far')
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(a),size=(1200,len(a)))
    return {key:{'mean_difference_per_scene':float((v:=np.array([x[key]-y[key]
                  for x,y in zip(b,a)],float)).mean()),
                 'paired_scene_bootstrap_95pct':[
                   float(q) for q in np.quantile(v[idx].mean(1),[.025,.975])]}
            for key in keys}


def run(visualmetro,tiles,phase,*,calibration=None,smoke=False):
    if phase not in ('cal','test'):raise ValueError('S77 phase CAL or TEST')
    if digest(SOURCE_CAL)!=LOCKED_S72_CAL_SHA256:
        raise ValueError('S72 direction CAL source changed')
    s75.frozen_s72(SOURCE_CAL) # verifies original S68/S69/S56 source hashes
    templates,split_sha=s72._source_data(visualmetro)
    if split_sha!='448f21ca3a709010ed972a596ed046cadff0643112905fd1d029c91f8b14faef':
        raise ValueError('Whole-frame source split drift')
    photos={p:sorted(tiles.glob(f'H50Z_030_01_{p}_*.png')) for p in ('005','006')}
    if len(photos['005'])!=21 or len(photos['006'])!=18:
        raise ValueError('Expected genuine private H50Z tiles (21+18)')
    result={'schema':SCHEMA,'experiment_id':EXPERIMENT_ID,'phase':phase,
            'smoke_NON_CONFIRMATORY':bool(smoke),
            'protocol_sha256':digest(PROTOCOL),'operator_sha256':digest(OPERATOR),
            'source_CAL_sha256':LOCKED_S72_CAL_SHA256,
            'source_frame_split_sha256':split_sha,
            'source_training_model_locks':s72.LOCKS,
            'native_GT_track_count':0,'physical_joins_authorized':False,
            'source_photographs_one_run':True,'visualmetro_070_not_H50Z_GT':True,
            'phase_scene_count_per_seed':2 if smoke else (N_CAL if phase=='cal' else N_TEST),
            'reserved_test_scenes_per_seed':N_TEST,
            'synthetic_only':True,'arms':ARMS,'directions':{}}
    frozen=None
    if phase=='test':
        if calibration is None or not calibration.is_file():
            raise FileNotFoundError('Must load saved S77 CAL before TEST')
        frozen=json.loads(calibration.read_text(encoding='utf8'))
        for field in ('schema','protocol_sha256','operator_sha256','source_CAL_sha256',
                      'source_frame_split_sha256','source_training_model_locks'):
            if result[field]!=frozen[field]:raise ValueError('S77 frozen CAL drift: '+field)
        if frozen.get('phase')!='cal' or frozen.get('smoke_NON_CONFIRMATORY'):
            raise ValueError('Reject smoke or non-CAL frozen decision')
        result['frozen_S77_CAL_SHA256']=digest(calibration)
    for direction_index,source in enumerate(('005','006')):
        target='006' if source=='005' else '005'
        label=f'{source}_to_{target}'
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        graph_arm=json.loads(SOURCE_CAL.read_text())['directions'][label]['CAL_selected']
        source_cutoff=json.loads(SOURCE_CAL.read_text())['directions'][label]['S56_frozen_CAL_cutoff']
        if float(cutoff)!=float(source_cutoff):raise ValueError('Original frozen S56 model cutoff changed')
        entry={'source_photo':source,'test_photo':target,'frozen_graph_arm':graph_arm,
               'frozen_S56_CAL_cutoff':float(cutoff)}
        if phase=='cal':
            samples=sample_scenes(photos[source],templates['cal'],CAL_SEEDS[source],
                                  2 if smoke else N_CAL,CAL_CONDITIONS,scorer,cutoff,graph_arm)
            stats={arm:{cond:aggregate(samples[cond][arm]) for cond in CAL_CONDITIONS} for arm in ARMS}
            entry['CAL_all_arms']=stats
            entry['CAL_decision']=select_from_CAL(stats)
            print('S77 CAL',label,entry['CAL_decision'],flush=True)
        else:
            prior=frozen['directions'][label]
            if prior['frozen_graph_arm']!=graph_arm or prior['frozen_S56_CAL_cutoff']!=float(cutoff):
                raise ValueError('S77 saved CAL graph arm or score cutoff drift')
            choice=prior['CAL_decision']['CAL_selected']
            if choice not in ARMS:raise ValueError('Unknown frozen arm')
            entry['CAL_selected']=choice
            entry['TEST_seeds']={}
            for si,seed in enumerate((TEST_SEEDS[source] if not smoke else TEST_SEEDS[source][:1])):
                samples=sample_scenes(photos[target],templates['test'],seed,
                                      2 if smoke else N_TEST,TEST_CONDITIONS,scorer,cutoff,graph_arm)
                entry['TEST_seeds'][str(seed)]={cond:{
                    'arms':{arm:aggregate(samples[cond][arm]) for arm in ARMS},
                    'paired_vs_baseline':{arm:paired(samples[cond]['baseline'],samples[cond][arm],
                         20267790+direction_index*20+si*4+ci) for arm in ARMS[1:]}}
                    for ci,cond in enumerate(TEST_CONDITIONS)}
                print('S77 TEST',label,seed,choice,flush=True)
        result['directions'][label]=entry
    if phase=='test':
        result['all_TEST_aggregates']={cond:{arm:{metric:int(sum(
             row[cond]['arms'][arm][metric] for direction in result['directions'].values()
             for row in direction['TEST_seeds'].values())) for metric in (*MEASURES,'scenes')}
             for arm in ARMS} for cond in TEST_CONDITIONS}
        result['CAL_selected_TEST_aggregate']={cond:{metric:int(sum(
             row[cond]['arms'][direction['CAL_selected']][metric]
             for direction in result['directions'].values()
             for row in direction['TEST_seeds'].values()))
             for metric in (*MEASURES,'scenes')} for cond in TEST_CONDITIONS}
        result['physical_auto_join_authorized']=False
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=('cal','test'))
    ap.add_argument('--visualmetro',type=Path,required=True)
    ap.add_argument('--photo-tiles',type=Path,required=True)
    ap.add_argument('--calibration',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Immutable S77 output already exists')
    j=run(a.visualmetro,a.photo_tiles,a.phase,calibration=a.calibration,smoke=a.smoke)
    a.out.parent.mkdir(exist_ok=True,parents=True)
    a.out.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('S77 persisted',a.phase,a.out,'ID',EXPERIMENT_ID,flush=True)
if __name__=='__main__':main()
