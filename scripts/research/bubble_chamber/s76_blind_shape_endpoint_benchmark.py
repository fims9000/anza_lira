#!/usr/bin/env python3
"""S76 frozen diagnostic: shape/endpoint audit of blind pixel predictions.

Fresh synthetic scene GT is consulted only AFTER all pixel-derived predictions.
This script NEVER changes or trains inference models and never auto-joins real
H50Z particles. Per-scene paths/points/photos are strictly PRIVATE in memory.
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
from bubble_chamber.s56_three_piece_scenes import make_three_piece_scene,SCENARIOS
from bubble_chamber.s69_orientation_invariant_audit import audit_orientation_invariant_synthetic
from bubble_chamber.s76_shape_endpoint_diagnostics import audit_shape_vs_endpoint,SCHEMA as AUDIT_SCHEMA
from bubble_chamber.s75_local_ridge_rescue import ARMS as PIXEL_ARMS,preprocess
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75

EXPERIMENT_ID='ANZA-LIRA-BC-S76-20261026'
SCHEMA='S76_FRESH_SCENE_ENDPOINT_BLIND_SHAPE_DIAGNOSTIC_V1'
PROTOCOL=ROOT/'docs/bubble_chamber/S76_FROZEN_ENDPOINT_BLIND_SHAPE_AUDIT_2026_10_26.md'
AUDITOR=ROOT/'bubble_chamber/s76_shape_endpoint_diagnostics.py'
SOURCE_CAL=ROOT/'docs/bubble_chamber/S72_PUBLIC_FROZEN_CAL_2026_10_26.json'
LOCKED_S72_CAL_SHA256='1ed7e58667aaab1a7d5ba91252db64b38bf8d3da517396b9e18e9e6bb484a952'
SEEDS={'005':(20267603,20267604),'006':(20267613,20267614)}
N_TEST=24
IMAGE_CONDITIONS=('clean','blur_sigma_11','blur_sigma_16')
ARMS=tuple(PIXEL_ARMS)
FIELDS=('predicted_curve_count','shape_associations','shape_true_fragments_matched',
        'shape_decoy_matched','shape_unmatched_polylines','shape_ambiguous_polylines',
        'shape_matched_both_endpoints_close','shape_matched_one_or_both_endpoints_far',
        'shape_true_both_endpoints_close','shape_true_one_or_both_endpoints_far',
        'shape_decoy_one_or_both_endpoints_far',
        'shape_only_true_pair_availability','shape_only_complete_chain_availability',
        'matched_neighbor_correct_port','matched_neighbor_wrong_port',
        'matched_neighbor_endpoint_unlocalized','matched_wrong_identity_pair',
        'unmatched_or_ambiguous_endpoint_blind','shape_endpoint_separated_complete_chains',
        'official_s69_correct_links','official_s69_noncorrect_links',
        'official_s69_wrong_port','official_s69_unmatched','official_s69_complete_paths',
        'selected_proposals')


def sha(path):return sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    if not rows:raise ValueError('Expected synthetic scenes')
    if any(set(FIELDS)-set(x) for x in rows):raise ValueError('Incomplete diagnostic result')
    return {**{k:int(sum(x[k] for x in rows)) for k in FIELDS},'scene_count':len(rows)}


def paired(a,b,seed):
    """Bootstrap differences per complete scene; no pixel-level pseudoreplication."""
    if not a or len(a)!=len(b):raise ValueError('Pairs must be on the same scenes')
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(a),size=(1200,len(a)))
    fields=('shape_true_fragments_matched','shape_matched_one_or_both_endpoints_far',
            'matched_neighbor_endpoint_unlocalized','matched_neighbor_wrong_port',
            'official_s69_correct_links','official_s69_noncorrect_links')
    return {key:{'paired_difference_per_scene':float((v:=np.asarray(
                [b[i][key]-a[i][key] for i in range(len(a))],float)).mean()),
                 'scene_bootstrap_95pct':[
                     float(x) for x in np.quantile(v[draws].mean(axis=1),(.025,.975))]}
            for key in fields}


def blind_pixel_predictions(pixels,scorer,cutoff,method):
    """Blind inference accepts ONLY pixels, pretrained original scorer and method."""
    if pixels.shape!=(192,192) or pixels.dtype!=np.uint8:raise ValueError('Wrong raster')
    return {arm:s68.infer_pixel_only(preprocess(pixels,arm),scorer,cutoff,method)
            for arm in ARMS}


def measure_prediction(pred,artificial_nodes):
    """Strict original and diagnostic evaluated after inference, with same data."""
    official=audit_orientation_invariant_synthetic(pred,artificial_nodes)
    diagnostic=audit_shape_vs_endpoint(pred,artificial_nodes)
    if official['selected_provisional_links']!=len(pred['chosen']):
        raise AssertionError('Original official evaluator changed proposal count')
    if sum(diagnostic[k] for k in ('matched_neighbor_correct_port','matched_neighbor_wrong_port',
            'matched_neighbor_endpoint_unlocalized','matched_wrong_identity_pair',
            'unmatched_or_ambiguous_endpoint_blind'))!=len(pred['chosen']):
        raise AssertionError('S76 does not account for every original proposal')
    return {**{k:int(diagnostic[k]) for k in FIELDS if k in diagnostic},
            'official_s69_correct_links':int(official['selected_synthetic_correct_links']),
            'official_s69_noncorrect_links':int(official['synthetic_noncorrect_selected']),
            'official_s69_wrong_port':int(official['selected_synthetic_wrong_endpoint']),
            'official_s69_unmatched':int(official['selected_synthetic_unmatched_or_ambiguous']),
            'official_s69_complete_paths':int(official['complete_three_piece_paths_synthetic']),
            'selected_proposals':len(pred['chosen'])}


def run(visualmetro,tiles,*,smoke=False):
    if sha(SOURCE_CAL)!=LOCKED_S72_CAL_SHA256:
        raise ValueError('Source S72 CAL content changed')
    # This also rejects changes in original S68/S69/S56 source logic.
    s75.frozen_s72(SOURCE_CAL)
    templates,split_sha=s72._source_data(visualmetro)
    if split_sha!='448f21ca3a709010ed972a596ed046cadff0643112905fd1d029c91f8b14faef':
        raise ValueError('Original visualMetro frame split drift')
    photographs={p:sorted(tiles.glob(f'H50Z_030_01_{p}_*.png')) for p in ('005','006')}
    if len(photographs['005'])!=21 or len(photographs['006'])!=18:
        raise ValueError('Wrong number of private real H50Z background crops')
    cal=json.loads(SOURCE_CAL.read_text())
    result={'schema':SCHEMA,'experiment_id':EXPERIMENT_ID,'phase':'TEST',
            'smoke_NON_CONFIRMATORY':bool(smoke),'protocol_sha256':sha(PROTOCOL),
            'auditor_sha256':sha(AUDITOR),'s69_original_sha256':sha(ROOT/'bubble_chamber/s69_orientation_invariant_audit.py'),
            'S72_calibration_sha256':sha(SOURCE_CAL),'source_frame_split_sha256':split_sha,
            'source_070_is_not_native_H50Z_truth':True,'backgrounds_from_one_physics_run':True,
            'native_H50Z_GT_count':0,'physical_auto_joins_authorized':False,
            'model_training_or_threshold_selection_performed':False,
            'S69_official_scoring_unchanged':True,'diagnostic_schema':AUDIT_SCHEMA,
            'scene_count_per_seed':2 if smoke else N_TEST,'seeds':SEEDS,
            'conditions':IMAGE_CONDITIONS,'arms':ARMS,'directions':{}}
    for di,source in enumerate(('005','006')):
        target='006' if source=='005' else '005'
        label=f'{source}_to_{target}'
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        method=cal['directions'][label]['CAL_selected']
        if float(cutoff)!=float(cal['directions'][label]['S56_frozen_CAL_cutoff']):
            raise ValueError('Frozen graph source score drift')
        direction={'source_photo':source,'target_photo':target,'frozen_graph_arm':method,
                   'frozen_S56_cutoff':float(cutoff),'seed_aggregates':{}}
        for si,seed in enumerate((SEEDS[source] if not smoke else SEEDS[source][:1])):
            rng=np.random.default_rng(seed)
            photos=photographs[target]
            data={cond:{arm:[] for arm in ARMS} for cond in IMAGE_CONDITIONS}
            for k in range(2 if smoke else N_TEST):
                filename=photos[int(rng.integers(len(photos)))]
                scene=make_three_piece_scene(filename,templates['test'],
                         seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
                for cond in IMAGE_CONDITIONS:
                    pix=s75.condition_pixels(scene.gray,cond)
                    # IMPORTANT: all arm predictions from pixels finish BEFORE synthetic GT evaluation.
                    predictions=blind_pixel_predictions(pix,scorer,cutoff,method)
                    for arm,pred in predictions.items():
                        data[cond][arm].append(measure_prediction(pred,scene.nodes))
                if (k+1)%8==0:print('S76',label,seed,'scenes',k+1,flush=True)
            direction['seed_aggregates'][str(seed)]={'conditions':{cond:{
                'arms':{arm:summarize(rows) for arm,rows in data[cond].items()},
                'blur_minus_clean_paired':({arm:paired(data['clean'][arm],data[cond][arm],
                           20267699+di*20+si*4+ci) for arm in ARMS} if cond!='clean' else {})}
                for ci,cond in enumerate(IMAGE_CONDITIONS)}}
        result['directions'][label]=direction
    overall={}
    for cond in IMAGE_CONDITIONS:
        overall[cond]={}
        for arm in ARMS:
            overall[cond][arm]={k:int(sum(
                entry['conditions'][cond]['arms'][arm][k]
                for d in result['directions'].values() for entry in d['seed_aggregates'].values()))
                for k in (*FIELDS,'scene_count')}
    result['aggregate_all_scenes']=overall
    result['status']='diagnostic_only_not_independent_physics_validation'
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Frozen results must not be overwritten')
    j=run(args.visualmetro,args.photo_tiles,smoke=args.smoke)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('S76 wrote',args.out,flush=True)
if __name__=='__main__':main()
