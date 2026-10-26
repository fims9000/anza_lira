#!/usr/bin/env python3
"""S75 pixel-only local ridge recovery: CAL-only choice and fresh blind TEST.

Source photo score cutoff and model frozen by S72. Every synthetic reference
is passed to *evaluation only*, after every pixel-domain prediction finishes.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2
import numpy as np

from bubble_chamber.s73_image_robustness import perturb
from bubble_chamber.s75_local_ridge_rescue import ARMS,preprocess,choose_cal_only
from bubble_chamber.s56_three_piece_scenes import make_three_piece_scene,SCENARIOS
from bubble_chamber.s69_orientation_invariant_audit import audit_orientation_invariant_synthetic
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
from scripts.research.bubble_chamber import s73_stress_test_pixel_graph as s73

EXPERIMENT_ID='ANZA-LIRA-BC-S75-20261026'
SCHEMA='S75_CAL_ONLY_LOCAL_RIDGE_THEN_FRESH_TEST_SYNTHETIC_ONLY_V1'
PROTOCOL=ROOT/'docs/bubble_chamber/S75_FROZEN_LOCAL_RIDGE_RESCUE_2026_10_26.md'
OPERATOR=ROOT/'bubble_chamber/s75_local_ridge_rescue.py'
CAL_SEEDS={'005':20267501,'006':20267511}
TEST_SEEDS={'005':(20267503,20267504),'006':(20267513,20267514)}
N_CAL=24
N_TEST=24
CAL_CONDITIONS=('clean','blur_sigma_11')
TEST_CONDITIONS=('clean','blur_sigma_11','blur_sigma_16')
METRICS=tuple(s68.METRICS)


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def frozen_s72(cal_file:Path):
    # Verify original S68/S69/S56 code *before* running generated scenes.
    for rel,expected in s72.LOCKS.items():
        if digest(ROOT/rel)!=expected:
            raise ValueError('Frozen S72 code drift: '+rel)
    return s73.load_locked_cal(cal_file)


def condition_pixels(gray,kind):
    if kind=='blur_sigma_16':
        if gray.dtype!=np.uint8 or gray.shape!=(192,192):raise ValueError('Invalid image')
        return cv2.GaussianBlur(gray,(0,0),sigmaX=1.6,sigmaY=1.6)
    return perturb(gray,kind,seed=0)  # clean/blur deterministic; no GT or RNG used


def sample(photo_paths,templates,seed,n,scorer,cutoff,graph_arm,conditions):
    if graph_arm not in s68.ARMS:raise ValueError('Unknown frozen S72 graph method')
    if not photo_paths:raise ValueError('No private photo tile paths')
    rng=np.random.default_rng(seed)
    result={c:{a:[] for a in ARMS} for c in conditions}
    for k in range(n):
        image=photo_paths[int(rng.integers(len(photo_paths)))]
        scene=make_three_piece_scene(image,templates,
                seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
        for condition in conditions:
            pixels=condition_pixels(scene.gray,condition)
            # All arms infer blind from the same image BEFORE looking at source GT.
            pred={a:s68.infer_pixel_only(preprocess(pixels,a),scorer,cutoff,graph_arm) for a in ARMS}
            for arm,estimate in pred.items():
                a=audit_orientation_invariant_synthetic(estimate,scene.nodes)
                result[condition][arm].append({metric:a[metric] for metric in METRICS})
        if k%8==7:print('S75 source seed',seed,'scenes',k+1,'/',n,flush=True)
    return result


def summarize(rows):
    return s68.sums(rows)


def run(data:Path,tiles:Path,s72_cal:Path,phase:str,*,frozen_s75_cal:Path|None=None,smoke=False):
    if phase not in ('cal','test'):raise ValueError('Frozen S75 phase CAL or TEST only')
    source_cal=frozen_s72(s72_cal)
    templates,split_sha=s72._source_data(data)
    if source_cal['source_frame_split_sha256']!=split_sha:raise ValueError('Changed source frame split')
    photos={p:sorted(tiles.glob(f'H50Z_030_01_{p}_*.png')) for p in ('005','006')}
    if [len(photos[p]) for p in ('005','006')]!=[21,18]:
        raise ValueError('Missing private original H50Z crop bank')
    result={'schema':SCHEMA,'experiment_id':EXPERIMENT_ID,'phase':phase,
        'smoke_NOT_CONFIRMATORY':bool(smoke),
        'protocol_sha256':digest(PROTOCOL), 'operator_sha256':digest(OPERATOR),
        'frozen_S72_calibration_sha256':s73.FROZEN_S72_CAL_SHA256,
        'original_S72_code_sha256':s72.LOCKS,'S37_frame_split_sha256':split_sha,
        'H50Z_photographic_backgrounds_one_physical_run':True,
        'visualmetro_film070_is_not_H50Z_native_ground_truth':True,
        'physical_H50Z_expert_GT_count':0,'physical_auto_join_authorized':False,
        'all_metrics_synthetic_injections_only':True,'directions':{}}
    saved=None
    if phase=='test':
        if not frozen_s75_cal or not frozen_s75_cal.is_file():
            raise FileNotFoundError('Immutable complete S75 CAL required before TEST')
        saved=json.loads(frozen_s75_cal.read_text(encoding='utf8'))
        for key in ('schema','protocol_sha256','operator_sha256','frozen_S72_calibration_sha256',
                    'original_S72_code_sha256','S37_frame_split_sha256'):
            if saved.get(key)!=result[key]:raise ValueError('S75 CAL provenance drift: '+key)
        if saved.get('phase')!='cal' or saved.get('smoke_NOT_CONFIRMATORY'):
            raise ValueError('Cannot use smoke or non-CAL data for confirmatory TEST')
        result['frozen_CAL_SHA256']=digest(frozen_s75_cal)
    for source in ('005','006'):
        target='006' if source=='005' else '005'
        direction=f'{source}_to_{target}'
        chosen_graph=source_cal['directions'][direction]['CAL_selected']
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        if float(cutoff)!=float(source_cal['directions'][direction]['S56_frozen_CAL_cutoff']):
            raise ValueError('Original S56 frozen score cutoff changed')
        row={'source_photo':source,'test_photo':target,
             'frozen_S72_graph_method':chosen_graph,'frozen_S56_cutoff':float(cutoff)}
        if phase=='cal':
            sampled=sample(photos[source],templates['cal'],CAL_SEEDS[source],
                           2 if smoke else N_CAL,scorer,cutoff,chosen_graph,CAL_CONDITIONS)
            stats={arm:{cond:summarize(sampled[cond][arm]) for cond in CAL_CONDITIONS} for arm in ARMS}
            row['CAL_all_arms']=stats
            row['CAL_only_choice']=choose_cal_only(stats)
            print('S75 CAL',direction,row['CAL_only_choice'],flush=True)
        else:
            prior=saved['directions'][direction]
            if (prior['frozen_S72_graph_method']!=chosen_graph or
                prior['frozen_S56_cutoff']!=float(cutoff)):
                raise ValueError('Frozen S75 CAL scorer/method changed')
            choice=prior['CAL_only_choice']['chosen_CAL_only']
            if choice not in ARMS:raise ValueError('Saved S75 choice outside frozen grid')
            row['frozen_CAL_preprocess']=choice
            row['TEST_seeds']={}
            for seed in (TEST_SEEDS[source] if not smoke else TEST_SEEDS[source][:1]):
                sampled=sample(photos[target],templates['test'],seed,
                               2 if smoke else N_TEST,scorer,cutoff,chosen_graph,TEST_CONDITIONS)
                row['TEST_seeds'][str(seed)]={'conditions':{cond:{
                    'arms':{arm:summarize(sampled[cond][arm]) for arm in ARMS},
                    'CAL_selected_minus_identity_paired_scene_bootstrap':s68.paired(
                        sampled[cond]['identity'],sampled[cond][choice],seed+175+ci)}
                    for ci,cond in enumerate(TEST_CONDITIONS)}}
                print('S75 TEST',direction,seed,'choice',choice,flush=True)
        result['directions'][direction]=row
    if phase=='test':
        result['overall_conditions']={cond:{arm:{k:int(sum(
            z['conditions'][cond]['arms'][arm][k]
            for d in result['directions'].values() for z in d['TEST_seeds'].values()))
            for k in (*METRICS,'scene_count')} for arm in ARMS} for cond in TEST_CONDITIONS}
        result['frozen_CAL_choice_TEST']={cond:{k:int(sum(
            z['conditions'][cond]['arms'][d['frozen_CAL_preprocess']][k]
            for d in result['directions'].values() for z in d['TEST_seeds'].values()))
            for k in (*METRICS,'scene_count')} for cond in TEST_CONDITIONS}
        result['real_particle_auto_join_authorized']=False
        # The strict synthetic gate is reported, but never used to tune TEST.
        comparisons=[]
        for name,row in result['directions'].items():
            for cond in TEST_CONDITIONS:
                total=lambda arm:{k:sum(z['conditions'][cond]['arms'][arm][k]
                            for z in row['TEST_seeds'].values())
                            for k in ('selected_synthetic_correct_links','synthetic_noncorrect_selected',
                                      'complete_three_piece_paths_synthetic')}
                baseline=total('identity');chosen=total(row['frozen_CAL_preprocess'])
                comparisons.append({
                    'direction':name,'condition':cond,
                    'does_not_reduce_correct':chosen['selected_synthetic_correct_links']>=baseline['selected_synthetic_correct_links'],
                    'does_not_increase_noncorrect':chosen['synthetic_noncorrect_selected']<=baseline['synthetic_noncorrect_selected'],
                    'does_not_reduce_full_paths':chosen['complete_three_piece_paths_synthetic']>=baseline['complete_three_piece_paths_synthetic']})
        result['frozen_synthetic_only_stability_gate']={
            'evaluated_direction_condition_cells':len(comparisons),
            'all_conditions_passed':bool(all(all(v for k,v in x.items() if k not in ('direction','condition')) for x in comparisons)),
            'cells':comparisons,'real_physics_deployment_authorized':False}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('cal','test'))
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--s72-calibration',type=Path,required=True)
    p.add_argument('--s75-calibration',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Frozen output must never be overwritten')
    j=run(args.visualmetro,args.photo_tiles,args.s72_calibration,args.phase,
          frozen_s75_cal=args.s75_calibration,smoke=args.smoke)
    args.out.parent.mkdir(exist_ok=True,parents=True)
    args.out.write_text(json.dumps(j,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print('S75',args.phase,'saved',args.out,'ID',EXPERIMENT_ID,flush=True)
if __name__=='__main__':main()
