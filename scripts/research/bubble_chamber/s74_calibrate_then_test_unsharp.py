#!/usr/bin/env python3
"""S74: immutable CAL, fresh TEST, blind sharpening with S69 synthetic evaluator."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np

from bubble_chamber.s74_unsharp_preprocess import ARMS,preprocess,choose_cal_only
from bubble_chamber.s73_image_robustness import perturb
from bubble_chamber.s69_orientation_invariant_audit import audit_orientation_invariant_synthetic
from bubble_chamber.s56_three_piece_scenes import make_three_piece_scene,SCENARIOS
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
from scripts.research.bubble_chamber import s73_stress_test_pixel_graph as s73

EXPERIMENT_ID='ANZA-LIRA-BC-S74-20261026'
SCHEMA='S74_FROZEN_CAL_UNSHARP_THEN_FRESH_SYNTHETIC_TEST_V1'
PROTOCOL=ROOT/'docs/bubble_chamber/S74_FROZEN_UNSHARP_CAL_TO_FRESH_TEST_2026_10_26.md'
PREDICTOR=ROOT/'bubble_chamber/s74_unsharp_preprocess.py'
CAL_SEEDS={'005':20267401,'006':20267411}
TEST_SEEDS={'005':(20267403,20267404),'006':(20267413,20267414)}
N_CAL=24
N_TEST=24
CONDITIONS=('clean','blur_sigma_11')
METRICS=tuple(s68.METRICS)


def digest(p):return sha256(Path(p).read_bytes()).hexdigest()


def freeze_originals(calfile:Path):
    for path,expected in s72.LOCKS.items():
        if digest(ROOT/path)!=expected:raise ValueError('Original S72 source changed: '+path)
    return s73.load_locked_cal(calfile)


def sample(photos:list[Path],templates,seed:int,n:int,scorer,cutoff:float,method:str):
    if method not in s68.ARMS:raise ValueError('Only unchanged S72 CAL-selected detector')
    rng=np.random.default_rng(seed)
    results={cond:{arm:[] for arm in ARMS} for cond in CONDITIONS}
    for index in range(n):
        image=photos[int(rng.integers(len(photos)))]
        scene=make_three_piece_scene(image,templates,
            seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[index%len(SCENARIOS)])
        for cond in CONDITIONS:
            pixels=perturb(scene.gray,cond,seed=0) # blur deterministic; seed unused here
            # Prediction from pixels only; synthetic reference must not enter either predictor.
            predictions={arm:s68.infer_pixel_only(preprocess(pixels,arm),scorer,cutoff,method)
                         for arm in ARMS}
            for arm,pred in predictions.items():
                a=audit_orientation_invariant_synthetic(pred,scene.nodes)
                results[cond][arm].append({metric:a[metric] for metric in METRICS})
    return results


def summarize(rows):
    if not rows:raise ValueError('No paired scenes')
    return s68.sums(rows)


def run(visualmetro:Path,tiles:Path,s72_calibration:Path,phase:str,
        *,frozen_s74_cal:Path|None=None,smoke=False):
    if phase not in ('cal','test'):raise ValueError('Frozen S74 phase is CAL or TEST')
    old=freeze_originals(s72_calibration)
    templates,split_sha=s72._source_data(visualmetro)
    if split_sha!=old['source_frame_split_sha256']:
        raise ValueError('Changed S37 whole-frame source split')
    photo_banks={p:sorted(tiles.glob(f'H50Z_030_01_{p}_*.png')) for p in ('005','006')}
    if [len(photo_banks[p]) for p in ('005','006')]!=[21,18]:
        raise ValueError('Missing private H50Z photo banks')
    out={'schema':SCHEMA,'phase':phase,'smoke_NOT_CONFIRMATORY':smoke,'experiment_id':EXPERIMENT_ID,
         'S74_protocol_sha256':digest(PROTOCOL),'S74_image_operator_sha256':digest(PREDICTOR),
         'S72_calibration_sha256':s73.FROZEN_S72_CAL_SHA256,
         'S72_source_code_fingerprints':s72.LOCKS,'source_frame_split_sha256':split_sha,
         'all_metrics_synthetic_only':True,'native_H50Z_verified_GT_count':0,
         'native_physical_auto_join_authorized':False,'H50Z_backgrounds_one_physical_run':True,
         'film070_shapes_not_H50Z_native_GT':True,'directions':{}}
    old_s74=None
    if phase=='test':
        if frozen_s74_cal is None or not frozen_s74_cal.is_file():
            raise FileNotFoundError('Fresh test requires complete S74 immutable CAL')
        old_s74=json.loads(frozen_s74_cal.read_text())
        for k in ('schema','S74_protocol_sha256','S74_image_operator_sha256',
                  'S72_calibration_sha256','S72_source_code_fingerprints',
                  'source_frame_split_sha256'):
            if out[k]!=old_s74.get(k):raise ValueError('Changed CAL provenance '+k)
        if old_s74['phase']!='cal' or old_s74['smoke_NOT_CONFIRMATORY']:
            raise ValueError('Cannot test with smoke / non-CAL snapshot')
        out['frozen_S74_CAL_sha256']=digest(frozen_s74_cal)
    for source in ('005','006'):
        target='006' if source=='005' else '005'
        direction=f'{source}_to_{target}'
        frozen_method=old['directions'][direction]['CAL_selected']
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        if float(cutoff)!=float(old['directions'][direction]['S56_frozen_CAL_cutoff']):
            raise ValueError('Original S56 source CAL cutoff drift')
        row={'source_photo':source,'test_photo':target,'S72_CAL_method':frozen_method,
             'frozen_S56_cutoff':float(cutoff)}
        if phase=='cal':
            pairs=sample(photo_banks[source],templates['cal'],CAL_SEEDS[source],
                         2 if smoke else N_CAL,scorer,cutoff,frozen_method)
            stats={arm:{cond:summarize(pairs[cond][arm]) for cond in CONDITIONS} for arm in ARMS}
            chosen=choose_cal_only(stats)
            row['CAL_aggregates']=stats;row['CAL_only_choice']=chosen
            print('S74 CAL',direction,chosen,flush=True)
        else:
            prior=old_s74['directions'][direction]
            if prior['S72_CAL_method']!=frozen_method or prior['frozen_S56_cutoff']!=float(cutoff):
                raise ValueError('Saved S74 source method/cutoff modified')
            selected=prior['CAL_only_choice']['chosen_CAL_only']
            if selected not in ARMS:raise ValueError('Unfrozen CAL choice')
            row['frozen_CAL_unsharp_choice']=selected
            row['test_seeds']={}
            for seed in (TEST_SEEDS[source] if not smoke else TEST_SEEDS[source][:1]):
                res=sample(photo_banks[target],templates['test'],seed,
                           2 if smoke else N_TEST,scorer,cutoff,frozen_method)
                row['test_seeds'][str(seed)]={'conditions':{
                    cond:{'arms':{arm:summarize(res[cond][arm]) for arm in ARMS},
                          'chosen_minus_identity':s68.paired(res[cond]['identity'],res[cond][selected],seed+74+ci)}
                    for ci,cond in enumerate(CONDITIONS)}}
                print('S74 TEST',direction,seed,'frozen',selected,
                    'correct clean/blur',[row['test_seeds'][str(seed)]['conditions'][c]['arms'][selected]
                                          ['selected_synthetic_correct_links'] for c in CONDITIONS],flush=True)
        out['directions'][direction]=row
    if phase=='test':
        out['all_TEST_conditions']={cond:{arm:{k:int(sum(
             d['test_seeds'][str(seed)]['conditions'][cond]['arms'][arm][k]
             for src,d in zip(('005','006'),out['directions'].values())
             for seed in TEST_SEEDS[src])) for k in (*METRICS,'scene_count')}
             for arm in ARMS} for cond in CONDITIONS}
        out['CAL_selected_TEST']={cond:{k:int(sum(
             row['conditions'][cond]['arms'][d['frozen_CAL_unsharp_choice']][k]
             for d in out['directions'].values() for row in d['test_seeds'].values()))
             for k in (*METRICS,'scene_count')} for cond in CONDITIONS}
        out['native_physics_auto_promotion_authorized']=False
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('cal','test'))
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--s72-calibration',type=Path,required=True)
    p.add_argument('--s74-calibration',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Cannot overwrite frozen research result')
    res=run(a.visualmetro,a.photo_tiles,a.s72_calibration,a.phase,
            frozen_s74_cal=a.s74_calibration,smoke=a.smoke)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(res,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
