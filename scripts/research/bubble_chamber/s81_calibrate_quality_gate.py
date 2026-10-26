#!/usr/bin/env python3
"""S81 image-quality-gated S80 scale consensus: new CAL then fresh TEST.

The gate is fitted using source-CAL *pixels only*; synthetic truth is consumed
solely by the separate CAL selection / heldout evaluation after all predictions.
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
from bubble_chamber.s81_quality_gated_instances import (
    ARMS,fine_coarse_ratio,thresholds_from_CAL_clean,activated,choose_cal_only)
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75
from scripts.research.bubble_chamber import s80_calibrate_then_test_multiscale as s80

SCHEMA='S81_QUALITY_GATED_MULTISCALE_CAL_THEN_FRESH_TEST_SYNTHETIC_ONLY_V1'
EXPERIMENT_ID='ANZA-LIRA-BC-S81-20261026'
PROTOCOL=ROOT/'docs/bubble_chamber/S81_FROZEN_PIXEL_QUALITY_GATED_MULTISCALE_2026_10_26.md'
OPERATOR=ROOT/'bubble_chamber/s81_quality_gated_instances.py'
SOURCE_S80_CAL=ROOT/'docs/bubble_chamber/S80_PUBLIC_FROZEN_CAL_2026_10_26.json'
CAL_SEEDS={'005':20268101,'006':20268111}
TEST_SEEDS={'005':(20268103,20268104),'006':(20268113,20268114)}
CAL_SCENES=16
TEST_SCENES=24
CAL_CONDITIONS=('clean','blur_sigma_11')
TEST_CONDITIONS=('clean','blur_sigma_11','blur_sigma_16')
MEASURES=(*s80.MEASURES,'quality_gate_triggered')


def digest(p):return sha256(Path(p).read_bytes()).hexdigest()


def infer_from_pixels(gray,scorer,cutoff,graph_arm,quality_thresholds):
    if gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('S81 inference only sees 192x192 uint8 photograph pixels')
    if set(quality_thresholds)!=set(ARMS[1:]):
        raise ValueError('S81 needs BOTH immutable source-CAL quality thresholds')
    score=fine_coarse_ratio(gray)
    # All predictions are image-only. Source artificial identities never enter.
    original=s80.infer_from_pixels(gray,scorer,cutoff,graph_arm)
    baseline=original['baseline']
    out={'baseline':{**baseline,'quality_gate_triggered':0}}
    for arm in ARMS[1:]:
        gate=activated(score,float(quality_thresholds[arm]))
        chosen=original['s80_consensus_add6'] if gate else baseline
        out[arm]={**chosen,'quality_gate_triggered':int(gate)}
        if not gate and (out[arm]['polylines'] is not baseline['polylines'] or
                          out[arm]['chosen'] is not baseline['chosen']):
            raise AssertionError('Quality gate changed baseline without activation')
    return out


def source_CAL_pixel_quality(photos,templates,seed,n):
    if not photos:raise ValueError('Missing authentic source background')
    rng=np.random.default_rng(seed)
    clean=[]
    for k in range(n):
        scene=make_three_piece_scene(photos[int(rng.integers(len(photos)))],templates,
             seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
        clean.append(fine_coarse_ratio(scene.gray))
    return thresholds_from_CAL_clean(clean)


def sample(photos,templates,seed,n,conditions,scorer,cutoff,graph_arm,thresholds):
    if not photos:raise ValueError('Empty H50Z photo domain')
    rng=np.random.default_rng(seed)
    out={cond:{arm:[] for arm in ARMS} for cond in conditions}
    for k in range(n):
        scene=make_three_piece_scene(photos[int(rng.integers(len(photos)))],templates,
                seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
        for condition in conditions:
            pixels=s75.condition_pixels(scene.gray,condition)
            blind=infer_from_pixels(pixels,scorer,cutoff,graph_arm,thresholds)
            audit=s80.score_after_inference(blind,scene.nodes)
            for arm in ARMS:
                value={**audit['baseline']} if arm=='baseline' else audit[arm]
                value['quality_gate_triggered']=int(blind[arm]['quality_gate_triggered'])
                out[condition][arm].append(value)
        if (k+1)%8==0:print('S81',seed,k+1,'/',n,flush=True)
    return out


def aggregate(rows):
    if not rows or any(set(MEASURES)-set(r) for r in rows):
        raise ValueError('Incomplete S81 post-hoc synthetic audit')
    return {**{k:int(sum(row[k] for row in rows)) for k in MEASURES},'scenes':len(rows)}


def paired(a,b,seed):
    if len(a)!=len(b) or not a:raise ValueError('Scenes not paired')
    rng=np.random.default_rng(seed)
    drawn=rng.integers(0,len(a),size=(1200,len(a)))
    metrics=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
             'complete_three_piece_paths_synthetic','shape_true_fragments_matched',
             'shape_true_one_or_both_endpoints_far','scene_any_noncorrect')
    return {key:{'delta_total':int((v:=np.asarray([y[key]-x[key] for x,y in zip(a,b)],int)).sum()),
                 'per_scene':float(v.mean()),
                 'paired_scene_bootstrap_95pct':[float(q) for q in np.quantile(v[drawn].mean(1),(.025,.975))]}
            for key in metrics}


def run(visualmetro,tiles,phase,calibration=None,smoke=False):
    if phase not in ('cal','test'):raise ValueError('CAL then TEST only')
    if digest(s80.S72_CAL)!=s80.LOCKED_S72_SHA256:
        raise ValueError('Original S72 source-CAL changed')
    source80=json.loads(SOURCE_S80_CAL.read_text())
    if source80['operator_sha256']!=s80.digest(s80.OPERATOR):
        raise ValueError('S80 scale-specific source changed after historical CAL')
    frozen_s72=s75.frozen_s72(s80.S72_CAL)
    templates,split=s72._source_data(visualmetro)
    if split!=s80.SPLIT_SHA:raise ValueError('Frozen source-frame split drift')
    photos={p:sorted(tiles.glob(f'H50Z_030_01_{p}_*.png')) for p in ('005','006')}
    if (len(photos['005']),len(photos['006']))!=(21,18):
        raise ValueError('Private tiles must be same two H50Z photographic runs/views')
    public={'schema':SCHEMA,'phase':phase,'experiment_id':EXPERIMENT_ID,
            'smoke_NOT_CONFIRMATORY':bool(smoke),'protocol_sha256':digest(PROTOCOL),
            'operator_sha256':digest(OPERATOR),'frozen_S80_operator_sha256':source80['operator_sha256'],
            'S72_CAL_sha256':s80.LOCKED_S72_SHA256,'source_frame_split_sha256':split,
            'S72_code_locks':s72.LOCKS,'source_photo_run_count':1,'native_H50Z_GT':0,
            'physics_auto_join_authorized':False,'synthetic_only':True,
            'CAL_condition_quantiles':{'q20':.20,'q40':.40},'arms':ARMS,'directions':{}}
    frozen=None
    if phase=='test':
        if calibration is None or not Path(calibration).is_file():
            raise FileNotFoundError('S81 fresh TEST requires saved CAL first')
        frozen=json.loads(Path(calibration).read_text())
        for key in ('schema','protocol_sha256','operator_sha256','frozen_S80_operator_sha256',
                    'S72_CAL_sha256','source_frame_split_sha256','S72_code_locks'):
            if frozen.get(key)!=public[key]:raise ValueError('S81 source or CAL drift: '+key)
        if frozen.get('phase')!='cal' or frozen.get('smoke_NOT_CONFIRMATORY'):
            raise ValueError('S81 CAL must be complete non-smoke phase')
        public['frozen_CAL_sha256']=digest(calibration)
    for di,source in enumerate(('005','006')):
        target='006' if source=='005' else '005';direction=f'{source}_to_{target}'
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        old=frozen_s72['directions'][direction]
        graph_arm=old['CAL_selected']
        if float(cutoff)!=float(old['S56_frozen_CAL_cutoff']):
            raise ValueError('S56 score thresholds shifted')
        row={'training_photo':source,'heldout_photo':target,'S56_cutoff':float(cutoff),
             'S72_frozen_branch':graph_arm}
        if phase=='cal':
            thresholds=source_CAL_pixel_quality(photos[source],templates['cal'],CAL_SEEDS[source],
                             CAL_SCENES)
            obs=sample(photos[source],templates['cal'],CAL_SEEDS[source],
                       2 if smoke else CAL_SCENES,CAL_CONDITIONS,scorer,cutoff,graph_arm,thresholds)
            stats={arm:{cond:aggregate(obs[cond][arm]) for cond in CAL_CONDITIONS} for arm in ARMS}
            row['CAL_only_quality_thresholds']=thresholds
            row['CAL_arms']=stats
            # No synthetic TEST info, only CAL data enters this selection.
            row['CAL_decision']=choose_cal_only(stats)
            print('S81 CAL',direction,'thresholds',thresholds,'decision',row['CAL_decision'],flush=True)
        else:
            past=frozen['directions'][direction]
            if float(cutoff)!=past['S56_cutoff'] or graph_arm!=past['S72_frozen_branch']:
                raise ValueError('Frozen S56 classifier policy changed')
            thresholds=past['CAL_only_quality_thresholds']
            if set(thresholds)!=set(ARMS[1:]):raise ValueError('Frozen gate thresholds absent')
            row['CAL_selected']=past['CAL_decision']['CAL_selected']
            row['frozen_quality_thresholds']=thresholds
            row['test_seeds']={}
            for si,seed in enumerate((TEST_SEEDS[source] if not smoke else TEST_SEEDS[source][:1])):
                obs=sample(photos[target],templates['test'],seed,
                           2 if smoke else TEST_SCENES,TEST_CONDITIONS,scorer,cutoff,graph_arm,thresholds)
                row['test_seeds'][str(seed)]={cond:{
                    'arms':{arm:aggregate(obs[cond][arm]) for arm in ARMS},
                    'paired_vs_baseline':{arm:paired(obs[cond]['baseline'],obs[cond][arm],
                                  20268190+di*40+si*10+ci) for arm in ARMS[1:]}}
                    for ci,cond in enumerate(TEST_CONDITIONS)}
                print('S81 TEST',direction,seed,'CAL',row['CAL_selected'],flush=True)
        public['directions'][direction]=row
    if phase=='test':
        public['all_TEST_aggregates']={cond:{arm:{key:int(sum(
            seed_row[cond]['arms'][arm][key]
            for d in public['directions'].values()
            for seed_row in d['test_seeds'].values())) for key in (*MEASURES,'scenes')}
            for arm in ARMS} for cond in TEST_CONDITIONS}
        public['CAL_selected_TEST_aggregate']={cond:{key:int(sum(
            seed_row[cond]['arms'][d['CAL_selected']][key]
            for d in public['directions'].values()
            for seed_row in d['test_seeds'].values())) for key in (*MEASURES,'scenes')}
            for cond in TEST_CONDITIONS}
    return public


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('cal','test'))
    ap.add_argument('--visualmetro',type=Path,required=True)
    ap.add_argument('--photo-tiles',type=Path,required=True)
    ap.add_argument('--calibration',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Do not overwrite immutable CAL/TEST')
    result=run(a.visualmetro,a.photo_tiles,a.phase,a.calibration,a.smoke)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('S81 wrote',a.out,'SHA256',digest(a.out),flush=True)
if __name__=='__main__':main()
