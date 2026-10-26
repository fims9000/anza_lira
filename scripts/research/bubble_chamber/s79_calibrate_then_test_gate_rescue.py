#!/usr/bin/env python3
"""S79: pixel-supported gate rescue, source CAL then fresh TEST, no GT in inference."""
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
from bubble_chamber.s56_global_path_assembly import pair_rows,scored_edges
from bubble_chamber.s59_candidate_retrieval import filter_rows
from bubble_chamber.s69_orientation_invariant_audit import audit_orientation_invariant_synthetic
from bubble_chamber.s79_pixel_gate_rescue import ARMS,solve_s79_from_pixels
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75
from scripts.research.bubble_chamber import s73_stress_test_pixel_graph as s73

EXPERIMENT_ID='ANZA-LIRA-BC-S79-20261026'
SCHEMA='S79_FROZEN_PIXEL_SUPPORTED_GATE_RESCUE_SYNTHETIC_ONLY_V1'
PROTOCOL=ROOT/'docs/bubble_chamber/S79_FROZEN_PIXEL_SUPPORTED_GATE_RESCUE_2026_10_26.md'
OPERATOR=ROOT/'bubble_chamber/s79_pixel_gate_rescue.py'
SOURCE_CAL=ROOT/'docs/bubble_chamber/S72_PUBLIC_FROZEN_CAL_2026_10_26.json'
S72_SHA256='1ed7e58667aaab1a7d5ba91252db64b38bf8d3da517396b9e18e9e6bb484a952'
CAL_SEEDS={'005':20267901,'006':20267911}
TEST_SEEDS={'005':(20267903,20267904),'006':(20267913,20267914)}
N_CAL=16
N_TEST=24
CAL_CONDITIONS=('clean','blur_sigma_11')
TEST_CONDITIONS=('clean','blur_sigma_11','blur_sigma_16')
BOOTSTRAP=1200
METRICS=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
         'complete_three_piece_paths_synthetic','scene_any_noncorrect',
         'selected_provisional_links','candidate_polylines','candidate_pairs_after_strict_gate',
         'solver_failure','raw_S77_gap_hypotheses','pixel_alternatives_added',
         'duplicates_same_ports_suppressed','rescued_from_strict_pair_gate',
         'different_port_competitors','selected_pixel_alternatives','original_polylines_retained')


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def infer_arms_from_pixels(gray,scorer,cutoff,graph_arm):
    """Pixel-only inference. No oracle/reference/label argument can be passed."""
    if gray.shape!=(192,192) or gray.dtype!=np.uint8:raise ValueError('192x192 uint8 pixel input')
    source=s68.infer_pixel_only(gray,scorer,cutoff,graph_arm)
    lines=source['polylines']
    baseline={**source,'raw_S77_gap_hypotheses':0,'pixel_alternatives_added':0,
       'duplicates_same_ports_suppressed':0,'rescued_from_strict_pair_gate':0,
       'different_port_competitors':0,'selected_pixel_alternatives':0,'original_polylines_retained':len(lines)}
    outputs={'baseline':baseline}
    if len(lines)<3:
        for arm in ARMS[1:]:outputs[arm]={**baseline}
        return outputs
    all_rows=pair_rows(lines)
    all_scores=np.asarray(scorer(all_rows),float)
    by_pair={(r['i'],r['j']):float(score) for r,score in zip(all_rows,all_scores)}
    rows=filter_rows(all_rows,'strict_s39')
    if len(rows)!=source['pair_count']:
        raise AssertionError('Frozen S56 strict pair gate count mismatch')
    geom=scored_edges(rows,np.asarray([by_pair[(r['i'],r['j'])] for r in rows],float))
    for arm in ARMS[1:]:
        candidate=solve_s79_from_pixels(gray,lines,geom,all_rows,all_scores,
            cutoff,arm,baseline_chosen=source['chosen'],baseline_success=source['solver_success'])
        if len(candidate['polylines'])!=len(lines) or any(a is not b for a,b in zip(candidate['polylines'],lines)):
            raise AssertionError('Alternative layer erased an original machine line')
        if len(candidate['chosen'])>len(lines)-1:
            raise AssertionError('Not a bounded forest')
        outputs[arm]=candidate
    return outputs


def sample(photos,templates,seed,n,conditions,scorer,cutoff,graph_arm):
    if not photos:raise ValueError('No private H50Z crop bank')
    rng=np.random.default_rng(seed)
    collected={c:{a:[] for a in ARMS} for c in conditions}
    for idx in range(n):
        photo=photos[int(rng.integers(len(photos)))]
        injected=make_three_piece_scene(photo,templates,
            seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[idx%len(SCENARIOS)])
        for condition in conditions:
            pix=s75.condition_pixels(injected.gray,condition)
            # Pixel-only inference finishes before source truth is accessed.
            estimates=infer_arms_from_pixels(pix,scorer,cutoff,graph_arm)
            for arm,pred in estimates.items():
                report=audit_orientation_invariant_synthetic(pred,injected.nodes)
                row={k:int(report[k]) for k in METRICS if k in report}
                for k in ('raw_S77_gap_hypotheses','pixel_alternatives_added',
                          'duplicates_same_ports_suppressed','rescued_from_strict_pair_gate',
                          'different_port_competitors','selected_pixel_alternatives','original_polylines_retained'):
                    row[k]=int(pred[k])
                collected[condition][arm].append(row)
        if idx%8==7:print('S79',seed,'synthetic scenes',idx+1,'/',n,flush=True)
    return collected


def aggregate(rows):
    if not rows:raise ValueError('No scenes')
    if any(any(k not in row for k in METRICS) for row in rows):raise ValueError('Missing full metrics')
    return {**{k:int(sum(x[k] for x in rows)) for k in METRICS},'scenes':len(rows)}


def choose_on_CAL(stats):
    if set(stats)!=set(ARMS) or any(set(x)!=set(CAL_CONDITIONS) for x in stats.values()):
        raise ValueError('Frozen arms/conditions absent from CAL')
    base=stats['baseline'];eligible=[]
    for arm in ARMS[1:]:
        cur=stats[arm]
        clean=cur['clean'];blur=cur['blur_sigma_11']
        bc=base['clean'];bb=base['blur_sigma_11']
        if (blur['selected_synthetic_correct_links']>=bb['selected_synthetic_correct_links']+2
            and blur['synthetic_noncorrect_selected']<=bb['synthetic_noncorrect_selected']
            and clean['synthetic_noncorrect_selected']<=bc['synthetic_noncorrect_selected']
            and clean['selected_synthetic_correct_links']>=bc['selected_synthetic_correct_links']-1
            and blur['complete_three_piece_paths_synthetic']>=bb['complete_three_piece_paths_synthetic']
            and clean['complete_three_piece_paths_synthetic']>=bc['complete_three_piece_paths_synthetic']
            and blur['solver_failure']==clean['solver_failure']==0):
            eligible.append(arm)
    chosen=min(eligible,key=lambda a:(stats[a]['blur_sigma_11']['synthetic_noncorrect_selected'],
        -stats[a]['blur_sigma_11']['selected_synthetic_correct_links'],ARMS.index(a))) if eligible else 'baseline'
    return {'CAL_selected':chosen,'eligible':eligible,'no_native_physics_auto_joins':True}


def paired(a,b,seed):
    if not a or len(a)!=len(b):raise ValueError('Unpaired scenarios')
    rng=np.random.default_rng(seed);draw=rng.integers(0,len(a),size=(BOOTSTRAP,len(a)))
    return {k:{'delta_total':int((d:=np.asarray([y[k]-x[k] for x,y in zip(a,b)],float)).sum()),
               'mean_delta_per_scene':float(d.mean()),
               'paired_scene_bootstrap_95pct':[float(x) for x in np.quantile(d[draw].mean(1),[.025,.975])]}
            for k in ('selected_synthetic_correct_links','synthetic_noncorrect_selected',
                      'scene_any_noncorrect','complete_three_piece_paths_synthetic')}


def run(visualmetro:Path,tiles:Path,phase:str,*,calibration:Path|None=None,smoke:bool=False):
    if phase not in ('cal','test'):raise ValueError('Invalid phase')
    if digest(SOURCE_CAL)!=S72_SHA256:raise ValueError('Frozen S72 CAL drift')
    frozen_source=s75.frozen_s72(SOURCE_CAL)
    templates,split=s72._source_data(visualmetro)
    if split!='448f21ca3a709010ed972a596ed046cadff0643112905fd1d029c91f8b14faef':
        raise ValueError('Frozen S37 frame split drift')
    photos={s:sorted(tiles.glob(f'H50Z_030_01_{s}_*.png')) for s in ('005','006')}
    if len(photos['005'])!=21 or len(photos['006'])!=18:raise ValueError('Private tile bank mismatch')
    result={'schema':SCHEMA,'experiment_id':EXPERIMENT_ID,'phase':phase,
            'smoke_NON_CONFIRMATORY':bool(smoke),'protocol_sha256':digest(PROTOCOL),
            'operator_sha256':digest(OPERATOR),'S72_CAL_SHA256':S72_SHA256,
            'S72_code_locks':s72.LOCKS,'source_frame_split_sha256':split,
            'synthetic_only':True,'film_070_NOT_H50Z_physics_GT':True,
            'H50Z_backgrounds_one_physical_run':True,
            'native_H50Z_GT_count':0,'native_physical_auto_joins_authorized':False,
            'arms':ARMS,'directions':{}}
    prior=None
    if phase=='test':
        if calibration is None or not calibration.is_file():raise FileNotFoundError('Frozen complete S79 CAL required')
        prior=json.loads(calibration.read_text(encoding='utf8'))
        for key in ('schema','protocol_sha256','operator_sha256','S72_CAL_SHA256','S72_code_locks',
                    'source_frame_split_sha256'):
            if result[key]!=prior.get(key):raise ValueError('S79 immutable CAL drift: '+key)
        if prior.get('smoke_NON_CONFIRMATORY') or prior.get('phase')!='cal':
            raise ValueError('Never confirm TEST against smoke/TEST data')
        result['S79_frozen_CAL_SHA256']=digest(calibration)
    for di,source in enumerate(('005','006')):
        target='006' if source=='005' else '005';label=f'{source}_to_{target}'
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        graph_arm=frozen_source['directions'][label]['CAL_selected']
        if float(cutoff)!=float(frozen_source['directions'][label]['S56_frozen_CAL_cutoff']):
            raise ValueError('S56 source-trained scorer cutoff changed')
        entry={'source_photo':source,'destination_photo':target,'frozen_S72_extractor':graph_arm,
               'original_S56_cutoff':float(cutoff)}
        if phase=='cal':
            metrics=sample(photos[source],templates['cal'],CAL_SEEDS[source],
                           2 if smoke else N_CAL,CAL_CONDITIONS,scorer,cutoff,graph_arm)
            all_stats={arm:{c:aggregate(metrics[c][arm]) for c in CAL_CONDITIONS} for arm in ARMS}
            entry['CAL_all_arms']=all_stats
            entry['CAL_decision']=choose_on_CAL(all_stats)
            print('S79 CAL',label,entry['CAL_decision'],flush=True)
        else:
            frozen=prior['directions'][label]
            if frozen['original_S56_cutoff']!=float(cutoff) or frozen['frozen_S72_extractor']!=graph_arm:
                raise ValueError('S79 frozen CAL scorer drift')
            choice=frozen['CAL_decision']['CAL_selected']
            if choice not in ARMS:raise ValueError('Invalid CAL choice')
            entry['CAL_selected']=choice;entry['test_seeds']={}
            for si,seed in enumerate(TEST_SEEDS[source] if not smoke else TEST_SEEDS[source][:1]):
                observations=sample(photos[target],templates['test'],seed,
                    2 if smoke else N_TEST,TEST_CONDITIONS,scorer,cutoff,graph_arm)
                entry['test_seeds'][str(seed)]={cond:{
                    'arms':{arm:aggregate(observations[cond][arm]) for arm in ARMS},
                    'paired_vs_baseline':{arm:paired(observations[cond]['baseline'],observations[cond][arm],
                         seed+20267990+di*200+ci*7+si) for arm in ARMS[1:]}}
                    for ci,cond in enumerate(TEST_CONDITIONS)}
                print('S79 TEST',label,'seed',seed,'CAL chose',choice,flush=True)
        result['directions'][label]=entry
    if phase=='test':
        result['all_TEST_aggregates']={cond:{arm:{key:int(sum(
           samples[cond]['arms'][arm][key] for d in result['directions'].values()
           for samples in d['test_seeds'].values())) for key in (*METRICS,'scenes')}
           for arm in ARMS} for cond in TEST_CONDITIONS}
        result['CAL_selected_TEST_aggregate']={cond:{key:int(sum(
           samples[cond]['arms'][d['CAL_selected']][key] for d in result['directions'].values()
           for samples in d['test_seeds'].values())) for key in (*METRICS,'scenes')}
           for cond in TEST_CONDITIONS}
    return result


def main():
    a=argparse.ArgumentParser();a.add_argument('phase',choices=('cal','test'))
    a.add_argument('--visualmetro',type=Path,required=True)
    a.add_argument('--photo-tiles',type=Path,required=True)
    a.add_argument('--calibration',type=Path)
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--smoke',action='store_true')
    args=a.parse_args()
    if args.out.exists():raise FileExistsError('Never replace a frozen experiment output')
    j=run(args.visualmetro,args.photo_tiles,args.phase,calibration=args.calibration,smoke=args.smoke)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('S79 output',args.out,'sha256',digest(args.out),flush=True)
if __name__=='__main__':main()
