#!/usr/bin/env python3
"""S73: fresh paired domain-shift experiment, immutable S72 CAL selection.

All ground-truth associations are posthoc and are synthetically injected;
never treat them as independent real physical bubble-chamber track labels.
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

from bubble_chamber.s73_image_robustness import CONDITIONS,perturb
from bubble_chamber.s69_orientation_invariant_audit import audit_orientation_invariant_synthetic
from bubble_chamber.s56_three_piece_scenes import make_three_piece_scene,SCENARIOS
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72

EXPERIMENT_ID='ANZA-LIRA-BC-S73-20261026'
SCHEMA='S73_FRESH_CORRECTED_PIXEL_DOMAIN_STRESS_SYNTHETIC_ONLY_V1'
PROTOCOL=ROOT/'docs/bubble_chamber/S73_FROZEN_PIXEL_DOMAIN_STRESS_TEST_2026_10_26.md'
FROZEN_S72_CAL_SHA256='1ed7e58667aaab1a7d5ba91252db64b38bf8d3da517396b9e18e9e6bb484a952'
FROZEN_S72_PROTOCOL_SHA256='738a97157a8a6ae6ed118a61af2e8e54ee8fa2be479db7e155b9bd0f87a865e2'
SEEDS={'005':(20267303,20267304),'006':(20267313,20267314)}
N_SCENES=24
BOOTSTRAP=1200
AUDIT_METRICS=('candidate_polylines','candidate_pairs_after_strict_gate',
 'source_node_matches','synthetic_decoy_matched','selected_provisional_links',
 'selected_synthetic_correct_links','synthetic_noncorrect_selected',
 'scene_any_noncorrect','complete_three_piece_paths_synthetic','solver_failure')
PRIMARY_METRICS=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
 'scene_any_noncorrect','complete_three_piece_paths_synthetic')


def digest(p):return sha256(Path(p).read_bytes()).hexdigest()


def load_locked_cal(calibration:Path)->dict:
    if digest(calibration)!=FROZEN_S72_CAL_SHA256:
        raise ValueError('S72 CAL changed since preregistration')
    j=json.loads(calibration.read_text())
    if j.get('schema')!=s72.SCHEMA or j.get('phase')!='cal':
        raise ValueError('Only original S72 CAL is valid')
    if j.get('protocol_sha256')!=FROZEN_S72_PROTOCOL_SHA256:
        raise ValueError('Original S72 frozen protocol changed')
    if j.get('frozen_source_sha256')!=s72.LOCKS or len(j.get('directions',{}))!=2:
        raise ValueError('Incorrect S72 source fingerprint')
    if (j['directions']['005_to_006']['CAL_selected']!='s68_branch_endpoint_refine'
      or j['directions']['006_to_005']['CAL_selected']!='s39_original'):
        raise ValueError('Frozen S72 CAL direction choice changed')
    return j


def stress_seed(scene_seed:int,scene_idx:int,kind_idx:int)->int:
    if scene_seed not in sum((list(x) for x in SEEDS.values()),[]):
        raise ValueError('Not an S73 locked scene seed')
    if not isinstance(scene_idx,int) or not 0<=scene_idx<N_SCENES:
        raise ValueError('Invalid S73 scene index')
    if not isinstance(kind_idx,int) or not 0<=kind_idx<len(CONDITIONS):
        raise ValueError('Invalid condition index')
    return int((scene_seed*100_003+scene_idx*157+kind_idx*65537)%(2**32))


def _paired(a:list[dict],b:list[dict],seed:int)->dict:
    if len(a)!=len(b) or not a:
        raise ValueError('Comparison must be same artificial scene in same order')
    rng=np.random.default_rng(seed)
    picks=rng.integers(0,len(a),size=(BOOTSTRAP,len(a)))
    result={}
    for key in PRIMARY_METRICS:
        d=np.asarray([z[key]-x[key] for x,z in zip(a,b)],dtype=float)
        result[key]={'delta_total':int(d.sum()),'delta_mean_per_scene':float(d.mean()),
          'paired_scene_bootstrap_95pct':[float(v) for v in np.quantile(d[picks].mean(axis=1),[.025,.975])]}
    return result


def _sum_rows(rows:list[dict])->dict:
    if not rows:raise ValueError('No TEST observations')
    return {'scene_count':len(rows),**{k:int(sum(r[k] for r in rows)) for k in AUDIT_METRICS}}


def run(visualmetro:Path,tiles:Path,calibration:Path,*,smoke:bool=False)->dict:
    for path,expected in s72.LOCKS.items():
        if digest(ROOT/path)!=expected:raise ValueError('Original source/evaluator was modified: '+path)
    if digest(s72.PROTOCOL)!=FROZEN_S72_PROTOCOL_SHA256:
        raise ValueError('S72 protocol drift')
    cal=load_locked_cal(calibration)
    templates,split_sha=s72._source_data(visualmetro)
    if cal['source_frame_split_sha256']!=split_sha:
        raise ValueError('Historical S72 source split changed')
    counts={p:len(list(tiles.glob(f'H50Z_030_01_{p}_*.png'))) for p in ('005','006')}
    if counts!={'005':21,'006':18}:
        raise ValueError('Private H50Z photo-bank shape changed')
    public={'experiment_id':EXPERIMENT_ID,'schema':SCHEMA,
        'phase':'smoke_NOT_confirmatory' if smoke else 'fresh_stress_TEST',
        'S72_CAL_sha256':FROZEN_S72_CAL_SHA256,
        'S72_protocol_sha256':FROZEN_S72_PROTOCOL_SHA256,
        'S73_protocol_sha256':digest(PROTOCOL),
        'S72_frozen_source_hashes':s72.LOCKS,
        'source_frame_split_sha256':split_sha,
        'conditions':list(CONDITIONS),'methods':list(s68.ARMS),
        'new_synthetic_scene_seeds':{k:list(v) for k,v in SEEDS.items()},
        'scenes_per_seed':2 if smoke else N_SCENES,
        'native_H50Z_verified_GT_count':0,'native_physical_auto_joins_authorized':False,
        'H50Z_photos_from_one_physical_run':True,
        'film_070_geometry_NOT_native_H50Z_GT':True,
        'all_labels_are_synthetic_injection_only':True,
        'directions':{}}
    for source in ('005','006'):
        target='006' if source=='005' else '005'
        name=f'{source}_to_{target}'
        chosen=cal['directions'][name]['CAL_selected']
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        if float(cutoff)!=float(cal['directions'][name]['S56_frozen_CAL_cutoff']):
            raise ValueError('Scorer calibration drift since S72')
        photos=sorted(tiles.glob(f'H50Z_030_01_{target}_*.png'))
        entry={'frozen_CAL_selected':chosen,'source_photo':source,'target_photo':target,
               'frozen_S56_cutoff':float(cutoff),'test_seeds':{}}
        for seed in (SEEDS[source] if not smoke else SEEDS[source][:1]):
            rng=np.random.default_rng(seed)
            samples={kind:{arm:[] for arm in s68.ARMS} for kind in CONDITIONS}
            for index in range(2 if smoke else N_SCENES):
                background=photos[int(rng.integers(len(photos)))]
                artificial=make_three_piece_scene(background,templates['test'],
                    seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[index%len(SCENARIOS)])
                for kind_index,kind in enumerate(CONDITIONS):
                    pixels=perturb(artificial.gray,kind,stress_seed(seed,index,kind_index))
                    # Blind pixel-only inference FIRST for all methods.
                    predictions={arm:s68.infer_pixel_only(pixels,scorer,cutoff,arm)
                                 for arm in s68.ARMS}
                    # Synthetic true reference enters strictly AFTER all predictions.
                    for arm,pred in predictions.items():
                        audit=audit_orientation_invariant_synthetic(pred,artificial.nodes)
                        samples[kind][arm].append({metric:audit[metric] for metric in AUDIT_METRICS})
                if index%8==7:
                    print('S73',name,'seed',seed,'scene',index+1,flush=True)
            out={'conditions':{}}
            for kidx,kind in enumerate(CONDITIONS):
                control=samples['clean']
                out['conditions'][kind]={
                  'arms':{arm:_sum_rows(samples[kind][arm]) for arm in s68.ARMS},
                  'CAL_selected_minus_S39':_paired(samples[kind]['s39_original'],
                     samples[kind][chosen],seed+773+kidx),
                  'domain_shift_vs_clean':{arm:_paired(control[arm],samples[kind][arm],
                     seed+1700+kidx*19+ai) for ai,arm in enumerate(s68.ARMS)}
                }
            entry['test_seeds'][str(seed)]=out
        public['directions'][name]=entry
    # Descriptive stringent gate predeclared, no tuning or selection from test outcomes.
    overall={}
    for kind in CONDITIONS:
        per_arm={arm:_sum_rows_from_summaries([row['conditions'][kind]['arms'][arm]
                  for d in public['directions'].values() for row in d['test_seeds'].values()])
                 for arm in s68.ARMS}
        direction_selected=_sum_rows_from_summaries([
                row['conditions'][kind]['arms'][d['frozen_CAL_selected']]
                for d in public['directions'].values() for row in d['test_seeds'].values()])
        overall[kind]={'arm_totals':per_arm,'frozen_S72_direction_choice':direction_selected}
    public['overall_condition_totals']=overall
    criterions=[]
    for name,d in public['directions'].items():
        for kind in CONDITIONS[1:]:
            base=_sum_rows_from_summaries([x['conditions'][kind]['arms']['s39_original']
                                           for x in d['test_seeds'].values()])
            selected=_sum_rows_from_summaries([x['conditions'][kind]['arms'][d['frozen_CAL_selected']]
                                               for x in d['test_seeds'].values()])
            criterions.append(bool(
                selected['selected_synthetic_correct_links']>=base['selected_synthetic_correct_links']
                and selected['complete_three_piece_paths_synthetic']>=base['complete_three_piece_paths_synthetic']
                and selected['synthetic_noncorrect_selected']<=base['synthetic_noncorrect_selected']))
    public['frozen_method_stress_gate']={'conditional_synthetic_only_passed':bool(all(criterions)),
          'evaluated_direction_stressor_cells':len(criterions),
          'native_physics_promotion_authorized':False,
          'no_test_set_method_selection':True}
    return public


def _sum_rows_from_summaries(rows):
    if not rows:raise ValueError('No aggregate rows')
    return {k:int(sum(x[k] for x in rows)) for k in ('scene_count',*AUDIT_METRICS)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Never replace prior research results')
    result=run(a.visualmetro,a.photo_tiles,a.calibration,smoke=a.smoke)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('S73 experiment',EXPERIMENT_ID,'passed synthetic-only gate',
          result['frozen_method_stress_gate']['conditional_synthetic_only_passed'],flush=True)
if __name__=='__main__':main()
