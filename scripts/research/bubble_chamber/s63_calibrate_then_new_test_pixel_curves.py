#!/usr/bin/env python3
"""S63 frozen CAL-first pixel-only curve extraction, no native GT or auto-physics joins."""
from __future__ import annotations
import argparse
import json
import hashlib
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_simulated_gaps import extract_track_templates
from bubble_chamber.s56_three_piece_scenes import make_three_piece_scene,SCENARIOS
from bubble_chamber.s63_image_curve_variants import predict_variants,choose_cal_only,METHODS
from bubble_chamber.s62_image_candidate_diagnostics import (proxy_extracted_piece_support,
    scene_upper_bound_from_proxy_support)
from scripts.research.bubble_chamber.s37_benchmark_review_risk import split_source_frames

PROTOCOL=ROOT/'docs/bubble_chamber/S63_FROZEN_CAL_TO_NEW_TEST_PIXEL_CURVES_2026_10_26.md'
CAL_SEEDS={'005':(20266301,20266302),'006':(20266311,20266312)}
TEST_SEEDS={'005':(20266303,20266304),'006':(20266313,20266314)}
SCENES={'cal':12,'test':24}
SCHEMA='s63_machine_only_curves_CAL_then_fresh_TEST_SYNTHETIC_ONLY_v1'


def scene_metrics(gray,proxy_nodes):
    # ALL hypotheses from pixel-only image BEFORE consuming evaluator-only proxy nodes.
    estimates=predict_variants(gray)
    out={}
    for m,pieces in estimates.items():
        checked=[proxy_extracted_piece_support(pieces,q) for q in proxy_nodes]
        x=scene_upper_bound_from_proxy_support(checked)
        out[m]={**{k:int(x[k]) for k in ('proxy_true_fragments_usable',
             'synthetic_decoy_fragment_usable','proxy_true_joins_upper_bound',
             'proxy_complete_three_piece_paths_upper_bound')},'proposal_count':len(pieces)}
    return out


def totals(per_scene:list[dict],*,seed:int)->dict:
    total={m:{'true_fragments':0,'decoy_supported':0,'true_links':0,
        'full_paths':0,'proposals':0,'scenes':0} for m in METHODS}
    keymap={'true_fragments':'proxy_true_fragments_usable',
      'decoy_supported':'synthetic_decoy_fragment_usable',
      'true_links':'proxy_true_joins_upper_bound',
      'full_paths':'proxy_complete_three_piece_paths_upper_bound',
      'proposals':'proposal_count'}
    for row in per_scene:
        for m,d in row.items():
            for k,v in keymap.items():total[m][k]+=d[v]
            total[m]['scenes']+=1
    # Paired bootstrap at scene level, never over pixels or fragment samples.
    rng=np.random.default_rng(seed)
    n=len(per_scene)
    picks=rng.integers(0,n,size=(800,n))
    for m in METHODS:
        a=np.array([r[m]['proxy_true_joins_upper_bound'] for r in per_scene],float)
        b=np.array([r[m]['synthetic_decoy_fragment_usable'] for r in per_scene],float)
        diff=a-np.array([r['s39_original']['proxy_true_joins_upper_bound'] for r in per_scene],float)
        total[m]['scene_bootstrap_true_links_per_scene_95pct']=[float(q) for q in np.quantile(a[picks].mean(1),[.025,.975])]
        total[m]['scene_bootstrap_decoy_support_per_scene_95pct']=[float(q) for q in np.quantile(b[picks].mean(1),[.025,.975])]
        total[m]['scene_bootstrap_paired_delta_links_per_scene_95pct']=[float(q) for q in np.quantile(diff[picks].mean(1),[.025,.975])]
    return total


def run(visualmetro:Path,photo_tiles:Path,*,phase:str,calibration:Path|None=None,smoke:bool=False)->dict:
    if phase not in SCENES:raise ValueError('bad phase')
    if phase=='test' and (not calibration or not calibration.is_file()):
        raise FileNotFoundError('S63 must freeze CAL artifact before TEST')
    m=audited_large_manifest(visualmetro)
    train,cal,test,split_sha=split_source_frames(m)
    assert len(train)==79 and len(cal)==30 and len(test)==36
    templates=extract_track_templates(m,chosen_frames=cal if phase=='cal' else test)
    preamble={'schema':SCHEMA,'phase':phase,'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
      'source_frame_split_sha256':split_sha,'native_H50Z_physics_GT':0,
      'image_inference_never_receives_simulator_polylines':True,
      'synthetic_source_frame_count':len(cal if phase=='cal' else test),
      'photographic_runs_count':1,'no_physical_auto_joins':True,'smoke_ONLY':smoke,
      'methods':METHODS,'directions':{}}
    if phase=='test':
        old=json.loads(calibration.read_text())
        if old['phase']!='cal' or old['protocol_sha256']!=preamble['protocol_sha256'] or old['source_frame_split_sha256']!=split_sha or old['smoke_ONLY']!=smoke:
            raise ValueError('CAL artifact missing / changed / incompatible with locked protocol')
        frozen_selected=old['cal_only_decision']
        preamble['frozen_selected_before_test']=frozen_selected
        preamble['calibration_sha256']=hashlib.sha256(calibration.read_bytes()).hexdigest()
    collected=[]
    seedset=CAL_SEEDS if phase=='cal' else TEST_SEEDS
    count=2 if smoke else SCENES[phase]
    for src,target in [('005','006'),('006','005')]:
        photos=sorted(photo_tiles.glob(f'H50Z_030_01_{target}_*.png'))
        if len(photos)!=(18 if target=='006' else 21):raise ValueError('Invalid private H50Z tile count')
        entry={'target_H50Z_photo':target,'source_shapes_fold':phase.upper(),
               'scene_count':0,'per_seed':{}}
        seeds=seedset[src] if not smoke else seedset[src][:1]
        for seed in seeds:
            rng=np.random.default_rng(seed)
            samples=[]
            for k in range(count):
                photo=photos[int(rng.integers(len(photos)))]
                scene=make_three_piece_scene(photo,templates,seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
                samples.append(scene_metrics(scene.gray,scene.nodes))
            summary=totals(samples,seed=seed+30)
            entry['per_seed'][str(seed)]=summary
            entry['scene_count']+=count
            collected.extend(samples)
            print('S63',phase,src,'->',target,'seed',seed,'scenes',len(samples),
                'orig joins',summary['s39_original']['true_links'],flush=True)
        preamble['directions'][f'{src}_to_{target}']=entry
    preamble['total']=totals(collected,seed=20266337 if phase=='cal' else 20266339)
    if phase=='cal':preamble['cal_only_decision']=choose_cal_only(preamble['total'])
    else:
        preamble['TEST_used_frozen_CAL_selection_only']=True
        preamble['selected_method']=frozen_selected['chosen_on_CAL_only']
    return preamble


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--phase',choices=tuple(SCENES),required=True)
    p.add_argument('--calibration',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Never overwrite frozen CAL/TEST records')
    result=run(a.visualmetro,a.photo_tiles,phase=a.phase,calibration=a.calibration,smoke=a.smoke)
    a.out.parent.mkdir(exist_ok=True,parents=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('S63',a.phase,'frozen selection',result.get('cal_only_decision',result.get('frozen_selected_before_test')),flush=True)
if __name__=='__main__':main()
