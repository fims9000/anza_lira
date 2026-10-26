#!/usr/bin/env python3
"""S75B diagnostic: post-TEST paired CIs without new selection or model tuning.

Replays EXACT frozen S75 TEST scenes using the unchanged S75 processor/scorer
and asserts that every reproduced aggregate equals the recorded TEST JSON.
This is retrospective uncertainty description, NOT an independent experiment.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from bubble_chamber.s75_local_ridge_rescue import ARMS
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75

SCHEMA='S75_POSTHOC_PAIRED_SCENE_BOOTSTRAP_NOT_NEW_TEST_V1'


def run(source,visualmetro,tiles,s72_cal,s75_cal,s75_test):
    if source not in ('005','006'):raise ValueError('Unknown source photograph ID')
    frozen=json.loads(s75_test.read_text())
    cal=json.loads(s75_cal.read_text())
    if frozen['phase']!='test' or cal['phase']!='cal':raise ValueError('Frozen CAL and TEST required')
    if (frozen['frozen_CAL_SHA256'] != s75.digest(s75_cal) or
        frozen['protocol_sha256'] != s75.digest(s75.PROTOCOL) or
        frozen['operator_sha256'] != s75.digest(s75.OPERATOR)):
        raise ValueError('Original S75 TEST provenance mismatch')
    previous=s75.frozen_s72(s72_cal)
    templates,split_sha=s72._source_data(visualmetro)
    if split_sha != frozen['S37_frame_split_sha256']:
        raise ValueError('Changed S37 source split')
    target='006' if source=='005' else '005'
    row=frozen['directions'][f'{source}_to_{target}']
    scorer,cutoff=s68._fit_source(tiles,source,templates)
    method=previous['directions'][f'{source}_to_{target}']['CAL_selected']
    if row['frozen_S72_graph_method']!=method or float(cutoff)!=row['frozen_S56_cutoff']:
        raise ValueError('Original inference settings drift')
    photos=sorted(tiles.glob(f'H50Z_030_01_{target}_*.png'))
    if len(photos)!=(21 if target=='005' else 18):raise ValueError('Private photo bank missing')
    joined={kind:{arm:[] for arm in ARMS} for kind in s75.TEST_CONDITIONS}
    for seed in s75.TEST_SEEDS[source]:
        samples=s75.sample(photos,templates['test'],seed,s75.N_TEST,
                           scorer,cutoff,method,s75.TEST_CONDITIONS)
        for cond in s75.TEST_CONDITIONS:
            for arm in ARMS:
                actual=s75.summarize(samples[cond][arm])
                original=row['TEST_seeds'][str(seed)]['conditions'][cond]['arms'][arm]
                if actual != original:
                    raise AssertionError(f'Historic S75 TEST prediction changed: {source}/{seed}/{cond}/{arm}')
                joined[cond][arm].extend(samples[cond][arm])
        print('S75B posthoc replay SHA-check-compatible seed',seed,flush=True)
    return {'schema':SCHEMA,'source_photo':source,'heldout_target_photo':target,
            'all_results_synthetic_only':True,'original_S75_TEST_sha256':sha256(s75_test.read_bytes()).hexdigest(),
            'original_S75_CAL_sha256':sha256(s75_cal.read_bytes()).hexdigest(),
            'original_TEST_seeds_replayed':list(s75.TEST_SEEDS[source]),
            'no_test_set_reselection':True,'independent_new_test':False,
            'physical_native_ground_truth':0,'conditions':{
                cond:{arm:{'matched_48_scenes':len(joined[cond][arm]),
                      'arm_minus_identity_paired_scene_bootstrap':s68.paired(
                            joined[cond]['identity'],joined[cond][arm],20267591+idx*7+j)}
                      for j,arm in enumerate(ARMS[1:])}
                for idx,cond in enumerate(s75.TEST_CONDITIONS)}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True,choices=['005','006'])
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--s72-calibration',type=Path,required=True)
    p.add_argument('--s75-calibration',type=Path,required=True)
    p.add_argument('--s75-test',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Never overwrite posthoc receipt')
    result=run(a.source,a.visualmetro,a.photo_tiles,a.s72_calibration,a.s75_calibration,a.s75_test)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
if __name__=='__main__':main()
