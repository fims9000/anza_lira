#!/usr/bin/env python3
"""S75C: retrospective *synthetic-only* error taxonomy on locked S75 TEST.

All original pixel-to-graph inferences reproduced exactly; never select a new
model using these TEST labels, never infer native H50Z particle identity.
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
from bubble_chamber.s75_local_ridge_rescue import ARMS,preprocess
from scripts.research.bubble_chamber import s68_benchmark_branch_preservation as s68
from scripts.research.bubble_chamber import s72_fresh_corrected_branch_benchmark as s72
from scripts.research.bubble_chamber import s75_calibrate_then_test_local_ridges as s75

SCHEMA='S75C_POSTHOC_ERROR_TAXONOMY_SAME_TEST_NOT_INDEPENDENT_V1'
ERROR_KEYS=('selected_synthetic_wrong_identified',
            'selected_synthetic_unmatched_or_ambiguous',
            'selected_synthetic_wrong_endpoint')
AUX_KEYS=('unmatched_polylines','ambiguous_polylines')
DETAIL_KEYS=(*ERROR_KEYS,*AUX_KEYS,'source_node_matches',
             'synthetic_decoy_matched','selected_synthetic_correct_links',
             'synthetic_noncorrect_selected','complete_three_piece_paths_synthetic')


def _sum(rows):return {k:int(sum(row[k] for row in rows)) for k in DETAIL_KEYS}


def run(visualmetro,tiles,s72_cal,s75_cal,s75_test):
    test=json.loads(s75_test.read_text(encoding='utf8'))
    cal=json.loads(s75_cal.read_text(encoding='utf8'))
    if test.get('phase')!='test' or cal.get('phase')!='cal':raise ValueError('Frozen CAL and TEST required')
    if test['frozen_CAL_SHA256']!=s75.digest(s75_cal) or \
       test['protocol_sha256']!=s75.digest(s75.PROTOCOL) or \
       test['operator_sha256']!=s75.digest(s75.OPERATOR):
        raise ValueError('Original TEST fingerprint differs')
    previous=s75.frozen_s72(s72_cal)
    templates,split_sha=s72._source_data(visualmetro)
    if test['S37_frame_split_sha256']!=split_sha:raise ValueError('Split drift')
    public={'schema':SCHEMA,'analysis_status':'retrospective_only_NOT_NEW_TEST',
            'S75_TEST_sha256':sha256(s75_test.read_bytes()).hexdigest(),
            'S75_CAL_sha256':sha256(s75_cal.read_bytes()).hexdigest(),
            'source_split_sha256':split_sha,'all_reference_labels_synthetic_only':True,
            'new_training_or_threshold_selection':False,'native_physics_GT_count':0,
            'native_particle_auto_joins_authorized':False,'directions':{}}
    for source in ('005','006'):
        target='006' if source=='005' else '005'
        direction=f'{source}_to_{target}'
        scorer,cutoff=s68._fit_source(tiles,source,templates)
        method=previous['directions'][direction]['CAL_selected']
        if test['directions'][direction]['frozen_S72_graph_method']!=method or \
           test['directions'][direction]['frozen_S56_cutoff']!=float(cutoff):
            raise ValueError('Frozen source inference changed')
        photos=sorted(tiles.glob(f'H50Z_030_01_{target}_*.png'))
        if len(photos)!=(21 if target=='005' else 18):raise ValueError('Photo bank size changed')
        table={cond:{arm:[] for arm in ARMS} for cond in s75.TEST_CONDITIONS}
        for seed in s75.TEST_SEEDS[source]:
            rng=np.random.default_rng(seed)
            per_seed={c:{a:[] for a in ARMS} for c in s75.TEST_CONDITIONS}
            for k in range(s75.N_TEST):
                image=photos[int(rng.integers(len(photos)))]
                scene=make_three_piece_scene(image,templates['test'],
                    seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
                for cond in s75.TEST_CONDITIONS:
                    pix=s75.condition_pixels(scene.gray,cond)
                    # All pixel-only predictions first, synthetic identities only in auditor.
                    preds={arm:s68.infer_pixel_only(preprocess(pix,arm),scorer,cutoff,method)
                           for arm in ARMS}
                    for arm,pred in preds.items():
                        audit=audit_orientation_invariant_synthetic(pred,scene.nodes)
                        if sum(audit[x] for x in ERROR_KEYS)!=audit['synthetic_noncorrect_selected']:
                            raise AssertionError('Incomplete synthetic error accounting')
                        per_seed[cond][arm].append(audit)
                        table[cond][arm].append(audit)
            for cond in s75.TEST_CONDITIONS:
                for arm in ARMS:
                    original=test['directions'][direction]['TEST_seeds'][str(seed)]['conditions'][cond]['arms'][arm]
                    actual=s68.sums([{key:row[key] for key in s68.METRICS} for row in per_seed[cond][arm]])
                    if actual!=original:
                        raise AssertionError('S75C error taxonomy must reproduce the exact TEST predictions')
            print('S75C exact S75 TEST replay',seed,flush=True)
        public['directions'][direction]={'original_S72_graph_method':method,
            'heldout_scenes':48,
            'conditions':{cond:{arm:_sum(rows) for arm,rows in methods.items()}
                          for cond,methods in table.items()}}
    public['aggregate_96_scenes']={cond:{arm:{k:sum(
         d['conditions'][cond][arm][k] for d in public['directions'].values())
         for k in DETAIL_KEYS} for arm in ARMS} for cond in s75.TEST_CONDITIONS}
    return public


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--s72-calibration',type=Path,required=True)
    p.add_argument('--s75-calibration',type=Path,required=True)
    p.add_argument('--s75-test',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    x=p.parse_args()
    if x.out.exists():raise FileExistsError('Immutable posthoc receipt exists')
    result=run(x.visualmetro,x.photo_tiles,x.s72_calibration,x.s75_calibration,x.s75_test)
    x.out.parent.mkdir(parents=True,exist_ok=True)
    x.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
if __name__=='__main__':main()
