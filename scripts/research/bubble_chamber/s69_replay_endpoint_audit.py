#!/usr/bin/env python3
"""S69: independent pixel-only inference then two synthetic-only audit rules.

CAL first, fresh TEST second, historical S68 TEST replay last. Outputs only
privacy-safe aggregates; no raw photo, source track, or frame IDs.
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

from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_simulated_gaps import extract_track_templates
from bubble_chamber.s56_three_piece_scenes import make_three_piece_scene,SCENARIOS
from bubble_chamber.s64_pixel_to_path import audit_synthetic_scene
from bubble_chamber.s69_orientation_invariant_audit import audit_orientation_invariant_synthetic
from scripts.research.bubble_chamber.s37_benchmark_review_risk import split_source_frames
from scripts.research.bubble_chamber.s68_benchmark_branch_preservation import (
    ARMS, CAL_SEEDS, TEST_SEEDS as S68_HISTORIC_SEEDS, TEST_SCENES,
    infer_pixel_only, _photo_subfolder, _fit_source, choose_calibration_only,
    sums, paired,
)

SCHEMA='S69_reversal_invariant_synthetic_reaudit_v1'
PROTOCOL=ROOT/'docs/bubble_chamber/S69_FROZEN_REVERSAL_INVARIANT_AUDIT_2026_10_26.md'
S68_PROTOCOL=ROOT/'docs/bubble_chamber/S68_FROZEN_BRANCH_PRESERVATION_PROTOCOL_2026_10_26.md'
PREDICTOR=ROOT/'bubble_chamber/s68_branch_preserving_instances.py'
EVALUATOR=ROOT/'bubble_chamber/s69_orientation_invariant_audit.py'
S56_SCORER=ROOT/'scripts/research/bubble_chamber/s56_benchmark_global_paths.py'
FRESH_SEEDS={'005':(20266903,20266904),'006':(20266913,20266914)}
CAL_SCENES=16


def digest(p):return sha256(Path(p).read_bytes()).hexdigest()


def sample(tiles,templates,seed,n,scorer,cutoff):
    photos=sorted(tiles.glob('H50Z_030_01_*_*.png'))
    if not photos:raise ValueError('Missing private H50Z photo tiles')
    rng=np.random.default_rng(seed)
    old={name:[] for name in ARMS};new={name:[] for name in ARMS}
    for k in range(n):
        photo=photos[int(rng.integers(len(photos)))]
        scene=make_three_piece_scene(photo,templates,
            seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
        # Entire inference pass finishes before any GT is read.
        predictions={name:infer_pixel_only(scene.gray,scorer,cutoff,name) for name in ARMS}
        for name,pred in predictions.items():
            historic=audit_synthetic_scene(pred,scene.nodes)
            corrected=audit_orientation_invariant_synthetic(pred,scene.nodes)
            for key in ('candidate_polylines','candidate_pairs_after_strict_gate',
                        'selected_provisional_links','source_node_matches',
                        'solver_failure','synthetic_decoy_matched'):
                if historic[key]!=corrected[key]:raise AssertionError('Evaluator modified inference')
            if corrected['selected_synthetic_correct_links']<historic['selected_synthetic_correct_links']:
                raise AssertionError('Orientation correction must not reject old correctly oriented endpoints')
            if corrected['selected_synthetic_correct_links']+corrected['synthetic_noncorrect_selected']!=corrected['selected_provisional_links']:
                raise AssertionError('Accounting mismatch')
            old[name].append(historic);new[name].append(corrected)
    return old,new


def totals(rows):
    if not rows:raise ValueError('Empty fixed scenario set')
    fields=('candidate_polylines','candidate_pairs_after_strict_gate',
            'source_node_matches','synthetic_decoy_matched',
            'selected_provisional_links','selected_synthetic_correct_links',
            'synthetic_noncorrect_selected','scene_any_noncorrect',
            'complete_three_piece_paths_synthetic','solver_failure',
            'selected_synthetic_wrong_identified','selected_synthetic_unmatched_or_ambiguous',
            'selected_synthetic_wrong_endpoint')
    return {**{key:int(sum(row[key] for row in rows)) for key in fields},
            'scenes':len(rows),'synthetic_truth_links':4*len(rows)}


def paired_delta(old,new,seed):
    rng=np.random.default_rng(seed)
    n=len(old)
    if n!=len(new) or n==0:raise ValueError('Paired scenes expected')
    ix=rng.integers(0,n,size=(1200,n))
    keys=('selected_synthetic_correct_links','selected_synthetic_wrong_endpoint',
          'synthetic_noncorrect_selected','scene_any_noncorrect',
          'complete_three_piece_paths_synthetic')
    d={}
    for key in keys:
        v=np.asarray([y[key]-x[key] for x,y in zip(old,new)],float)
        d[key]={'mean_delta_per_scene':float(v.mean()),
                'scene_bootstrap_95pct':np.quantile(v[ix].mean(axis=1),[.025,.975]).tolist()}
    return d


def run(visualmetro,tiles,phase,calibration=None):
    if phase not in ('cal','fresh','historic'):raise ValueError('Invalid S69 phase')
    manifest=audited_large_manifest(visualmetro)
    train,cal,test,split_sha=split_source_frames(manifest)
    if (len(train),len(cal),len(test))!=(79,30,36):raise ValueError('Frame split drift')
    templates={name:extract_track_templates(manifest,chosen_frames=frames)
               for name,frames in (('train',train),('cal',cal),('test',test))}
    photos={photo:list(tiles.glob(f'H50Z_030_01_{photo}_*.png')) for photo in ('005','006')}
    if len(photos['005'])!=21 or len(photos['006'])!=18:raise ValueError('Private photo crop population drift')
    public={
        'schema':SCHEMA,'phase':phase,'experiment_id':'ANZA-LIRA-BC-S69-20261026',
        'protocol_sha256':digest(PROTOCOL),'evaluator_sha256':digest(EVALUATOR),
        'S68_predictor_sha256':digest(PREDICTOR),'S68_protocol_sha256':digest(S68_PROTOCOL),
        'S56_scorer_sha256':digest(S56_SCORER),'source_split_sha256':split_sha,
        'whole_frame_split':{'train':79,'cal':30,'test':36},
        'same_run_H50Z_photographic_backgrounds':True,
        'evaluation_only_synthetic_labels':True,
        'native_H50Z_verified_GT_count':0,'physical_auto_joins_authorized':False,
        'per_scene_raw_coordinates_or_frame_IDs_exported':False,'experiments':{},
    }
    frozen=None
    if phase!='cal':
        if calibration is None:raise ValueError('Frozen corrected CAL file required')
        frozen=json.loads(Path(calibration).read_text(encoding='utf8'))
        for key in ('schema','protocol_sha256','evaluator_sha256','S68_predictor_sha256',
                    'S68_protocol_sha256','S56_scorer_sha256','source_split_sha256'):
            if frozen.get(key)!=public[key]:raise ValueError('Corrected CAL provenance mismatch: '+key)
        if frozen.get('phase')!='cal':raise ValueError('Require CAL not TEST artifact')
        public['corrected_calibration_sha256']=digest(calibration)
    for source in ('005','006'):
        target='006' if source=='005' else '005'
        scorer,cutoff=_fit_source(tiles,source,templates)
        direction=f'{source}_to_{target}'
        entry={'source_photo':source,'test_photo':target,'S56_original_cutoff':cutoff}
        if phase=='cal':
            old,new=sample(_photo_subfolder(tiles,source),templates['cal'],
                           CAL_SEEDS[source],CAL_SCENES,scorer,cutoff)
            aggregates={name:totals(new[name]) for name in ARMS}
            entry['cal_seed']=CAL_SEEDS[source]
            entry['corrected_cal']={name:totals(new[name]) for name in ARMS}
            entry['historic_cal']={name:totals(old[name]) for name in ARMS}
            # Adapt exact prior promotion rule to public sum schema.
            expected={name:{**{k:aggregates[name][k] for k in (
               'selected_synthetic_correct_links','synthetic_noncorrect_selected',
               'scene_any_noncorrect','complete_three_piece_paths_synthetic')}} for name in ARMS}
            entry['corrected_cal_choice']=choose_calibration_only(expected)
        else:
            old_entry=frozen['experiments'][direction]
            if old_entry['S56_original_cutoff']!=cutoff:raise ValueError('Source-only scorer changed')
            entry['chosen_from_corrected_CAL']=old_entry['corrected_cal_choice']['CAL_only_selected']
            seeds=FRESH_SEEDS[source] if phase=='fresh' else S68_HISTORIC_SEEDS[source]
            entry['test_seeds']={}
            for seed in seeds:
                old,new=sample(_photo_subfolder(tiles,target),templates['test'],
                               seed,TEST_SCENES,scorer,cutoff)
                entry['test_seeds'][str(seed)]={
                    'scene_count':TEST_SCENES,
                    'historic_orientation_sensitive_totals':{name:totals(old[name]) for name in ARMS},
                    'corrected_orientation_invariant_totals':{name:totals(new[name]) for name in ARMS},
                    'corrected_minus_old':{name:paired_delta(old[name],new[name],seed+69)
                                            for name in ARMS},
                    'S68_arm_minus_S39_corrected':{name:paired_delta(new[ARMS[0]],new[name],seed+139)
                                                 for name in ARMS[1:]},
                }
        public['experiments'][direction]=entry
    return public


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('cal','fresh','historic'))
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--calibration',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Never overwrite frozen S69 result')
    r=run(a.visualmetro,a.photo_tiles,a.phase,a.calibration)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('S69',a.phase,'saved',a.out,flush=True)
if __name__=='__main__':main()
