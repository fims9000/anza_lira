#!/usr/bin/env python3
"""S65 source-CAL-only endpoint guard -> independent fresh synthetic TEST.

Select on CAL only, store immutable decision. On TEST check the signed hash
of the saved CAL JSON, split fingerprint and original S56 calibration cutoff.
No real particle labels, no physical authorization or native image outputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np

from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_fragment_link import calibrated_negative_threshold
from bubble_chamber.s35_simulated_gaps import extract_track_templates
from bubble_chamber.s56_three_piece_scenes import SCENARIOS,make_three_piece_scene
from bubble_chamber.s64_pixel_to_path import audit_synthetic_scene,infer_from_pixels
from bubble_chamber.s65_endpoint_guard import THRESHOLD_GRID,retain_by_endpoint_guard,choose_cal_only
from scripts.research.bubble_chamber.s37_benchmark_review_risk import split_source_frames
from scripts.research.bubble_chamber.s56_benchmark_global_paths import (
    generate,flatten,fit,PROTOCOL as S56_PROTOCOL)

SCHEMA='s65_CAL_pixel_endpoint_guard_then_FRESH_TEST_SYNTHETIC_ONLY_v1'
PROTOCOL=ROOT/'docs/bubble_chamber/S65_FROZEN_PIXEL_ENDPOINT_ABSTENTION_2026_10_26.md'
CAL_SEEDS={'005':20266501,'006':20266511}
TEST_SEEDS={'005':(20266503,20266504),'006':(20266513,20266514)}
CAL_SCENES=32
TEST_SCENES=24


def _lock_calibration(path):
    payload=path.read_bytes()
    doc=json.loads(payload)
    if doc.get('schema')!=SCHEMA or doc.get('phase')!='cal':
        raise ValueError('Not an authentic S65 CAL decision')
    return doc,sha256(payload).hexdigest()


def _fit_source(source,templates,tiles):
    scenes=generate(tiles,source,templates['train'],n=S56_PROTOCOL['train_scenes'],
                    seed=S56_PROTOCOL['train_seeds'][source])
    calibrate=generate(tiles,source,templates['cal'],n=S56_PROTOCOL['cal_scenes'],
                       seed=S56_PROTOCOL['cal_seeds'][source])
    r,y=flatten(scenes);q,z=flatten(calibrate)
    scorer=fit(r,y)
    cutoff=calibrated_negative_threshold(scorer(q),z,max_fpr=.05)
    return scorer,float(cutoff)


def _evaluate_one(gray,ref,scorer,cutoff,thresholds):
    # First, inference from pixels only. Artificial source curves are passed
    # solely to post-hoc scoring of frozen CAL or TEST outputs.
    inferred=infer_from_pixels(gray,scorer,cutoff,arm='s39_original')
    measured={}
    for threshold in thresholds:
        chosen=retain_by_endpoint_guard(gray,inferred['polylines'],inferred['chosen'],threshold)
        assert set(chosen)<=set(inferred['chosen'])
        result=audit_synthetic_scene({**inferred,'chosen':chosen},ref)
        result['abstained_selected_joins']=len(inferred['chosen'])-len(chosen)
        measured[threshold]=result
    return measured


def _sample(source,templates,tiles,seed,n,scorer,cutoff,thresholds):
    photos=sorted(tiles.glob(f'H50Z_030_01_{source}_*.png'))
    if len(photos)!=(21 if source=='005' else 18):
        raise ValueError('Missing genuine H50Z 512x512 crop bank')
    rng=np.random.default_rng(seed)
    out={th:[] for th in thresholds}
    for k in range(n):
        scene=make_three_piece_scene(photos[int(rng.integers(len(photos)))],templates,
                    seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
        measures=_evaluate_one(scene.gray,scene.nodes,scorer,cutoff,thresholds)
        for threshold,m in measures.items():out[threshold].append(m)
    return out


def _aggregate(rows):
    fields=('selected_provisional_links','selected_synthetic_correct_links',
            'synthetic_noncorrect_selected','scene_any_noncorrect',
            'complete_three_piece_paths_synthetic','abstained_selected_joins',
            'selected_synthetic_wrong_endpoint','selected_synthetic_wrong_identified',
            'selected_synthetic_unmatched_or_ambiguous')
    return {**{k:int(sum(x[k] for x in rows)) for k in fields},'n_synthetic_scenes':len(rows)}


def _paired(a,b,seed):
    assert len(a)==len(b) and a
    rng=np.random.default_rng(seed)
    picks=rng.integers(0,len(a),size=(1200,len(a)))
    keys=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
          'scene_any_noncorrect','complete_three_piece_paths_synthetic')
    return {k:{'mean_delta_per_scene':float((x:=np.asarray([q[k]-p[k]
                  for p,q in zip(a,b)],float)).mean()),
                'paired_scene_bootstrap_95pct':[float(v) for v in np.quantile(x[picks].mean(1),(.025,.975))]}
            for k in keys}


def run(data:Path,tiles:Path,phase:str,calibration:Path|None=None,*,smoke=False):
    if phase not in ('cal','test'):raise ValueError('CAL or TEST only')
    if phase=='test' and calibration is None:raise ValueError('TEST needs frozen CAL JSON')
    manifest=audited_large_manifest(data)
    train,cal,test,split_sha=split_source_frames(manifest)
    if (len(train),len(cal),len(test))!=(79,30,36):
        raise RuntimeError('S37 whole-frame split no longer matches original')
    templates={name:extract_track_templates(manifest,chosen_frames=frames)
               for name,frames in [('train',train),('cal',cal),('test',test)]}
    report={'schema':SCHEMA,'phase':phase,'smoke_NOT_CONFIRMATORY':smoke,
            'protocol_sha256':sha256(PROTOCOL.read_bytes()).hexdigest(),
            'source_frame_split_sha256':split_sha,
            'native_physics_GT_count':0,'physical_auto_joins_authorized':False,
            'all_metrics_are_SYNTHETIC_ONLY':True,'directions':{}}
    saved=None
    if phase=='test':
        saved,receipt=_lock_calibration(calibration)
        if saved['protocol_sha256']!=report['protocol_sha256'] or saved['source_frame_split_sha256']!=split_sha:
            raise ValueError('Protocol or frame split not equal to immutable CAL')
        if saved['smoke_NOT_CONFIRMATORY']:
            raise ValueError('Never use smoke calibration for confirmatory TEST')
        report['frozen_CAL_bytes_sha256']=receipt
    for source,target in (('005','006'),('006','005')):
        scorer,cutoff=_fit_source(source,templates,tiles)
        report_row={'source_training_photo':source,'test_photo':target,
                    'S56_CAL_only_cutoff':cutoff,'results':{}}
        if phase=='cal':
            cases=_sample(source,templates['cal'],tiles,CAL_SEEDS[source],
                          2 if smoke else CAL_SCENES,scorer,cutoff,[None,*THRESHOLD_GRID])
            totals={th:_aggregate(rows) for th,rows in cases.items()}
            decision=choose_cal_only({th:{'correct':v['selected_synthetic_correct_links'],
                                               'noncorrect':v['synthetic_noncorrect_selected'],
                                               'scenes':v['n_synthetic_scenes']}
                                       for th,v in totals.items()})
            report_row['CAL_decision']=decision
            report_row['frozen_CAL_grid_aggregates']={
                'unfiltered' if th is None else str(th):v for th,v in totals.items()}
            print('S65 CAL',source,'threshold',cutoff,'original',totals[None],
                  'chosen',decision['selected_CAL_only'],flush=True)
        else:
            decision=saved['directions'][f'{source}_to_{target}']['CAL_decision']
            if saved['directions'][f'{source}_to_{target}']['S56_CAL_only_cutoff']!=cutoff:
                raise ValueError('Frozen S56 scorer cutoff changed')
            selected=decision['selected_CAL_only']
            if selected is not None and selected not in THRESHOLD_GRID:
                raise ValueError('Unknown selected CAL threshold')
            report_row['frozen_CAL_decision']=decision
            report_row['TEST_seeds']={}
            for seed in (TEST_SEEDS[source] if not smoke else (20266591+int(source=='006'),)):
                cases=_sample(target,templates['test'],tiles,seed,
                              2 if smoke else TEST_SCENES,scorer,cutoff,
                              [None] if selected is None else [None,selected])
                original=cases[None];new=cases[selected]
                report_row['TEST_seeds'][str(seed)]={
                    'unfiltered':_aggregate(original),
                    'CAL_selected':_aggregate(new),
                    'CAL_selected_minus_unfiltered':_paired(original,new,seed+919)}
                print('S65 TEST',source,'->',target,'seed',seed,
                      'threshold',selected,'baseline',_aggregate(original),
                      'selected',_aggregate(new),flush=True)
        report['directions'][f'{source}_to_{target}']=report_row
    if phase=='test':
        all_orig=[s['unfiltered'] for d in report['directions'].values() for s in d['TEST_seeds'].values()]
        all_new=[s['CAL_selected'] for d in report['directions'].values() for s in d['TEST_seeds'].values()]
        keys=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
              'scene_any_noncorrect','complete_three_piece_paths_synthetic',
              'selected_provisional_links','abstained_selected_joins')
        report['aggregate_fresh_TEST']={k:{'unfiltered_total':int(sum(v[k] for v in all_orig)),
            'CAL_selected_total':int(sum(v[k] for v in all_new))} for k in keys}
        report['gate']={'safe_physics_claim_authorized':False,
                        'CAL_selected_reduces_noncorr_TEST':bool(sum(v['synthetic_noncorrect_selected'] for v in all_new)<
                                  sum(v['synthetic_noncorrect_selected'] for v in all_orig)),
                        'do_not_automatically_promote_after_TEST':True}
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['cal','test'])
    ap.add_argument('--visualmetro',type=Path,required=True)
    ap.add_argument('--photo-tiles',type=Path,required=True)
    ap.add_argument('--calibration',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Immutable CAL/TEST artefact already exists')
    report=run(args.visualmetro,args.photo_tiles,args.phase,args.calibration,smoke=args.smoke)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
if __name__=='__main__':main()
