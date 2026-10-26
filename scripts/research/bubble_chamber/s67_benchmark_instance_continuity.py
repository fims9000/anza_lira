#!/usr/bin/env python3
"""S67 preregistered CAL and independent TEST, image-only crossing hypotheses.

No real H50Z label is assumed; all reported success/error identities are
programmatically injected synthetic truth used *after* pixel-only inference.
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
from bubble_chamber.s35_fragment_link import calibrated_negative_threshold
from bubble_chamber.s56_three_piece_scenes import SCENARIOS,make_three_piece_scene
from bubble_chamber.s56_global_path_assembly import pair_rows,scored_edges,select_global_forest
from bubble_chamber.s59_candidate_retrieval import filter_rows
from bubble_chamber.s39_machine_auto import extract_ridges,ridge_evidence
from bubble_chamber.s67_instance_continuity import extract_instance_hypotheses,METHOD
from bubble_chamber.s64_pixel_to_path import audit_synthetic_scene
from scripts.research.bubble_chamber.s37_benchmark_review_risk import split_source_frames
from scripts.research.bubble_chamber.s56_benchmark_global_paths import (
    generate,flatten,fit,PROTOCOL as S56_PROTOCOL)

SCHEMA='S67_image_only_instance_aware_CAL_then_fresh_TEST_SYNTHETIC_ONLY_v1'
PROTOCOL=ROOT/'docs/bubble_chamber/S67_FROZEN_CROSSING_INSTANCE_PROTOCOL_2026_10_26.md'
PREDICTOR=ROOT/'bubble_chamber/s67_instance_continuity.py'
S56_MODEL=ROOT/'scripts/research/bubble_chamber/s56_benchmark_global_paths.py'
ARMS=('s39_original','s67_crossing_orientation_v1')
CAL_SEEDS={'005':20266701,'006':20266711}
TEST_SEEDS={'005':(20266703,20266704),'006':(20266713,20266714)}
CAL_SCENES=16
TEST_SCENES=24
BOOTSTRAPS=1200
METRICS=('candidate_polylines','candidate_pairs_after_strict_gate',
         'source_node_matches','synthetic_decoy_matched',
         'selected_provisional_links','selected_synthetic_correct_links',
         'synthetic_noncorrect_selected','scene_any_noncorrect',
         'complete_three_piece_paths_synthetic','solver_failure')


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()


def choose_calibration_only(aggregates):
    if set(aggregates)!=set(ARMS):raise ValueError('Missing frozen arms')
    for row in aggregates.values():
        if any(name not in row or not isinstance(row[name],int) or row[name]<0
               for name in ('selected_synthetic_correct_links','synthetic_noncorrect_selected',
                            'scene_any_noncorrect')):
            raise ValueError('Bad CAL aggregate')
    b=aggregates[ARMS[0]];n=aggregates[ARMS[1]]
    eligible=(n['selected_synthetic_correct_links']>=b['selected_synthetic_correct_links']+2
        and n['synthetic_noncorrect_selected']<=b['synthetic_noncorrect_selected']
        and n['scene_any_noncorrect']<=b['scene_any_noncorrect'])
    return {'CAL_only_selected':ARMS[1] if eligible else ARMS[0],
            'S67_CAL_eligible':bool(eligible),
            'future_native_H50Z_physical_auto_join_authorized':False}


def infer_pixel_only(gray,scorer,cutoff,arm):
    """Forbid passing generator geometry or truth to inference."""
    if arm not in ARMS:raise ValueError('Invalid frozen arm')
    if gray.shape!=(192,192) or gray.dtype!=np.uint8:
        raise ValueError('Invalid pixel-only image')
    if not np.isfinite(cutoff) or not 0<=cutoff<=1:raise ValueError('Invalid CAL cutoff')
    if arm==ARMS[0]:
        pieces=extract_ridges(ridge_evidence(gray),max_candidates=32)
        extra={'pixel_junctions':None,'provisional_continuity_pairs':0}
    else:
        proposal=extract_instance_hypotheses(gray)
        pieces=proposal['polylines']
        extra={k:proposal[k] for k in ('pixel_junctions','provisional_continuity_pairs')}
    if len(pieces)<3:
        return {'polylines':pieces,'pair_count':0,'chosen':[],
                'solver_success':True,'physics_auto_joins_authorized':False,**extra}
    rows=filter_rows(pair_rows(pieces),'strict_s39')
    if not rows:
        return {'polylines':pieces,'pair_count':0,'chosen':[],
                'solver_success':True,'physics_auto_joins_authorized':False,**extra}
    scores=np.asarray(scorer(rows),float)
    edges=scored_edges(rows,scores)
    chosen,qa=select_global_forest(edges,threshold=float(cutoff),margin=.04)
    if not qa['solver_success']:chosen=[]
    return {'polylines':pieces,'pair_count':len(rows),'chosen':chosen,
            'solver_success':bool(qa['solver_success']),
            'physics_auto_joins_authorized':False,**extra}


def sums(rows):
    if not rows:raise ValueError('No scenes')
    return {**{k:int(sum(x[k] for x in rows)) for k in METRICS},'scene_count':len(rows)}


def paired(a,b,seed):
    if len(a)!=len(b) or not a:raise ValueError('Unpaired scenes')
    rng=np.random.default_rng(seed)
    indexes=rng.integers(0,len(a),size=(BOOTSTRAPS,len(a)))
    names=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
           'scene_any_noncorrect','complete_three_piece_paths_synthetic')
    result={}
    for k in names:
        v=np.asarray([y[k]-x[k] for x,y in zip(a,b)],float)
        result[k]={'mean_delta_per_scene':float(v.mean()),
                   'paired_scene_bootstrap_95pct':
                   [float(q) for q in np.quantile(v[indexes].mean(axis=1),(.025,.975))]}
    return result


def sample(tiles,templates,seed,n,scorer,cutoff):
    photos=sorted(tiles.glob('H50Z_030_01_*_*.png'))
    if not photos:raise ValueError('No authenticated private photo crops')
    rng=np.random.default_rng(seed)
    metrics={a:[] for a in ARMS}
    for k in range(n):
        photo=photos[int(rng.integers(len(photos)))]
        scene=make_three_piece_scene(photo,templates,
            seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
        # Important: both arms predict BEFORE any generator truth is queried.
        predictions={a:infer_pixel_only(scene.gray,scorer,cutoff,a) for a in ARMS}
        for a,pred in predictions.items():
            m=audit_synthetic_scene(pred,scene.nodes)
            metrics[a].append({k:m[k] for k in METRICS})
    return metrics


def _fit_source(tiles,source,templates):
    train=generate(tiles,source,templates['train'],n=S56_PROTOCOL['train_scenes'],
                   seed=S56_PROTOCOL['train_seeds'][source])
    cal=generate(tiles,source,templates['cal'],n=S56_PROTOCOL['cal_scenes'],
                   seed=S56_PROTOCOL['cal_seeds'][source])
    x,y=flatten(train);u,v=flatten(cal)
    scorer=fit(x,y)
    return scorer,float(calibrated_negative_threshold(scorer(u),v,max_fpr=.05))


def run(visualmetro,tiles,phase,calibration=None,smoke=False):
    if phase not in ('cal','test'):raise ValueError('Wrong phase')
    manifest=audited_large_manifest(visualmetro)
    train,cal,test,split_sha=split_source_frames(manifest)
    if (len(train),len(cal),len(test))!=(79,30,36):raise ValueError('Frozen frame split changed')
    templates={x:extract_track_templates(manifest,chosen_frames=frames)
               for x,frames in (('train',train),('cal',cal),('test',test))}
    photos={p:sorted(tiles.glob(f'H50Z_030_01_{p}_*.png')) for p in ('005','006')}
    if len(photos['005'])!=21 or len(photos['006'])!=18:raise ValueError('Private source crop count changed')
    public={'schema':SCHEMA,'phase':phase,'smoke_NOT_confirmatory':bool(smoke),
            'protocol_sha256':digest(PROTOCOL),'predictor_sha256':digest(PREDICTOR),
            'S56_scorer_sha256':digest(S56_MODEL),'split_sha256':split_sha,
            'whole_frame_split':{'train':79,'cal':30,'test':36},
            'all_metrics_synthetic_injections_only':True,
            'source_photos_both_from_one_physical_run':True,
            'native_H50Z_verified_GT_count':0,'physical_auto_joins_authorized':False,
            'experiments':{}}
    frozen=None
    if phase=='test':
        if calibration is None:raise ValueError('Provide frozen CAL JSON')
        frozen=json.loads(Path(calibration).read_text(encoding='utf8'))
        for key in ('schema','protocol_sha256','predictor_sha256','S56_scorer_sha256','split_sha256'):
            if public[key]!=frozen[key]:raise ValueError('CAL lock mismatch: '+key)
        if frozen.get('phase')!='cal' or frozen.get('smoke_NOT_confirmatory'):
            raise ValueError('CAL must be a complete frozen non-smoke run')
        public['frozen_calibration_sha256']=digest(calibration)
    for source in ('005','006'):
        target='006' if source=='005' else '005'
        scorer,cutoff=_fit_source(tiles,source,templates)
        entry={'source_photo':source,'test_photo':target,'frozen_S56_cutoff':cutoff}
        if phase=='cal':
            rows=sample(tiles if smoke else _photo_subfolder(tiles,source),
                        templates['cal'],CAL_SEEDS[source],2 if smoke else CAL_SCENES,scorer,cutoff)
            totals={a:sums(rows[a]) for a in ARMS}
            entry['cal_totals']=totals
            entry['cal_choice']=choose_calibration_only(totals)
        else:
            old=frozen['experiments'][f'{source}_to_{target}']
            if old['frozen_S56_cutoff']!=cutoff:raise ValueError('S56 scorer cutoff changed')
            selected=old['cal_choice']['CAL_only_selected']
            entry['cal_selected_arm']=selected
            entry['test_seeds']={}
            for seed in (TEST_SEEDS[source] if not smoke else TEST_SEEDS[source][:1]):
                rows=sample(_photo_subfolder(tiles,target),templates['test'],seed,
                            2 if smoke else TEST_SCENES,scorer,cutoff)
                entry['test_seeds'][str(seed)]={'totals':{a:sums(rows[a]) for a in ARMS},
                     's67_minus_s39':paired(rows[ARMS[0]],rows[ARMS[1]],seed+67)}
        public['experiments'][f'{source}_to_{target}']=entry
    return public


def _photo_subfolder(tiles,photo):
    # The source never depends on labels: filter names only for domain split.
    from tempfile import TemporaryDirectory
    # Instead of duplicating private images or making symlinks, use a path-like
    # glob filter restricted to the filename photo ID; sample() accepts .glob().
    class _PhotoBank:
        def glob(self,pattern):
            return sorted(tiles.glob(f'H50Z_030_01_{photo}_*.png'))
    return _PhotoBank()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['cal','test'])
    parser.add_argument('--visualmetro',type=Path,required=True)
    parser.add_argument('--photo-tiles',type=Path,required=True)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Never overwrite immutable experiments')
    j=run(args.visualmetro,args.photo_tiles,args.phase,args.calibration,args.smoke)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(j,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    print('S67',args.phase,'experiments',list(j['experiments']),'saved',args.out,flush=True)
if __name__=='__main__':main()
