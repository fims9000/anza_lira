#!/usr/bin/env python3
"""S66: CAL-selected image-only endpoint tracing and fresh end-to-end TEST.

No source-truth can enter detector/endpoint tracing/graph scoring. Synthetic
identities are inspected only after all three variants have made predictions.
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
from bubble_chamber.s56_global_path_assembly import pair_rows,scored_edges,select_global_forest
from bubble_chamber.s59_candidate_retrieval import filter_rows
from bubble_chamber.s63_image_curve_variants import predict_variants
from bubble_chamber.s64_pixel_to_path import audit_synthetic_scene
from bubble_chamber.s66_image_endpoint_tracing import ARMS,extend_image_curves,choose_calibration_only
from scripts.research.bubble_chamber.s37_benchmark_review_risk import split_source_frames
from scripts.research.bubble_chamber.s65_calibrate_endpoint_abstention import _fit_source

SCHEMA='s66_pixel_end_extension_CAL_then_NEW_TEST_SYNTHETIC_ONLY_v1'
PROTOCOL=ROOT/'docs/bubble_chamber/S66_FROZEN_IMAGE_ENDPOINT_TRACING_2026_10_26.md'
CODE=ROOT/'bubble_chamber/s66_image_endpoint_tracing.py'
CAL_SEEDS={'005':20266601,'006':20266611}
TEST_SEEDS={'005':(20266603,20266604),'006':(20266613,20266614)}
CAL_SCENES=32
TEST_SCENES=24


def image_only_inference_variants(gray,scorer,cutoff):
    """No simulator geometry argument; only original pixels and source-CAL scorer."""
    if gray.shape!=(192,192) or gray.dtype!=np.uint8:raise ValueError('Bad image shape')
    candidates=predict_variants(gray)['s39_original']
    outs={}
    for arm in ARMS:
        pieces=extend_image_curves(gray,candidates,arm)
        if len(pieces)<3:
            outs[arm]={'polylines':pieces,'chosen':[],'pair_count':0,
                       'solver_success':True,'physics_auto_joins_authorized':False}
            continue
        rows=filter_rows(pair_rows(pieces),'strict_s39')
        if rows:
            scores=scorer(rows)
            chosen,qa=select_global_forest(scored_edges(rows,scores),threshold=cutoff,margin=.04)
            if not qa['solver_success']:chosen=[]
        else:chosen=[];qa={'solver_success':True}
        outs[arm]={'polylines':pieces,'chosen':chosen,'pair_count':len(rows),
                   'solver_success':qa['solver_success'],
                   'physics_auto_joins_authorized':False}
    return outs


def evaluate(gray,synthetic_proxy,scorer,cutoff):
    inferred=image_only_inference_variants(gray,scorer,cutoff)
    return {arm:audit_synthetic_scene(out,synthetic_proxy) for arm,out in inferred.items()}


def sample(photos,templates,seed,n,scorer,cutoff):
    rng=np.random.default_rng(seed)
    data={arm:[] for arm in ARMS}
    for k in range(n):
        scene=make_three_piece_scene(photos[int(rng.integers(len(photos)))],templates,
                    seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
        measured=evaluate(scene.gray,scene.nodes,scorer,cutoff)
        for arm in ARMS:data[arm].append(measured[arm])
    return data


def totals(rows):
    keys=('candidate_polylines','candidate_pairs_after_strict_gate',
          'source_node_matches','synthetic_decoy_matched','selected_provisional_links',
          'selected_synthetic_correct_links','synthetic_noncorrect_selected',
          'selected_synthetic_wrong_endpoint','selected_synthetic_unmatched_or_ambiguous',
          'scene_any_noncorrect','complete_three_piece_paths_synthetic','solver_failure')
    return {**{name:int(sum(r[name] for r in rows)) for name in keys},'scene_count':len(rows)}


def paired(a,b,seed):
    if len(a)!=len(b) or not a:raise ValueError('Unpaired CAL/TEST')
    rng=np.random.default_rng(seed)
    sample_ids=rng.integers(0,len(a),size=(1000,len(a)))
    return {k:{'delta_mean_per_scene':float((delta:=np.asarray([y[k]-x[k]
                for x,y in zip(a,b)],float)).mean()),
               'paired_scene_bootstrap_95pct':[
                   float(v) for v in np.quantile(delta[sample_ids].mean(1),[.025,.975])]}
            for k in ('selected_synthetic_correct_links','synthetic_noncorrect_selected',
                      'source_node_matches','complete_three_piece_paths_synthetic')}


def _paths(tiles,photo):
    photos=sorted(tiles.glob(f'H50Z_030_01_{photo}_*.png'))
    if len(photos)!=(21 if photo=='005' else 18):raise ValueError('Incorrect genuine H50Z bank')
    return photos


def run(data,tiles,phase,calibration=None,smoke=False):
    if phase not in ('cal','test'):raise ValueError('Wrong phase')
    manifest=audited_large_manifest(data)
    train,cal,test,split_sha=split_source_frames(manifest)
    if (len(train),len(cal),len(test))!=(79,30,36):raise ValueError('S37 split changed')
    templates={name:extract_track_templates(manifest,chosen_frames=frames)
               for name,frames in (('train',train),('cal',cal),('test',test))}
    digest=lambda x:sha256(Path(x).read_bytes()).hexdigest()
    public={'schema':SCHEMA,'phase':phase,'smoke_NOT_confirmatory':smoke,
            'protocol_sha256':digest(PROTOCOL),'predictor_code_sha256':digest(CODE),
            'source_frame_split_sha256':split_sha,'native_physics_GT':0,
            'source_backgrounds_same_run':True,
            'visualmetro_070_not_native_H50Z_physics':True,
            'all_metric_labels_are_SYNTHETIC_ONLY':True,
            'physics_auto_joins_authorized':False,'directions':{}}
    frozen=None
    if phase=='test':
        if calibration is None:raise ValueError('Must load immutable CAL')
        frozen=json.loads(calibration.read_text())
        if frozen.get('phase')!='cal' or frozen.get('schema')!=SCHEMA or frozen.get('smoke_NOT_confirmatory'):
            raise ValueError('Not valid confirmatory S66 CAL')
        for key in ('protocol_sha256','predictor_code_sha256','source_frame_split_sha256'):
            if frozen[key]!=public[key]:raise ValueError('CAL integrity mismatch for '+key)
        public['frozen_CAL_sha256']=digest(calibration)
    for source,target in (('005','006'),('006','005')):
        scorer,cutoff=_fit_source(source,templates,tiles)
        row={'SOURCE_photo':source,'TEST_photo':target,'S56_source_CAL_cutoff':cutoff}
        if phase=='cal':
            res=sample(_paths(tiles,source),templates['cal'],CAL_SEEDS[source],
                       2 if smoke else CAL_SCENES,scorer,cutoff)
            cal_totals={arm:totals(lines) for arm,lines in res.items()}
            choose=choose_calibration_only({arm:{'synthetic_correct':d['selected_synthetic_correct_links'],
                    'synthetic_noncorrect':d['synthetic_noncorrect_selected'],
                    'matched_source_fragments':d['source_node_matches'],
                    'synthetic_decoy_matched':d['synthetic_decoy_matched'],
                    'scenes':d['scene_count']} for arm,d in cal_totals.items()})
            row['CAL_all_variants']=cal_totals
            row['CAL_only_decision']=choose
            print('S66 CAL',source,'selection',choose,'totals',cal_totals,flush=True)
        else:
            prev=frozen['directions'][f'{source}_to_{target}']
            if prev['S56_source_CAL_cutoff']!=cutoff:raise ValueError('S56 unchanged fit required')
            chosen=prev['CAL_only_decision']['CAL_only_selected']
            if chosen not in ARMS:raise ValueError('Bad frozen method')
            row['frozen_selection']=chosen
            row['seed_results']={}
            for seed in (TEST_SEEDS[source] if not smoke else (20266691+int(source=='006'),)):
                results=sample(_paths(tiles,target),templates['test'],seed,
                                2 if smoke else TEST_SCENES,scorer,cutoff)
                result={'arms':{arm:totals(d) for arm,d in results.items()},
                        'chosen_minus_original':paired(results['s39_original'],results[chosen],seed+111)}
                row['seed_results'][str(seed)]=result
                print('S66 TEST',source,'->',target,seed,
                      'baseline',result['arms']['s39_original']['selected_synthetic_correct_links'],
                      'selected',chosen,result['arms'][chosen]['selected_synthetic_correct_links'],flush=True)
        public['directions'][f'{source}_to_{target}']=row
    if phase=='test':
        combined={arm:{} for arm in ARMS}
        for arm in ARMS:
            keys=list(next(iter(next(iter(public['directions'].values()))['seed_results'].values()))['arms'][arm])
            combined[arm]={k:int(sum(y['arms'][arm][k] for d in public['directions'].values()
                               for y in d['seed_results'].values())) for k in keys}
        public['all_TEST_arms_aggregate']=combined
        public['native_physical_GT_claim_allowed']=False
        public['promotion_after_TEST_forbidden_without_new_independent_data']=True
    return public


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('cal','test'))
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--calibration',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Never overwrite frozen CAL/TEST')
    j=run(args.visualmetro,args.photo_tiles,args.phase,args.calibration,smoke=args.smoke)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
if __name__=='__main__':main()
