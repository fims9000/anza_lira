#!/usr/bin/env python3
"""S64 held-out pixel-only full path evaluation, all metrics SYNTHETIC ONLY.

Pixel inference never receives simulator source nodes; ground truth enters only
separate post-hoc evaluation and optimistic upper bound. Source geometry
classifier is the original frozen S56 logistic/threshold fit on TRAIN/CAL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2
import numpy as np

from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_simulated_gaps import extract_track_templates
from bubble_chamber.s35_fragment_link import calibrated_negative_threshold
from bubble_chamber.s56_three_piece_scenes import SCENARIOS,TRUE_EDGES,make_three_piece_scene
from bubble_chamber.s56_global_path_assembly import pair_rows,scored_edges,select_global_forest
from bubble_chamber.s62_image_candidate_diagnostics import (
    proxy_extracted_piece_support,scene_upper_bound_from_proxy_support)
from bubble_chamber.s64_pixel_to_path import ARMS,audit_synthetic_scene,infer_from_pixels
from scripts.research.bubble_chamber.s37_benchmark_review_risk import split_source_frames
from scripts.research.bubble_chamber.s56_benchmark_global_paths import (
    generate,flatten,fit,PROTOCOL as S56_PROTOCOL)

SCHEMA='s64_pixel_to_polyline_to_graph_SYNTHETIC_ONLY_v1'
PROTOCOL=ROOT/'docs/bubble_chamber/S64_FROZEN_PIXEL_TO_PATH_PROTOCOL_2026_10_26.md'
SEEDS={'005':(20266403,20266404),'006':(20266413,20266414)}
SCENES_PER_SEED=24
METRICS=('candidate_polylines','candidate_pairs_after_strict_gate','source_node_matches',
         'synthetic_decoy_matched','selected_provisional_links','selected_synthetic_correct_links',
         'selected_synthetic_wrong_identified','selected_synthetic_unmatched_or_ambiguous',
         'selected_synthetic_wrong_endpoint','synthetic_noncorrect_selected',
         'scene_any_noncorrect','complete_three_piece_paths_synthetic','solver_failure',
         'union_proxy_fragments','union_proxy_links_upper','union_proxy_paths_upper')


def source_geometry_only_oracle(scene,scorer,threshold):
    """Diagnostic ONLY: forbidden seven generator nodes; never used by pixel inference."""
    rows=pair_rows(scene.nodes)
    edges=scored_edges(rows,scorer(rows))
    selected,qa=select_global_forest(edges,threshold=threshold,
                                    margin=S56_PROTOCOL['global_objective_stability_margin'])
    if not qa['solver_success']:
        selected=[]
    got={(e.i,e.j) for e in selected}
    truth=set(TRUE_EDGES)
    wrong_edges=got-truth
    full=sum(int(all(edge in got for edge in required) and
             not any((set(edge)&set(nodes)) for edge in wrong_edges))
             for nodes,required in (((0,1,2),((0,1),(1,2))),
                                     ((3,4,5),((3,4),(4,5)))))
    return {'selected_provisional_links':len(got),
            'selected_synthetic_correct_links':len(got&truth),
            'synthetic_noncorrect_selected':len(got-truth),
            'scene_any_noncorrect':int(bool(got-truth)),
            'complete_three_piece_paths_synthetic':full,
            'solver_failure':int(not qa['solver_success'])}


def evaluate_pixel_arms(gray,reference,scorer,threshold):
    """Important order: infer all arms FIRST, only then inspect reference."""
    inferred={arm:infer_from_pixels(gray,scorer,threshold,arm=arm) for arm in ARMS}
    result={}
    for arm,pred in inferred.items():
        x=audit_synthetic_scene(pred,reference)
        # S62 UNION upper bound is NOT an identity association; report separately.
        proxy=[proxy_extracted_piece_support(pred['polylines'],p) for p in reference]
        upper=scene_upper_bound_from_proxy_support(proxy)
        x.update(union_proxy_fragments=upper['proxy_true_fragments_usable'],
                 union_proxy_links_upper=upper['proxy_true_joins_upper_bound'],
                 union_proxy_paths_upper=upper['proxy_complete_three_piece_paths_upper_bound'])
        # Instance matching and union support are different measurements; do not
        # artificially force one to be an upper bound for the other.
        result[arm]=x
    return result,inferred


def _stats(rows,seed):
    if not rows:raise ValueError('Empty TEST scenes')
    out={};n=len(rows)
    rng=np.random.default_rng(seed)
    draw=rng.integers(0,n,size=(1200,n))
    for key in METRICS:
        a=np.asarray([r[key] for r in rows],float)
        out[key]={'total':int(a.sum()),'per_scene':float(a.mean()),
                  'scene_bootstrap_95pct':[float(q) for q in np.quantile(a[draw].mean(axis=1),[.025,.975])]}
    out['scene_count']=n
    out['native_verified_particle_tracks']=0
    out['synthetic_true_joins_total']=4*n
    out['synthetic_three_piece_tracks_total']=2*n
    return out


def _paired(rows_a,rows_b,seed):
    assert len(rows_a)==len(rows_b) and rows_a
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(rows_a),size=(1200,len(rows_a)))
    return {key:{'mean_delta_per_scene':float((b:=np.asarray([z[key] for z in rows_b],float)-
                np.asarray([z[key] for z in rows_a],float)).mean()),
                 'paired_scene_bootstrap_95pct':[float(q) for q in np.quantile(b[ix].mean(1),[.025,.975])]}
            for key in ('selected_synthetic_correct_links','synthetic_noncorrect_selected',
                        'scene_any_noncorrect','complete_three_piece_paths_synthetic')}


def _render_private(scene,inferred,path):
    image=cv2.cvtColor(scene.gray,cv2.COLOR_GRAY2BGR)
    layers=[image.copy(),image.copy()]
    for k,arm in enumerate(ARMS):
        pred=inferred[arm]
        for poly in pred['polylines']:
            cv2.polylines(layers[k],[np.rint(poly).astype('int32')],False,(0,200,200),1,cv2.LINE_AA)
        for e in pred['chosen']:
            a,b=pred['polylines'][e.i][e.end_i],pred['polylines'][e.j][e.end_j]
            pts=np.linspace(a,b,9)
            for idx in (0,2,4,6):
                cv2.line(layers[k],tuple(np.rint(pts[idx]).astype(int)),
                         tuple(np.rint(pts[idx+1]).astype(int)),(220,40,225),1,cv2.LINE_AA)
        cv2.putText(layers[k],arm[:17],(5,18),cv2.FONT_HERSHEY_SIMPLEX,.39,(255,255,255),1)
    joined=cv2.resize(np.hstack(layers),(1152,576),interpolation=cv2.INTER_NEAREST)
    cv2.putText(joined,'ARTIFICIAL INSERTIONS, NATIVE PHYSICS UNKNOWN',(5,555),
                cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    path.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(path),joined)


def run(visualmetro:Path,tiles:Path,*,smoke=False,private_dir=None):
    m=audited_large_manifest(visualmetro)
    train,cal,test,split_hash=split_source_frames(m)
    if (len(train),len(cal),len(test))!=(79,30,36):
        raise RuntimeError('S37 frozen frame split changed')
    templates={k:extract_track_templates(m,chosen_frames=v) for k,v in
               (('train',train),('cal',cal),('test',test))}
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    public={'schema':SCHEMA,'protocol_sha256':sha(PROTOCOL),
            'source_frame_split_sha256':split_hash,
            'S56_reference_scorer_module_sha256':sha(ROOT/'scripts/research/bubble_chamber/s56_benchmark_global_paths.py'),
            's63_extractor_module_sha256':sha(ROOT/'bubble_chamber/s63_image_curve_variants.py'),
            'source_complete_visualmetro_frames':len(m['frames']),
            'whole_frame_split_counts':{'train':79,'cal':30,'test':36},
            'source_photo_banks_one_physical_run':True,
            'source_visualmetro_070_not_matched_to_H50Z_photos':True,
            'inference_never_receives_generator_polylines':True,
            'S56_original_AZConv_UNet_code_unchanged':True,
            'native_H50Z_physics_GT_count':0,
            'smoke_NOT_confirmatory':bool(smoke),
            'synthetic_only_metrics':True,'directions':{}}
    ntrain=ncal=8 if smoke else S56_PROTOCOL['train_scenes']
    nscene=2 if smoke else SCENES_PER_SEED
    for source,target in (('005','006'),('006','005')):
        src_train=generate(tiles,source,templates['train'],n=ntrain,
                           seed=S56_PROTOCOL['train_seeds'][source])
        src_cal=generate(tiles,source,templates['cal'],n=ncal,
                         seed=S56_PROTOCOL['cal_seeds'][source])
        if any(not set(c['_PRIVATE_source_frames'])<=train for c in src_train):
            raise AssertionError('TRAIN source leak')
        if any(not set(c['_PRIVATE_source_frames'])<=cal for c in src_cal):
            raise AssertionError('CAL source leak')
        a,ay=flatten(src_train);b,by=flatten(src_cal)
        scorer=fit(a,ay)
        threshold=calibrated_negative_threshold(scorer(b),by,max_fpr=.05)
        photos=sorted(tiles.glob(f'H50Z_030_01_{target}_*.png'))
        if len(photos)!=(21 if target=='005' else 18):
            raise ValueError('Missing expected private H50Z crop bank')
        direction={'source_training_photo':source,'heldout_photo':target,
                   'source_curve_scorer':'original_S56_geometry_logistic',
                   'threshold_from_SOURCE_CAL_negative_FPR_only':float(threshold),
                   'test_seeds':{},'safety_gate_test_only_not_promoted_to_physical':False}
        for seed in (SEEDS[source] if not smoke else (20266491+int(source=='006'),)):
            rng=np.random.default_rng(seed)
            scenes={arm:[] for arm in ARMS}
            oracle=[]
            for k in range(nscene):
                photo=photos[int(rng.integers(len(photos)))]
                sc=make_three_piece_scene(photo,templates['test'],
                    seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[k%len(SCENARIOS)])
                if not set(sc.private_source_frames)<=test:raise AssertionError('TEST source leak')
                # Critical: pixel graph predictions generated before using source curves.
                metrics,predictions=evaluate_pixel_arms(sc.gray,sc.nodes,scorer,threshold)
                for arm in ARMS:scenes[arm].append(metrics[arm])
                # The explicitly disclosed seven-source-piece geometry upper diagnostic.
                oracle.append(source_geometry_only_oracle(sc,scorer,threshold))
                if private_dir and k==0:
                    _render_private(sc,predictions,
                        private_dir/f'S64_SYNTHETIC_PIXEL_TO_GRAPH_{source}_to_{target}_{seed}.png')
            results={arm:_stats(rows,seed+q*11+900) for q,(arm,rows) in enumerate(scenes.items())}
            results['oracle_geometry_only']={
                key:int(sum(x[key] for x in oracle)) for key in oracle[0]}
            results['junction_less_minus_original']=_paired(scenes['s39_original'],
                                                             scenes['junction_less'],seed+555)
            direction['test_seeds'][str(seed)]=results
            print('S64',source,'->',target,seed,'scenes',nscene,
                  'image selected',[(a,results[a]['selected_provisional_links']['total'],
                      results[a]['selected_synthetic_correct_links']['total'],
                      results[a]['synthetic_noncorrect_selected']['total']) for a in ARMS],
                  'oracle',results['oracle_geometry_only']['selected_synthetic_correct_links'],flush=True)
        public['directions'][f'{source}_to_{target}']=direction
    # Explicit global predeclared NO-PROMOTION if false proposals go up.
    orig=sum(v['selected_synthetic_correct_links']['total'] for d in public['directions'].values()
             for v in (z['s39_original'] for z in d['test_seeds'].values()))
    new=sum(v['selected_synthetic_correct_links']['total'] for d in public['directions'].values()
             for v in (z['junction_less'] for z in d['test_seeds'].values()))
    old_err=sum(z['s39_original']['synthetic_noncorrect_selected']['total'] for d in public['directions'].values()
                for z in d['test_seeds'].values())
    new_err=sum(z['junction_less']['synthetic_noncorrect_selected']['total'] for d in public['directions'].values()
                for z in d['test_seeds'].values())
    old_path=sum(z['s39_original']['complete_three_piece_paths_synthetic']['total'] for d in public['directions'].values()
                 for z in d['test_seeds'].values())
    new_path=sum(z['junction_less']['complete_three_piece_paths_synthetic']['total'] for d in public['directions'].values()
                 for z in d['test_seeds'].values())
    gate=bool(new>orig and new_path>old_path and new_err<=old_err)
    public['aggregate_prespecified_gate']={'old_synthetic_correct_joins':orig,
            'new_synthetic_correct_joins':new,'old_synthetic_noncorrect_joins':old_err,
            'new_synthetic_noncorrect_joins':new_err,
            'old_synthetic_complete_paths':old_path,'new_synthetic_complete_paths':new_path,
            'experimental_passes_synthetic_only':gate,
            'authorize_native_H50Z_physics_auto_joins':False,
            'default_detector_unchanged':'s39_original'}
    return public


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--private-dir',type=Path)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Frozen runs must never overwrite existing results')
    result=run(args.visualmetro,args.photo_tiles,smoke=args.smoke,private_dir=args.private_dir)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    print('S64 wrote',args.out,'synthetic gate',result['aggregate_prespecified_gate'],flush=True)
if __name__=='__main__':main()
