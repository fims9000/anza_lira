#!/usr/bin/env python3
"""S39 held-out injected pixel detection; do not call synthetic strokes native GT.

All methods get the same fixed top-3.5% pixel budget. Photo 005/006 are one
physical run and cannot establish run-disjoint generalization. VisualMetro
frame sets are disjoint (no identical trajectory in pretrain and holdout).
"""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from scipy.stats import rankdata
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s39_machine_auto import (ridge_evidence,contrast_baseline,
    canny_score,fixed_budget_mask)
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_simulated_gaps import frame_split,extract_track_templates,make_scene
from scripts.research.bubble_chamber.s35_benchmark_fragment_links import load_model
from scripts.research.bubble_chamber.s39_run_machine_only import predict_cross_photo

EXPECTED_FRACTION=.035
SEEDS={'005':20261127,'006':20261128}


def _rank_score(a:np.ndarray)->np.ndarray:
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('Invalid detector map')
    return rankdata(a,method='average').reshape(a.shape).astype('float32')/float(a.size)


def detection_scores(gray,az_map):
    ridge=ridge_evidence(gray)['score']
    contrast=contrast_baseline(gray)
    gradient=canny_score(gray)
    az=az_map.astype('float32')
    return {'multiscale_ridge':ridge,'local_darkness':contrast,
            'sobel_edge':gradient,'original_AZUNet_cross_photo':az,
            'ridge_plus_AZUNet_rank_fusion':.5*(_rank_score(ridge)+_rank_score(az)),
            'ridge_plus_local_contrast_rank_fusion':.5*(_rank_score(ridge)+_rank_score(contrast))}


def tolerant_fraction(detected:np.ndarray, region:np.ndarray, *,radius=2)->float:
    if region.dtype!=bool or detected.dtype!=bool or detected.shape!=region.shape:raise ValueError('Bad mask input')
    if not np.any(region):raise ValueError('Empty synthetic known region')
    expansion=cv2.dilate(detected.astype('uint8'),np.ones((radius*2+1,radius*2+1),dtype='uint8'))>0
    return float(np.mean(expansion[region]))


def run(visualmetro:Path,tiles:Path,model_dir:Path,output:Path,*,n=40):
    if n<12:raise ValueError('Need multiple held-out scenes per direction')
    torch.set_num_threads(2)
    manifest=audited_large_manifest(visualmetro)
    train_frames,test_frames=frame_split(manifest)
    if len(train_frames|test_frames)!=145 or train_frames&test_frames:
        raise AssertionError('Frame-disjoint track geometry audit failed')
    templates=extract_track_templates(manifest,chosen_frames=test_frames)
    out=[]
    for test_photo in ('005','006'):
        train_photo='006' if test_photo=='005' else '005'
        model,ck=load_model(model_dir,train_photo)
        source_files=sorted(tiles.glob(f'H50Z_030_01_{test_photo}_*.png'))
        if len(source_files)!=(21 if test_photo=='005' else 18):raise ValueError('Wrong H50Z tile source count')
        rng=np.random.default_rng(SEEDS[test_photo])
        for i in range(n):
            photo=source_files[int(rng.integers(len(source_files)))]
            scenario=('parallel','crossing','scratched','empty_gap')[i%4]
            scene=make_scene(photo,templates,seed=int(rng.integers(0,2**31)),scenario=scenario)
            if set(scene.source_frame_ids)&train_frames:raise ValueError('Source frame leaked')
            az=predict_cross_photo(scene.gray,model,step=64)
            scores=detection_scores(scene.gray,az)
            ground=scene.extra_masks['injected_visible_strokes'].astype(bool)
            decoy=scene.extra_masks['injected_false_decoy'].astype(bool)
            for arm,arr in scores.items():
                candidate=fixed_budget_mask(arr,fraction=EXPECTED_FRACTION)
                out.append({'test_photo':test_photo,'source_AZ_training_photo':train_photo,
                    'source_film':'070','synthetic_scene':i,'scenario':scenario,'arm':arm,
                    'synthetic_true_injection_tolerant_coverage':tolerant_fraction(candidate,ground),
                    'injected_decoy_scratch_tolerant_coverage':tolerant_fraction(candidate,decoy),
                    'activated_pixel_fraction':float(candidate.mean()),
                    'native_background_is_unlabeled':True})
        print('S39 synthetic scenes tested',test_photo,n,flush=True)
    arms=sorted({r['arm'] for r in out})
    # All CIs are *simulation-scene bootstrap* (one physical film+run); they
    # must not be presented as physical-domain confidence intervals.
    rng=np.random.default_rng(20261025)
    results={}
    for test_photo in ('005','006'):
        rows=defaultdict(list)
        for r in out:
            if r['test_photo']==test_photo:rows[r['arm']].append(r)
        metrics={}
        for name in arms:
            data=rows[name]
            hit=np.array([x['synthetic_true_injection_tolerant_coverage'] for x in data])
            dec=np.array([x['injected_decoy_scratch_tolerant_coverage'] for x in data])
            budget=np.array([x['activated_pixel_fraction'] for x in data])
            indices=rng.integers(0,len(hit),size=(2000,len(hit)))
            bs=hit[indices].mean(1)
            metrics[name]={'injected_track_hit_tolerance_2px':float(hit.mean()),
                           'scene_bootstrap_95pct_CI_SYNTHETIC_ONLY':np.quantile(bs,[.025,.975]).tolist(),
                           'injected_scratch_activation_tolerance_2px':float(dec.mean()),
                           'activated_pixel_fraction':float(budget.mean()),
                           'n_synthetic_scenes':len(hit)}
        results[f'{test_photo}_test_photo']=metrics
    report={'schema':'s39_injected_mask_detection_public_v1',
        'date':'2026-10-25','true_masks':'injected dark strokes only, unrelated to real H50Z tracks',
        'native_photo_track_GT_count':0,
        'source_visualMetro_complete_frames':len(manifest['frames']),
        'source_heldout_visualMetro_frame_count':len(test_frames),
        'source_visualMetro_train_frame_count':len(train_frames),
        'source_photo_count':2,'independent_physical_runs':1,
        'photo_disjoint_AZ_checkpoint_to_test_photo':True,
        'shared_activation_budget':EXPECTED_FRACTION,
        'predeclared_test_seeds':SEEDS,
        'synthetic_scenes_per_photo':n,
        'comparators':arms,
        'CI_warning':'Scene bootstrap describes simulator variability, not physical run-disjoint uncertainty.',
        'no_real_physics_precision_or_recall_can_be_claimed':True,
        'results':results}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visualmetro-extracted',type=Path,required=True)
    p.add_argument('--photo-tiles',type=Path,required=True)
    p.add_argument('--model-checkpoints',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenes-per-photo',type=int,default=40)
    a=p.parse_args();run(a.visualmetro_extracted,a.photo_tiles,a.model_checkpoints,a.output,n=a.scenes_per_photo)
if __name__=='__main__':main()
