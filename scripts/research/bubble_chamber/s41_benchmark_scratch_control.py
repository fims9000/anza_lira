#!/usr/bin/env python3
"""Fixed S41 synthetic-origin scratch control (NOT native track truth)."""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.large_archive import audited_large_manifest
from bubble_chamber.s35_simulated_gaps import frame_split,extract_track_templates,make_scene
from bubble_chamber.s41_machine_quality import ridge_support_maps,candidate_stability

SEEDS={'005':2026102605,'006':2026102606}
SCENARIOS=('parallel','crossing','scratched','empty_gap')


def _auc_from_4_true_and_1_scratch(true:list[float],scratch:float)->float:
    return float(np.mean([(v>scratch)+.5*(v==scratch) for v in true]))


def run(visualmetro:Path,tiles:Path,out:Path,*,scenes=28)->dict:
    if scenes<8 or scenes%4:raise ValueError('Use >=8 scenes divisible by 4')
    dataset=audited_large_manifest(visualmetro)
    train,test=frame_split(dataset)
    if len(test)!=36 or train & test:raise AssertionError('Source frame leakage')
    templates=extract_track_templates(dataset,chosen_frames=test)
    detailed=[]
    for photo in ('005','006'):
        sources=sorted(tiles.glob(f'H50Z_030_01_{photo}_*.png'))
        if len(sources)!=(21 if photo=='005' else 18):raise ValueError('Unexpected private photo crops')
        rng=np.random.default_rng(SEEDS[photo])
        for i in range(scenes):
            scene=make_scene(sources[int(rng.integers(len(sources)))],templates,
                             seed=int(rng.integers(0,2**31)),scenario=SCENARIOS[i%4])
            if set(scene.source_frame_ids)&train:raise AssertionError('Measured track source leakage')
            maps=ridge_support_maps(scene.gray)
            t=[candidate_stability(p,maps) for p in [*scene.left,*scene.right[:2]]]
            decoy=candidate_stability(scene.right[2],maps)
            baseline=[r['pixel_ridge_support_fraction_by_view']['identity'] for r in t]
            stable=[r['all_views_min_support'] for r in t]
            base_neg=decoy['pixel_ridge_support_fraction_by_view']['identity']
            stable_neg=decoy['all_views_min_support']
            detailed.append({'photo':photo,'scene':i,'scenario':scene.scenario,
                'baseline_true_mean':np.mean(baseline).item(),
                'baseline_scratch':base_neg,
                'worst_true_mean':np.mean(stable).item(),
                'worst_scratch':stable_neg,
                'baseline_AUC_track_vs_decoy_SIMULATED':_auc_from_4_true_and_1_scratch(baseline,base_neg),
                'worst_view_AUC_track_vs_decoy_SIMULATED':_auc_from_4_true_and_1_scratch(stable,stable_neg),
                'true_accepted':int(sum(r['photometric_machine_stable_NOT_GT'] for r in t)),
                'scratch_accepted':int(decoy['photometric_machine_stable_NOT_GT'])})
        print(f'S41 scratch injection: {photo}: {scenes} scenes',flush=True)
    results={}
    ci_rng=np.random.default_rng(20261026)
    for photo in ('005','006'):
        selected=[d for d in detailed if d['photo']==photo]
        auc0=np.asarray([x['baseline_AUC_track_vs_decoy_SIMULATED'] for x in selected])
        auc1=np.asarray([x['worst_view_AUC_track_vs_decoy_SIMULATED'] for x in selected])
        indices=ci_rng.integers(0,len(selected),(2000,len(selected)))
        delta=auc1-auc0
        results[photo]={
            'n_scenes':len(selected),'simulated_true_line_sources':4*len(selected),
            'simulated_scratch_sources':len(selected),
            'original_detector_object_AUC_SIMULATED':round(float(auc0.mean()),5),
            'worst_view_object_AUC_SIMULATED':round(float(auc1.mean()),5),
            'worst_minus_original_AUC':round(float(delta.mean()),5),
            'delta_95pct_scene_bootstrap_CI_SIMULATOR_ONLY':np.quantile(delta[indices].mean(axis=1),[.025,.975]).round(5).tolist(),
            'original_ridge_track_centerline_support':round(float(np.mean([x['baseline_true_mean'] for x in selected])),5),
            'original_ridge_scratch_centerline_support':round(float(np.mean([x['baseline_scratch'] for x in selected])),5),
            'min_view_track_centerline_support':round(float(np.mean([x['worst_true_mean'] for x in selected])),5),
            'min_view_scratch_centerline_support':round(float(np.mean([x['worst_scratch'] for x in selected])),5),
            'track_acceptance_heuristic_fraction':round(sum(x['true_accepted'] for x in selected)/(4*len(selected)),5),
            'scratch_acceptance_heuristic_fraction':round(sum(x['scratch_accepted'] for x in selected)/len(selected),5),
        }
    report={'schema':'s41_scratch_origin_control_v1','source_visualMetro_frame_count':145,
        'heldout_source_frames':36,'source_photo_count':2,'independent_physical_runs':1,
        'real_physics_track_GT_labels':0,
        'simulator_note':'The synthetic decoy is drawn with the same stroke generator as the measured-curve injections, and therefore looks like a track by design; scratch class is synthetic origin only.',
        'no_human_labels_used':True,'seeds':SEEDS,'scenarios':SCENARIOS,
        'photo_results':results,'no_native_precision_or_recall_claim':True}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def cli():
    ap=argparse.ArgumentParser()
    ap.add_argument('--visualmetro',type=Path,required=True)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--scenes',type=int,default=28)
    a=ap.parse_args()
    result=run(a.visualmetro,a.tiles,a.output,scenes=a.scenes)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':cli()
