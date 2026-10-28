#!/usr/bin/env python3
"""S85 public integration benchmark: detector swaps + solver ablations.

ALL data here are generated synthetically; neither original H50Z photographs
nor film-070 measured templates enter this file. This is a component benchmark,
not evidence of real-world particle-tracking performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import cv2
import numpy as np

from bubble_chamber.interop_api import ScoreBinding,assemble_lines
from bubble_chamber.interop_detectors import (
    HessianRidgeDetector, OpenCVLSDDetector, OpenCVHoughDetector)
from bubble_chamber.s56_three_piece_scenes import SCENARIOS,make_three_piece_scene
from bubble_chamber.s56_global_path_assembly import (
    pair_rows,scored_edges,select_port_greedy,select_node_mutual)
from bubble_chamber.s59_candidate_retrieval import filter_rows
from bubble_chamber.s85_image_disturbances import condition_pixels
from scripts.research.bubble_chamber.s80_calibrate_then_test_multiscale import score_after_inference

EXPERIMENT_ID='ANZA-LIRA-BC-S85-INTEROP-20261028'
PROTOCOL=ROOT/'docs/bubble_chamber/S85_INTEROP_COMPARISON_PROTOCOL_2026_10_28.md'
SEED=20268588
SCENES=20
CONDITIONS=('clean','blur_sigma_11','scratch_grain')
CUT_OFF=.70  # uncalibrated toy-geometry score; never production S56 policy
MARGIN=.04
DETECTOR_CLASSES=(HessianRidgeDetector,OpenCVLSDDetector,OpenCVHoughDetector)
SOLVERS=('none','node_mutual','port_greedy','global_milp')
METRICS=('selected_synthetic_correct_links','synthetic_noncorrect_selected',
         'complete_three_piece_paths_synthetic','shape_true_fragments_matched',
         'solver_failure','candidate_polylines','candidate_pairs_after_strict_gate')


def score_geometry_for_QA(rows:list[dict]) -> np.ndarray:
    """Fixed geometry-only heuristic; NOT a calibrated trained probability.

    Identical parameters across detectors to reveal the *need* for recalibration.
    Never deploy this scorer as if it were the source-trained S56 model.
    """
    if not rows:return np.array([],float)
    gap=np.array([x['endpoint_gap_px'] for x in rows])
    a=np.array([x['outgoing_tangent_A_to_gap'] for x in rows])
    b=np.array([x['outgoing_tangent_B_to_gap'] for x in rows])
    mutual=np.array([x['outgoing_tangent_mutual'] for x in rows])
    curve=np.array([x['chord_curvature_consistency'] for x in rows])
    return np.clip(.40+.20*a+.20*b+.12*mutual+.08*curve-.10*gap/36.,0,1)


def make_fake_templates():
    """Independent generated sine curves, NOT visualMetro film-070 templates."""
    templates=[]
    for i in range(28):
        t=np.linspace(0,1,64)
        points=np.column_stack((40+720*t,200+7*np.sin(2*np.pi*t+i*.27)))
        templates.append({'frame':'GENERATED_NOT_FILM070','points':points,'id':f'toy{i}'})
    return templates


def write_fake_backgrounds(directory:Path):
    """2 synthetic 512x512 8-bit images; never native H50Z photographs."""
    sources=[]
    y,x=np.indices((512,512))
    for i in range(2):
        rng=np.random.default_rng(SEED+300+i)
        texture=rng.normal(0,3.5+i*1.2,size=(512,512))
        gray=np.clip(205+8*np.sin(x/(40+i*7))+6*np.cos(y/31)+texture,0,255).astype('uint8')
        path=directory/f'GENERATED_NOT_H50Z_{i}.png'
        if not cv2.imwrite(str(path),gray):raise OSError('Could not generate toy backgrounds')
        sources.append(path)
    return sources


def _score_scene(lines,gray,solver):
    """Complete pixel-only graph proposal, then evaluator runs separately."""
    pieces=list(lines)
    if solver=='global_milp':
        # API enforces the scorer/detector binding in actual integrations.
        result=assemble_lines(pieces,ScoreBinding('toy-uncalibrated','toy-input',
                             score_geometry_for_QA,CUT_OFF,MARGIN,
                             'TOY_NO_SOURCE_CAL'),detector_id='toy-input',solver_timeout=1.)
        from bubble_chamber.s56_global_path_assembly import Edge
        edges=[Edge(e.source_line,e.target_line,e.source_end,e.target_end,e.score)
               for e in result.joins]
        pairs=result.pairs_considered
        solver_success=result.solver_success
    else:
        edges=[];pairs=0;solver_success=True
        if len(pieces)>=3:
            rows=filter_rows(pair_rows(pieces),'strict_s39')
            pairs=len(rows)
            if solver!='none' and rows:
                scored=scored_edges(rows,score_geometry_for_QA(rows))
                if solver=='port_greedy':edges=select_port_greedy(scored,threshold=CUT_OFF,margin=MARGIN)
                elif solver=='node_mutual':edges=select_node_mutual(scored,threshold=CUT_OFF,margin=MARGIN)
                else:raise ValueError('Unknown fixed solver ablation')
    return {'polylines':pieces,'chosen':edges,'pair_count':pairs,
            'solver_success':solver_success,'physics_auto_joins_authorized':False,
            'added_pixel_instances':0,'duplicate_suppressed':0,
            'scale_consensus_rejected':0,'original_polylines_retained':len(pieces)}


def measure(scene,pred):
    """Only AFTER blind graph proposal; source nodes never enter _score_scene."""
    return {k:int(v) for k,v in score_after_inference({'result':pred},scene.nodes)['result'].items()
            if k in METRICS}


def aggregate(rows):
    return {**{k:int(sum(r[k] for r in rows)) for k in METRICS},'scenes':len(rows)}


def paired_delta(a,b,key='complete_three_piece_paths_synthetic'):
    rng=np.random.default_rng(20268599)
    dif=np.asarray([q[key]-p[key] for p,q in zip(a,b)],float)
    indices=rng.integers(0,len(dif),size=(1000,len(dif)))
    return {'delta_total':int(dif.sum()),'mean_delta_per_scene':float(dif.mean()),
            'scene_bootstrap_95pct':[float(v) for v in np.quantile(dif[indices].mean(1),(.025,.975))]}


def run(scenes=SCENES):
    if not PROTOCOL.is_file() or scenes<2 or scenes>SCENES:raise ValueError('Frozen S85 integration protocol/scene count needed')
    detectors={cls().detector_id:cls(max_lines=12) for cls in DETECTOR_CLASSES}
    names=[*detectors,'ORACLE_SYNTHETIC_EVALUATION_ONLY']
    assert len(detectors)==3
    observations={cond:{name:{s:[] for s in SOLVERS} for name in names} for cond in CONDITIONS}
    rng=np.random.default_rng(SEED)
    with tempfile.TemporaryDirectory(prefix='s85_interop_fake_') as temp:
        photos=write_fake_backgrounds(Path(temp))
        templates=make_fake_templates()
        for i in range(scenes):
            photo=photos[int(rng.integers(len(photos)))]
            scene=make_three_piece_scene(photo,templates,
                 seed=int(rng.integers(1,2**31-1)),scenario=SCENARIOS[i%len(SCENARIOS)])
            noise_seed=int(rng.integers(1,2**31-1))
            for cond in CONDITIONS:
                pix=condition_pixels(scene.gray,cond,noise_seed=noise_seed)
                # No reference nodes may enter any real image detector.
                extracted={}
                for name,detector in detectors.items():
                    extracted[name]=detector.detect(pix)
                # Evaluator-side oracle: makes a positive upper-bound control.
                extracted['ORACLE_SYNTHETIC_EVALUATION_ONLY']=scene.nodes
                for name,lines in extracted.items():
                    for solver in SOLVERS:
                        prediction=_score_scene(lines,pix,solver)
                        assessment=measure(scene,prediction)
                        observations[cond][name][solver].append(assessment)
    from scipy import __version__ as scipy_version
    from skimage import __version__ as skimage_version
    from sklearn import __version__ as sklearn_version
    locked_files=(
       'scripts/research/bubble_chamber/interop_comparison.py',
       'bubble_chamber/interop_api.py',
       'bubble_chamber/interop_detectors.py',
       'bubble_chamber/s56_global_path_assembly.py',
       'bubble_chamber/s59_candidate_retrieval.py',
       'bubble_chamber/s39_machine_auto.py',
       'bubble_chamber/s69_orientation_invariant_audit.py',
       'bubble_chamber/s76_shape_endpoint_diagnostics.py',
       'bubble_chamber/s85_image_disturbances.py')
    locks={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in locked_files}
    result={'schema':'S85_INTEROP_DETECTOR_SWAP_QA_V1',
            'experiment_id':EXPERIMENT_ID,'mode':'GENERATED_TOY_ONLY_NOT_H50Z',
            'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            'component_source_sha256':locks,
            'versions':{'python':'.'.join(map(str,sys.version_info[:3])),
                        'numpy':np.__version__,'opencv':cv2.__version__,
                        'scipy':scipy_version,'scikit_image':skimage_version,
                        'scikit_learn':sklearn_version},
            'source_not_film070':True,'image_backgrounds_not_H50Z':True,
            'source_train_calibration_per_detector':False,
            'shared_uncalibrated_heuristic':True,
            'cutoff_used_only_for_toy_diagnostic':CUT_OFF,'margin':MARGIN,
            'scenes_per_condition':scenes,'conditions':CONDITIONS,
            'smoke_NOT_CONFIRMATORY':scenes!=SCENES,
            'frozen_design_scenes_per_condition':SCENES,
            'methods':names,'solvers':SOLVERS,
            'physical_auto_joins_authorized':False,'native_GT_count':0,
            'results':{cond:{name:{solver:aggregate(observations[cond][name][solver]) for solver in SOLVERS}
                    for name in names} for cond,obs in observations.items()},
            'paired_global_vs_greedy_chain':{cond:{name:paired_delta(observations[cond][name]['port_greedy'],
                       observations[cond][name]['global_milp']) for name in names}
                       for cond in CONDITIONS}}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scenes',type=int,default=SCENES)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Immutable benchmark output already exists')
    result=run(a.scenes)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('Generated-only benchmark, not H50Z:',a.out)
    for cond in CONDITIONS:
        print(cond,[(name,result['results'][cond][name]['global_milp']['complete_three_piece_paths_synthetic']) for name in result['methods']])
if __name__=='__main__':main()
