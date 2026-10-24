#!/usr/bin/env python3
"""Post-hoc curve extraction from FROZEN S12 original-model checkpoints.

The source is expert click geometry rendered as SIMULATED pixels/masks.
No physical mass, magnetic field, true particle identity or real-BMP accuracy
can be inferred from the 2D pixel parabola. Broken/ambiguous tracks abstain.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber import s7_original_retrain as s7
from bubble_chamber.s12_curve_geometry import fit_single_track, sample_fitted_curve, evaluate_curves
from bubble_chamber.s11_axial_topology import per_frame_confusion


def _source_hash(path):
    sha=hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda:fh.read(2**20),b''):sha.update(chunk)
    return sha.hexdigest()


def evaluate(checkpoint_dir:Path, measurements:Path,output:Path,*,model_names:list[str],examples_to_save:int=4):
    s7.verify_sources(ROOT)
    protocol=json.loads((checkpoint_dir/'s12_metrics.json').read_text())
    cfg=s7.RunConfig(**protocol['config'])
    torch.set_num_threads(max(1,cfg.cpu_threads))
    if cfg.source!='visualmetro': raise ValueError('Only real expert clicks source (with synthetic pixels) is assessed')
    samples,meta=s7.make_datasets(cfg,measurements)
    if s7.split_fingerprints(samples)!=protocol['data_fingerprints']:
        raise RuntimeError('Dataset fingerprint mismatch: refuse to evaluate')
    test=s7.TrackDataset(samples['test'])
    groups=test.groups()
    output.mkdir(exist_ok=True,parents=True)
    findings={}
    for name in model_names:
        if name not in protocol['results']:raise KeyError(f'model {name} not trained')
        setting=protocol['results'][name]['setting']
        model=s7.build_models(cfg)[setting['architecture']]
        ck=checkpoint_dir/name/'best_final.pt'
        saved=torch.load(ck,map_location='cpu',weights_only=False)
        if saved.get('source_blobs')!=s7.SOURCE_BLOBS or saved.get('data_fingerprints')!=protocol['data_fingerprints']:
            raise RuntimeError('Checkpoint provenance mismatch')
        model.load_state_dict(saved['state_dict'])
        threshold=float(saved['validation_selected_threshold'])
        per_arm={'checkpoint_sha256':_source_hash(ck),'validation_threshold':threshold}
        for condition in ('clean','stress'):
            probability,truth=s7.predict_probabilities(model,test,cfg,stress=condition=='stress')
            masks=probability>=threshold
            true=truth>=.5
            result=evaluate_curves(masks,true)
            per_arm[condition]=result
            if condition=='clean' and examples_to_save:
                for i in range(min(examples_to_save,len(masks))):
                    pred_fit=fit_single_track(masks[i])
                    real=np.uint8(np.clip(samples['test'][i][0]*255,0,255))
                    rgb=cv2.cvtColor(real,cv2.COLOR_GRAY2BGR)
                    rgb[true[i],:]=[40,140,220]
                    rgb[masks[i],:]=[90,240,100]
                    if pred_fit.accepted_for_review:
                        curve=sample_fitted_curve(pred_fit,80)
                        coords=np.round(curve).astype(np.int32)
                        cv2.polylines(rgb,[coords],False,(255,0,180),1,cv2.LINE_AA)
                    cv2.imwrite(str(output/f'{name}_example_{i:02d}_{pred_fit.status}.png'),rgb)
        findings[name]=per_arm
    report={'schema':'S12_REVIEW_ONLY_2D_QUADRATIC_FIT_V1', 'sample_frames':groups,
            'input_source':meta,'model_seed':cfg.seed,'data_seed':s7.effective_data_seed(cfg),
            'frozen_training_result':str(checkpoint_dir.name),'models':findings,
            'gates':{'test_labels_used_in_model_training':False,'manual_real_image_truth':False,'automatic_physical_merges':False,
                     'true_particle_identity_inferred':False,'magnetic_field_used':False},
            'interpretation':'A fitted 2D polynomial is a geometry review aid, NOT a physical particle trajectory.'}
    (output/'curve_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--training-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--models',default='unet_base,az_base,az_direct05,az_direct_cldice05')
    args=p.parse_args()
    report=evaluate(args.training_dir,args.measurements,args.out_dir,model_names=args.models.split(','))
    for name,x in report['models'].items():
        print(name, 'eligible',x['clean']['fitted_pairs'],'clean error px',x['clean']['median_endpoint_error_px'],
              'stress eligible',x['stress']['fitted_pairs'],'stress error px',x['stress']['median_endpoint_error_px'])

if __name__=='__main__':main()
