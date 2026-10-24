#!/usr/bin/env python3
"""Strictly synthetic held-out evaluation of S17 physics-book pretraining.

Does not use a real film in model fitting or accuracy calculation. Threshold
0.5 is fixed before test, so this is diagnostic, NOT a benchmark of real-film.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.s14_train_sparse_anza import build_model,verify_model_source


def _metric(prob:np.ndarray,truth:np.ndarray,threshold:float=.5)->dict:
    pred=np.asarray(prob)>=threshold;real=np.asarray(truth)>0
    if pred.shape!=real.shape:raise ValueError('misaligned prediction and target')
    tp=int((pred&real).sum());fp=int((pred&~real).sum());fn=int((~pred&real).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'dice':round(float(2*tp/max(1,2*tp+fp+fn)),6),
            'precision':round(tp/max(1,tp+fp),6),'recall':round(tp/max(1,tp+fn),6)}


def evaluate(data_dir:Path, model_dirs:dict[str,Path],out_path:Path)->dict:
    model_shas=verify_model_source()
    root=Path(data_dir)
    pairs=json.loads((root/'s17_book_test_truth_only.json').read_text())['examples']
    if not pairs:raise ValueError('empty independent synthetic test set')
    y=[];x=[]
    for p in pairs:
        im=cv2.imread(str(root/p['image']),cv2.IMREAD_GRAYSCALE)
        gt=cv2.imread(str(root/p['visible_gt']),cv2.IMREAD_GRAYSCALE)
        if im is None or gt is None or im.shape!=gt.shape:
            raise ValueError('invalid synthetic test pair')
        x.append(im);y.append((gt>0).astype(np.uint8))
    report={'schema':'s17_book_synthetic_eval_v1','source':'ONLY artificially rendered stopped hadroida curves',
            'threshold':0.5,'test_images':len(pairs),'source_models_sha':model_shas,
            'heldout_real_BMPs':0,'expert_pixel_masks':0,'real_photo_accuracy':'UNKNOWN',
            'comparison_is_statistical_proof':False,'models':{}}
    torch.set_num_threads(2)
    for name,folder in model_dirs.items():
        checkpoint=torch.load(Path(folder)/'best.pt',map_location='cpu',weights_only=False)
        model=build_model(checkpoint['architecture'],checkpoint['architecture_widths'],checkpoint['rules'])
        model.load_state_dict(checkpoint['model_weights']);model.eval()
        scores=[]
        with torch.no_grad():
            for im in x:
                inp=torch.from_numpy(((255-im).astype(np.float32)/255)[None,None])
                p=torch.sigmoid(model(inp)).cpu().numpy()[0,0]
                scores.append(p)
        pred=np.stack(scores);truth=np.stack(y)
        stats=_metric(pred,truth)
        stats['predicted_positive_fraction']=round(float((pred>=.5).mean()),6)
        stats['truth_positive_fraction']=round(float(truth.mean()),6)
        stats['source_train_pixel_labels']='synthetic 1/0/255, NOT HUMAN EXPERT despite shared trainer API'
        report['models'][name]=stats
    out_path.parent.mkdir(parents=True,exist_ok=True)
    out_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--az-checkpoint-dir',type=Path,required=True)
    p.add_argument('--unet-checkpoint-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(evaluate(args.data_dir,{'azunet':args.az_checkpoint_dir,'unet':args.unet_checkpoint_dir},args.out),indent=2))

if __name__=='__main__':main()
