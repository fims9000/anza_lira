#!/usr/bin/env python3
"""S13 matched shallow AZConv guide versus matched ordinary convolution guide.

Original `models/azconv.py` and `models/segmentation.py` must be unchanged.
Expert visualMetro line geometry, but all 2D pixels/targets are synthetic.
"""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber import s7_original_retrain as s7
from scripts.research.bubble_chamber.s11_train_axial_topology import prepare_source
from bubble_chamber.s11_axial_topology import per_frame_confusion,count_metrics,binary_cldice
from bubble_chamber.s13_hybrid_az import instantiate_matched,GuidedUNet

ARMS=('unet_base','conv_guided','az_guided','az_pure')


def run(cfg:s7.RunConfig, measurements:Path,output:Path, *,arms=ARMS):
    s7.verify_sources(ROOT)
    torch.set_num_threads(cfg.cpu_threads)
    rows,metadata=prepare_source(cfg,measurements)
    fingers=s7.split_fingerprints(rows)
    output.mkdir(parents=True,exist_ok=True)
    groups=s7.TrackDataset(rows['test']).groups()
    results={}
    for name in ARMS:
        if name not in arms:continue
        random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
        if name=='az_pure':
            model=s7.build_models(cfg)['original_azunet']
        else:
            model=instantiate_matched(cfg.seed,widths=cfg.widths,rules=cfg.num_rules)[name]
        run_data,clean,stress=s7.train_one(name,model,rows,cfg,output,data_fingerprints=fingers)
        metrics={}
        for label,(pred,truth) in (('clean',clean),('stress',stress)):
            by_frame=per_frame_confusion(pred,truth,groups)
            metrics[label]={'counts':count_metrics(by_frame),
                            'macro_cldice':float(np.mean([binary_cldice(x,y) for x,y in zip(pred,truth)])),
                            'per_frame_counts':{k:v.tolist() for k,v in by_frame.items()},
                            'source_frames':groups}
        results[name]={'train':run_data,'test':metrics,
                       'gate':model.learned_guide_gate() if isinstance(model,GuidedUNet) else None}
        print(name,'clean',metrics['clean']['counts']['dice'],
              'stress',metrics['stress']['counts']['dice'],'gate',results[name]['gate'],flush=True)
    report={'schema':'S13_ORIGINAL_AZ_GUIDED_UNET_AND_CONTROLS_V1',
        'data_seed':s7.effective_data_seed(cfg),'config':asdict(cfg),
        'exact_source_blob_sha':s7.SOURCE_BLOBS,'source':metadata,
        'data_fingerprints':fingers,'results':results,'registered_arms':ARMS,
        'scientific_limitations':['generated grayscale pixels only, NOT authentic BMP',
            'one film 070, no independent fill/run',
            'capacity differs across models; conv-guided control added',
            'no true particle identity/charge, mass or momentum known',
            'false joining is not measured by Dice alone',
            'if selected after S12 testing this protocol is exploratory, not confirmatory']}
    (output/'s13_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--data-seed',type=int,default=20261031)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--size',type=int,default=64)
    p.add_argument('--train-limit',type=int,default=120)
    p.add_argument('--val-limit',type=int,default=26)
    p.add_argument('--test-limit',type=int,default=25)
    p.add_argument('--cpu-threads',type=int,default=3)
    p.add_argument('--arms',default=','.join(ARMS))
    args=p.parse_args()
    selected=tuple(x.strip() for x in args.arms.split(','))
    if not selected or len(set(selected))!=len(selected) or any(x not in ARMS for x in selected):p.error('Invalid model arms')
    cfg=s7.RunConfig(source='visualmetro',seed=args.seed,data_seed=args.data_seed,
        epochs=args.epochs,batch_size=args.batch_size,size=args.size,
        train_limit=args.train_limit,val_limit=args.val_limit,test_limit=args.test_limit,
        widths=(4,6,8,12),num_rules=2,cpu_threads=args.cpu_threads,
        az_residual=True,robust_train=True)
    run(cfg,args.measurements,args.out_dir,arms=selected)
if __name__=='__main__':main()
