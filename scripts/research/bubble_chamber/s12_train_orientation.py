#!/usr/bin/env python3
"""S12 AZ geometry-gradient supervision, frozen data and matched U-Net controls.

The original AZConv2d kernel remains byte-for-byte unchanged. The additional
training term supervises learned local orientation using a structure tensor
computed from the GENERATED target mask, not real camera labels.
"""
from __future__ import annotations

import argparse,json,random
from dataclasses import asdict,dataclass
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber import s7_original_retrain as s7
from scripts.research.bubble_chamber.s11_train_axial_topology import prepare_source
from bubble_chamber.s11_axial_topology import (cldice_loss,per_frame_confusion,count_metrics,binary_cldice,paired_frame_bootstrap)
from bubble_chamber.s12_orientation_field import az_orientation_field_loss

@dataclass(frozen=True)
class Arm:
    name:str
    architecture:str
    cldice_weight:float=0.
    direct_orientation_weight:float=0.

ARMS=(
    Arm('unet_base','plain_unet'),
    Arm('unet_cldice05','plain_unet',.05),
    Arm('az_base','original_azunet'),
    Arm('az_direct05','original_azunet',0.,.05),
    Arm('az_direct_cldice05','original_azunet',.05,.05),
)


def run(cfg:s7.RunConfig,measurements:Path|None,output:Path,*,arms:tuple[str,...]):
    s7.verify_sources(ROOT)
    torch.set_num_threads(cfg.cpu_threads)
    output.mkdir(parents=True,exist_ok=True)
    rows,metadata=prepare_source(cfg,measurements)
    fingerprints=s7.split_fingerprints(rows)
    test_groups=s7.TrackDataset(rows['test']).groups()
    original_loss=s7.masked_loss
    results={}
    try:
        for arm in ARMS:
            if arm.name not in arms:continue
            random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
            model=s7.build_models(cfg)[arm.architecture]
            def combined(logits,y,*,_arm=arm,_model=model):
                base=original_loss(logits,y)
                if _arm.cldice_weight:
                    base=base+_arm.cldice_weight*cldice_loss(logits,y,iterations=4)
                if _arm.direct_orientation_weight:
                    aligned,info=az_orientation_field_loss(_model,y)
                    if info['valid_stages']==0 and bool((y>0).any()):
                        raise RuntimeError('No learnable AZ local axes received valid supervision')
                    base=base+_arm.direct_orientation_weight*aligned
                return base
            s7.masked_loss=combined
            result,clean,stress=s7.train_one(arm.name,model,rows,cfg,output,data_fingerprints=fingerprints)
            evals={}
            for name,(pred,truth) in [('clean',clean),('stress',stress)]:
                count=per_frame_confusion(pred,truth,test_groups)
                d=np.array([binary_cldice(p,t) for p,t in zip(pred,truth)])
                evals[name]={'counts':count_metrics(count),'macro_cldice':float(d.mean()),
                             'cldice_per_image':d.tolist(),'source_frames':test_groups,
                             'per_frame_counts':{k:v.tolist() for k,v in count.items()}}
            results[arm.name]={'setting':asdict(arm),'train_result':result,'test':evals}
            (output/arm.name/'s12_config.json').write_text(json.dumps({'setting':asdict(arm),
               'source_blobs':s7.SOURCE_BLOBS,'fingerprints':fingerprints,
               'real_pixel_truth':False,'physical_particle_ID_truth':False},indent=2)+'\n')
    finally:
        s7.masked_loss=original_loss
    comparisons={}
    for condition in ('clean','stress'):
        for cand,base in [('az_direct05','az_base'),('az_direct_cldice05','az_base'),
                          ('az_direct_cldice05','az_direct05'),('az_direct_cldice05','unet_cldice05'),
                          ('unet_cldice05','unet_base')]:
            if cand not in results or base not in results:continue
            def frame_counts(arm):
                x=results[arm]['test'][condition]['per_frame_counts']
                return {k:np.asarray(v,np.int64) for k,v in x.items()}
            comparisons[f'{condition}_{cand}_minus_{base}']=paired_frame_bootstrap(
                frame_counts(base),frame_counts(cand),seed=s7.effective_data_seed(cfg)+74)
    report={'schema':'S12_DIRECT_GT_TANGENT_TO_AUTHENTIC_AZCONV_V1',
        'source_git_blobs':s7.SOURCE_BLOBS,'method':'structure_tensor_double_angle_gradient_to_original_AZConv',
        'source':metadata,'data_fingerprints':fingerprints,'config':asdict(cfg),
        'registered_arms':[asdict(x) for x in ARMS],'results':results,'comparisons':comparisons,
        'scientific_status':'SYNTHETIC_PIXEL_TARGETS_FROM_EXPERT_GEOMETRY_NOT_REAL_PHOTOGRAPH_SEGMENTATION',
        'auto_particle_merge':False,'real_photo_dice':None,
        'limitations':['Same physical film 070; no independent filling/run validation',
                       'Dense masks, contrast and noise are artificially generated',
                       'No instance-level false bridge labels or mass/momentum measurements',
                       'Method selected after exploratory S11; not a pre-registered independent physical study']}
    (output/'s12_metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    for name,x in results.items():print(name,'clean',x['test']['clean']['counts']['dice'],
                              'stress',x['test']['stress']['counts']['dice'],flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',choices=('visualmetro','procedural'),default='visualmetro')
    p.add_argument('--measurements',type=Path)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--data-seed',type=int,default=20261027)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--size',type=int,default=64)
    p.add_argument('--train-limit',type=int,default=120)
    p.add_argument('--val-limit',type=int,default=26)
    p.add_argument('--test-limit',type=int,default=25)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--cpu-threads',type=int,default=3)
    p.add_argument('--arms',default=','.join(x.name for x in ARMS))
    args=p.parse_args()
    armids=tuple(x.strip() for x in args.arms.split(','))
    if len(set(armids))!=len(armids) or any(x not in {y.name for y in ARMS} for x in armids):p.error('invalid arm')
    cfg=s7.RunConfig(source=args.source,seed=args.seed,data_seed=args.data_seed,
         epochs=args.epochs,size=args.size,batch_size=args.batch_size,
         widths=(4,6,8,12),num_rules=2,cpu_threads=args.cpu_threads,
         train_limit=args.train_limit,val_limit=args.val_limit,test_limit=args.test_limit,
         az_residual=True,robust_train=True)
    run(cfg,args.measurements,args.out_dir,arms=armids)
if __name__=='__main__':main()
